"""Test blocking: embed test S1 and S2/S3 with the fine-tuned e5 blocker, FAISS IVF-Flat per country and source; forward top-10 per
source + reverse top-2 (top-1 kept, cap 10 per S1). Writes artifacts_test/fwd__{country} and rev__{country}."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import os, sys, time
import numpy as np
import polars as pl
import faiss
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import rd  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

T = f"{ER_ROOT}/data/test"
MODEL = f"{ER_ROOT}/models/blocker_e5"
OUT = "artifacts_test"; os.makedirs(OUT, exist_ok=True)
K_FWD, K_REV, NPROBE = 10, 2, 64
RES = faiss.StandardGpuResources()
T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
text = lambda df: ("query: " + df["business_name"] + " | " + df["business_address"]).to_list()


def ivf(X):
    nl = int(2 ** round(np.log2(4 * np.sqrt(len(X)))))
    idx = faiss.index_cpu_to_gpu(RES, 0, faiss.index_factory(X.shape[1], f"IVF{nl},Flat", faiss.METRIC_INNER_PRODUCT))
    idx.train(X[np.random.default_rng(0).choice(len(X), size=min(len(X), 50 * nl), replace=False)]); idx.add(X)
    faiss.GpuParameterSpace().set_index_parameter(idx, "nprobe", NPROBE)
    return idx


def search(idx, Q, k, chunk=500_000):
    Ds, Is = [], []
    for i in range(0, len(Q), chunk):
        D, I = idx.search(Q[i:i + chunk], k); Ds.append(D); Is.append(I)
    return np.concatenate(Ds), np.concatenate(Is)


s1 = rd(f"{T}/test_source1.tsv")
aux = pl.concat([rd(f"{T}/test_source2.tsv").with_columns(src=pl.lit("S2")), rd(f"{T}/test_source3.tsv").with_columns(src=pl.lit("S3"))])
model = SentenceTransformer(MODEL, device="cuda"); model.max_seq_length = 64; model.half()


def emb(df, path):
    if os.path.exists(path):
        return np.load(path)
    t = time.time()
    E = model.encode(text(df), batch_size=1024, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    np.save(path, E); log(f"  embedded {len(E):,} ({len(E)/(time.time()-t):,.0f}/s)")
    return E


for c in sorted(s1["country"].unique().to_list()):
    if os.path.exists(f"{OUT}/rev__{c}.parquet"):
        continue
    S1c, A = s1.filter(pl.col("country") == c), aux.filter(pl.col("country") == c)
    log(f"{c}: {S1c.height:,} S1, {A.height:,} aux")
    E_s1 = emb(S1c, f"{OUT}/emb_s1__{c}.npy").astype(np.float32)
    E_aux = emb(A, f"{OUT}/emb_aux__{c}.npy").astype(np.float32)
    parts = []
    for src in ["S2", "S3"]:
        m = (A["src"] == src).to_numpy(); ids = A.filter(pl.col("src") == src)["entity_id"]
        idx = ivf(E_aux[m]); D, I = search(idx, E_s1, K_FWD); del idx
        ok = I.ravel() >= 0
        parts.append(pl.DataFrame({"s1": np.repeat(S1c["entity_id"].to_numpy(), K_FWD)[ok], "a": ids.gather(I.ravel()[ok]),
                                   "rank": np.tile(np.arange(1, K_FWD + 1, dtype=np.int8), len(I))[ok], "score": D.ravel()[ok]}))
        log(f"  forward {src} done")
    pl.concat(parts).write_parquet(f"{OUT}/fwd__{c}.parquet")
    idx = ivf(E_s1); D, I = search(idx, E_aux, K_REV); del idx
    ok = I.ravel() >= 0
    pl.DataFrame({"a": np.repeat(A["entity_id"].to_numpy(), K_REV)[ok], "s1": S1c["entity_id"].gather(I.ravel()[ok]),
                  "rank": np.tile(np.arange(1, K_REV + 1, dtype=np.int8), len(I))[ok], "score": D.ravel()[ok]}).write_parquet(f"{OUT}/rev__{c}.parquet")
    log(f"  reverse done")
    del E_s1, E_aux; torch.cuda.empty_cache()
log("done")
