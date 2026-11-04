"""Dense blocking for all train S1: FAISS GPU IVF-Flat (nprobe 64). Forward top-10 per source (S2, S3) for every S1; reverse top-2 S1 for
every aux record (top-1 becomes a candidate, capped at 10 per S1; top-2 feeds the owner-margin feature)."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import os, sys, time
import numpy as np
import polars as pl
import faiss

sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load, COUNTRIES  # noqa: E402

EMB = f"{ER_ROOT}/work/blocker/artifacts/e5b_dev_v1"
OUT = "artifacts"; os.makedirs(OUT, exist_ok=True)
K_FWD, K_REV, NPROBE = 10, 2, 64
RES = faiss.StandardGpuResources()
T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


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


s1, aux, _, _ = load()
for c in COUNTRIES:
    if os.path.exists(f"{OUT}/rev__{c}.parquet"):
        continue
    S1c, A = s1.filter(pl.col("country") == c), aux.filter(pl.col("country") == c)   # same row order as the embedding files
    E_s1 = np.load(f"{EMB}/emb_s1__{c}.npy").astype(np.float32)
    E_aux = np.load(f"{EMB}/emb_aux__{c}.npy").astype(np.float32)
    parts = []
    for src in ["S2", "S3"]:
        m = (A["src"] == src).to_numpy(); ids = A.filter(pl.col("src") == src)["entity_id"]
        idx = ivf(E_aux[m]); D, I = search(idx, E_s1, K_FWD); del idx
        ok = I.ravel() >= 0
        parts.append(pl.DataFrame({"s1": np.repeat(S1c["entity_id"].to_numpy(), K_FWD)[ok], "a": ids.gather(I.ravel()[ok]),
                                   "rank": np.tile(np.arange(1, K_FWD + 1, dtype=np.int8), len(I))[ok], "score": D.ravel()[ok]}))
        log(f"forward {c} {src}: {S1c.height:,} queries x {m.sum():,} aux")
    pl.concat(parts).write_parquet(f"{OUT}/fwd__{c}.parquet")
    idx = ivf(E_s1); D, I = search(idx, E_aux, K_REV); del idx
    ok = I.ravel() >= 0
    pl.DataFrame({"a": np.repeat(A["entity_id"].to_numpy(), K_REV)[ok], "s1": S1c["entity_id"].gather(I.ravel()[ok]),
                  "rank": np.tile(np.arange(1, K_REV + 1, dtype=np.int8), len(I))[ok], "score": D.ravel()[ok]}).write_parquet(f"{OUT}/rev__{c}.parquet")
    log(f"reverse {c}: {A.height:,} aux -> {S1c.height:,} S1")
log("done")
