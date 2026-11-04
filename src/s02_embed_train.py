"""Embed all train S1 and S2/S3 records of one country with the fine-tuned e5 blocker (fp16 .npy files, used by s03_block_train.py)."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import argparse, os, sys, time, numpy as np, polars as pl, torch
sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--tag"); ap.add_argument("--country")
ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
OUT = f"artifacts/{args.tag}"; os.makedirs(OUT, exist_ok=True)
s1, aux, edges, split = load()
S1c = s1.filter(pl.col("country") == args.country)
A = aux.filter(pl.col("country") == args.country)
text = lambda df: ("query: " + df["business_name"] + " | " + df["business_address"]).to_list()
model = SentenceTransformer(args.model, device=args.device); model.max_seq_length = 64
model.half()

def emb(df, name):
    path = f"{OUT}/emb_{name}__{args.country}.npy"
    if os.path.exists(path):
        return np.load(path)
    t = time.time()
    E = model.encode(text(df), batch_size=1024, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
    np.save(path, E); print(f"embedded {name} {len(E):,} in {time.time()-t:.0f}s ({len(E)/(time.time()-t):,.0f}/s)", flush=True)
    return E

E_s1, E_aux = emb(S1c, "s1"), emb(A, "aux")
print(f"{args.country}: S1 {len(E_s1):,}, aux {len(E_aux):,} embeddings in {OUT}")
