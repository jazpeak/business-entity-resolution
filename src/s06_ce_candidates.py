"""Cross-encoder candidates: top-5 prefilter candidates per S1 with raw text, and the holdoutA sub-split by sha1(entity_id)[16:24] % 3:
A0 (cross-encoder training) / A1a (fusion training) / A1b (decoder tuning)."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import hashlib, os, sys
import polars as pl

sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load

K = int(os.environ.get("CE_TOPK", "5"))
os.makedirs("artifacts", exist_ok=True)
s1, aux, edges, split = load()
S = pl.read_parquet("../prefilter/artifacts/scored_pairs.parquet")
C = (S.with_columns(ce_rank=pl.col("p").rank("ordinal", descending=True).over("s1"))
       .filter(pl.col("ce_rank") <= K))

# A0 trains the cross-encoder; A1a trains fusion; A1b tunes decoding.
def subpart(x):
    if x != "holdoutA": return x
    return ["A0", "A1a", "A1b"][int(hashlib.sha1(x.encode()).hexdigest()[16:24], 16) % 3]

parts = C.select("s1", "part").unique().with_columns(
    ce_part=pl.struct("s1", "part").map_elements(
        lambda z: z["part"] if z["part"] != "holdoutA" else ["A0", "A1a", "A1b"][int(hashlib.sha1(z["s1"].encode()).hexdigest()[16:24], 16) % 3],
        return_dtype=pl.Utf8))
C = C.join(parts, on=["s1", "part"])
V1 = s1.select(s1="entity_id", name1="business_name", addr1="business_address")
V2 = aux.select(a="entity_id", name2="business_name", addr2="business_address")
C = C.join(V1, on="s1").join(V2, on="a")
C.write_parquet("artifacts/ce_candidates.parquet", compression="zstd")
print(C.group_by("ce_part").agg(rows=pl.len(), positives=pl.col("label").sum()).sort("ce_part"))
