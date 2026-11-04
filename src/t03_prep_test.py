"""Test cross-encoder inputs: shortlist = prefilter top-5 + ranks 6-10 with p >= 0.01 -> artifacts/test_meta.parquet,
[MISSING]-tag text in 2 shards (ce_test_{0,1}.parquet) and the empty-address pairs for the name-only model (nce_test.parquet)."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import os, sys
import polars as pl
sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import rd  # noqa: E402

T = f"{ER_ROOT}/data/test"
B = f"{ER_ROOT}/work/ce_candidates/artifacts_test"
TAG = "[MISSING]"
os.makedirs("artifacts", exist_ok=True)
s1 = rd(f"{T}/test_source1.tsv")
aux = pl.concat([rd(f"{T}/test_source2.tsv"), rd(f"{T}/test_source3.tsv")])
V1 = s1.select(s1="entity_id", name1="business_name", addr1="business_address")
V2 = aux.select(a="entity_id", name2="business_name", addr2="business_address")
parts = []
for c in ["France", "India", "US"]:
    b = pl.read_parquet(f"{B}/base__{c}.parquet").with_columns(ce_rank=pl.col("p").rank("ordinal", descending=True).over("s1"))
    parts.append(b.filter((pl.col("ce_rank") <= 5) | ((pl.col("ce_rank") <= 10) & (pl.col("p") >= 0.01))).with_columns(country=pl.lit(c)))
M = pl.concat(parts)
M.select("s1", "a", "ce_rank", "country").write_parquet("artifacts/test_meta.parquet")
X = M.join(V1, on="s1").join(V2, on="a")
e = pl.col("addr2") == ""
X = X.with_columns(addr1=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr1")), addr2=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr2")),
                   label=pl.lit(0, pl.Int8), ce_part=pl.lit("test"))
assert X.height == M.height, "text join lost pairs"
X = X.with_row_index("i")
for k in range(2):
    X.filter(pl.col("i") % 2 == k).drop("i").write_parquet(f"artifacts/ce_test_{k}.parquet")
X.filter(pl.col("addr2") == TAG).drop("i").write_parquet("artifacts/nce_test.parquet")
print(M.group_by("country").agg(pairs=pl.len(), s1=pl.col("s1").n_unique(), per_s1=pl.len() / pl.col("s1").n_unique()).sort("country"))
print(f"shortlist {M.height:,} pairs | empty-address (name-only) {X.filter(pl.col('addr2') == TAG).height:,} | test S1 {s1.height:,}, with shortlist {M['s1'].n_unique():,}")
