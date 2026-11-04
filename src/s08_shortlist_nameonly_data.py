"""Wider shortlist + name-only cross-encoder data.
1. Shortlist = prefilter top-5 + ranks 6-10 with p >= 0.01; extra pairs to score for A1a/A1b/holdoutB and dev competitors.
2. Name-only cross-encoder training data: blocker candidates of empty-address aux (same-name twins as hard negatives).
3. Name-only scoring set: every empty-address shortlist pair."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import hashlib, os, sys
import polars as pl
sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load  # noqa: E402

M1 = f"{ER_ROOT}/work/prefilter/artifacts"
M4 = f"{ER_ROOT}/work/ce_tagged/artifacts"
os.makedirs("artifacts", exist_ok=True)
K, TAG, NCE_TRAIN = 10, "[MISSING]", 700_000
s1, aux, edges, split = load()

def sub(e, part):
    if part != "holdoutA": return part
    return ["A0", "A1a", "A1b"][int(hashlib.sha1(e.encode()).hexdigest()[16:24], 16) % 3]
S = pl.read_parquet(f"{M1}/scored_pairs.parquet").with_columns(ce_rank=pl.col("p").rank("ordinal", descending=True).over("s1"))
parts = S.select("s1", "part").unique()
parts = parts.with_columns(ce_part=pl.Series([sub(e, p) for e, p in zip(parts["s1"].to_list(), parts["part"].to_list())]))
S = S.join(parts.select("s1", "ce_part"), on="s1")
V1 = s1.select(s1="entity_id", name1="business_name", addr1="business_address")
V2 = aux.select(a="entity_id", name2="business_name", addr2="business_address")
def tag(df):
    e = pl.col("addr2") == ""
    return df.with_columns(addr1=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr1")), addr2=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr2")))

SL = (pl.col("ce_rank") <= 5) | ((pl.col("ce_rank") <= K) & (pl.col("p") >= 0.01))   # shortlist: top-5 + ranks 6-10 with prefilter p >= 0.01
top = S.filter(SL)
tgt = top.filter(pl.col("ce_part").is_in(["A1a", "A1b", "holdoutB"]))
comp = top.filter(pl.col("ce_part") == "dev").join(tgt.filter(pl.col("ce_part") == "holdoutB").select("a").unique(), on="a", how="semi")
done = pl.concat([pl.read_parquet(f"{M4}/ce_candidates_tag.parquet", columns=["s1", "a"]), pl.read_parquet(f"{M4}/ce_dev_competitors_B_tag.parquet", columns=["s1", "a"])])
allk = pl.concat([tgt, comp])
allk.select("s1", "a", "ce_rank", "ce_part").write_parquet("artifacts/top10_meta.parquet")     # ce_rank for every top-10 pair
new = tag(allk.join(done, on=["s1", "a"], how="anti").join(V1, on="s1").join(V2, on="a"))
new.write_parquet("artifacts/ce_new_tag.parquet")
print("top-10 set:", allk.group_by("ce_part").len().sort("ce_part")); print("new pairs to score:", new.group_by("ce_part").len().sort("ce_part"))

emp = aux.filter(pl.col("empty_addr")).select(a="entity_id")
tr = (S.filter((pl.col("ce_part") == "dev") & SL).join(emp, on="a", how="semi").join(V1, on="s1").join(V2, on="a"))
print(f"dev empty-address candidate pairs: {tr.height:,} (positives {tr['label'].sum():,})")
tr = tag(tr.sample(min(NCE_TRAIN, tr.height), seed=3)).with_columns(ce_part=pl.lit("train"))
tr.write_parquet("artifacts/nce_train.parquet"); print(f"name-only train sample {tr.height:,}, positives {tr['label'].sum():,}")
sc = tag(allk.join(emp, on="a", how="semi").join(V1, on="s1").join(V2, on="a"))
sc.write_parquet("artifacts/nce_score.parquet"); print("name-only scoring set:", sc.group_by("ce_part").len().sort("ce_part"))
