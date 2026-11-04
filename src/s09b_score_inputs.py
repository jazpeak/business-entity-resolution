"""Every shortlist pair to be scored for fusion training/validation (A1a, A1b, holdoutB + dev competitors), in the [MISSING]-tag text
format, split in 2 shards: tagged cross-encoder candidates + dev competitors + the extra ranks-6-10 shortlist pairs. Run from work/ce_street.
-> artifacts/score_in_{0,1}.parquet (used by the street-aware and mmBERT cross-encoders)"""
import os
import polars as pl
ER_ROOT = os.environ.get("ER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T4, SL = f"{ER_ROOT}/work/ce_tagged/artifacts", f"{ER_ROOT}/work/shortlist/artifacts"
cols = ["s1", "a", "label", "ce_part", "name1", "addr1", "name2", "addr2"]
d = pl.concat([pl.read_parquet(f"{T4}/ce_candidates_tag.parquet").filter(pl.col("ce_part").is_in(["A1a", "A1b", "holdoutB"])).select(cols),
               pl.read_parquet(f"{T4}/ce_dev_competitors_B_tag.parquet").select(cols),
               pl.read_parquet(f"{SL}/ce_new_tag.parquet").select(cols)], how="vertical_relaxed").unique(["s1", "a"]).with_row_index("i")
os.makedirs("artifacts", exist_ok=True)
for k in range(2): d.filter(pl.col("i") % 2 == k).drop("i").write_parquet(f"artifacts/score_in_{k}.parquet")
print("pairs to score", d.height)
