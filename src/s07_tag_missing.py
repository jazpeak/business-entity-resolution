"""Tagged cross-encoder inputs: when the aux address is empty, both addresses are replaced by '[MISSING]' so the model judges the pair by
name only. Records with an address are unchanged."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import polars as pl
M2 = f"{ER_ROOT}/work/ce_candidates/artifacts"
TAG = "[MISSING]"
def tag(df):
    e = pl.col("addr2") == ""
    return df.with_columns(addr1=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr1")),
                           addr2=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr2")))
c = tag(pl.scan_parquet(f"{M2}/ce_candidates.parquet").filter(pl.col("ce_part").is_in(["A0", "A1a", "A1b", "holdoutB"])).collect())
c.write_parquet("artifacts/ce_candidates_tag.parquet")
d = tag(pl.read_parquet(f"{M2}/ce_dev_competitors_B.parquet")); d.write_parquet("artifacts/ce_dev_competitors_B_tag.parquet")
print(c.group_by("ce_part").agg(pl.len(), tagged=(pl.col("addr2") == TAG).sum()).sort("ce_part"))
print("competitors", d.height, "tagged", (d["addr2"] == TAG).sum())
print(c.filter(pl.col("addr2") == TAG).select("name1", "addr1", "name2", "addr2", "label").head(3))
