"""Frozen entity-level split: holdout = int(sha1(entity_id)[:8], 16) % 5 == 0 (20%), the rest is dev. Writes s1_split_v1.tsv."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import hashlib, polars as pl
D = f"{ER_ROOT}/data/train"
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False).fill_null("")
s1 = rd(f"{D}/train_source1.tsv").select("entity_id", "country")
gt = rd(f"{D}/train_ground_truth.tsv")
card = gt.select(pl.col("source1_entity_id").alias("entity_id"),
                 card=pl.when(pl.col("matched_entity_ids") == "").then(0).otherwise(pl.col("matched_entity_ids").str.split(",").list.len()))
h = [int(hashlib.sha1(e.encode()).hexdigest()[:8], 16) % 5 for e in s1["entity_id"].to_list()]
out = (s1.with_columns(fold=pl.Series(h, dtype=pl.Int8)).join(card, on="entity_id")
         .with_columns(split=pl.when(pl.col("fold") == 0).then(pl.lit("holdout")).otherwise(pl.lit("dev")))
         .rename({"entity_id": "source1_entity_id"}).sort("source1_entity_id"))
out.write_csv("s1_split_v1.tsv", separator="\t")
print(out.group_by("split", "country").agg(pl.len(), singleton_pct=(pl.col("card") == 0).mean() * 100, mean_card=pl.col("card").mean()).sort("split", "country"))
