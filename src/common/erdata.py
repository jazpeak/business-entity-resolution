"""Shared data loading (train data, frozen split, stratified samples)."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # package root (code/business_entity_resolution)

import sys
import polars as pl

sys.path.insert(0, f"{ER_ROOT}/src/common")

D = f"{ER_ROOT}/data/train"
SPLIT = f"{ER_ROOT}/work/splits/s1_split_v1.tsv"
NONLATIN = r"[^\p{Latin}\p{Common}\p{Inherited}\s]"
COUNTRIES = ["US", "India"]
W = {"US": 0.6, "India": 0.4}          # train S1 country mix, used for weighted averages


def rd(p):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def load():
    s1 = rd(f"{D}/train_source1.tsv")
    aux = pl.concat([rd(f"{D}/train_source2.tsv").with_columns(src=pl.lit("S2")),
                     rd(f"{D}/train_source3.tsv").with_columns(src=pl.lit("S3"))])
    aux = aux.with_columns(native=pl.col("business_name").str.contains(NONLATIN), empty_addr=pl.col("business_address") == "")
    gt = rd(f"{D}/train_ground_truth.tsv")
    edges = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "a"}))
    split = pl.read_csv(SPLIT, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8})
    return s1, aux, edges, split


def stratified_sample(split, which, n_per_country, seed=0):
    """n S1 per country from split `which` ('holdout'/'dev'), proportional over true-match cardinality 0..7+."""
    pool = split.filter(pl.col("split") == which).with_columns(cbin=pl.col("card").clip(0, 7))
    quota = pool.group_by("country", "cbin").len().with_columns(
        q=(pl.col("len") / pl.col("len").sum().over("country") * n_per_country).round().cast(pl.Int64))
    return (pool.sample(fraction=1.0, shuffle=True, seed=seed).with_columns(r=pl.int_range(pl.len()).over("country", "cbin"))
                .join(quota.select("country", "cbin", "q"), on=["country", "cbin"]).filter(pl.col("r") < pl.col("q"))
                .select(pl.col("source1_entity_id").alias("entity_id"), "country", "card"))
