"""Street-aware cross-encoder training data, with no bias on validation. Clean pool = dev S1 whose shortlist never shares an aux with an
A1a/A1b/holdoutB shortlist, and no pair whose aux belongs to a holdout S1. Adds same-name (translit token-set >= 85) non-matches outside the
shortlist (street / house-number look-alikes); [MISSING]-tag text format."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import hashlib, os, sys
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load  # noqa: E402
from textviews import v_translit  # noqa: E402

TAG = "[MISSING]"
os.makedirs("artifacts", exist_ok=True)
s1, aux, edges, split = load()
S = pl.read_parquet(f"{ER_ROOT}/work/prefilter/artifacts/scored_pairs.parquet", columns=["s1", "a", "label", "part", "p"])
S = S.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("s1"))
S = S.with_columns(sl=(pl.col("r") <= 5) | ((pl.col("r") <= 10) & (pl.col("p") >= 0.01)))
sub = lambda e: ["A0", "A1a", "A1b"][int(hashlib.sha1(e.encode()).hexdigest()[16:24], 16) % 3]
ha = S.filter(pl.col("part") == "holdoutA").select("s1").unique()
ha = ha.with_columns(cp=pl.Series([sub(x) for x in ha["s1"].to_list()]))
eval_s1 = pl.concat([ha.filter(pl.col("cp").is_in(["A1a", "A1b"])).select("s1"), S.filter(pl.col("part") == "holdoutB").select("s1").unique()])
eval_aux = S.filter(pl.col("sl")).join(eval_s1, on="s1", how="semi").select("a").unique()
hold_aux = edges.join(split.filter(pl.col("split") == "holdout").select(s1="source1_entity_id"), on="s1", how="semi").select("a")
dev = S.filter(pl.col("part") == "dev")
dirty = dev.filter(pl.col("sl")).join(eval_aux, on="a", how="semi").select("s1").unique()
clean = dev.join(dirty, on="s1", how="anti").join(hold_aux, on="a", how="anti").join(eval_aux, on="a", how="anti")
print(f"clean dev S1 {clean['s1'].n_unique():,}")

V1 = s1.select(s1="entity_id", name1="business_name", addr1="business_address")
V2 = aux.select(a="entity_id", name2="business_name", addr2="business_address")
short = clean.filter(pl.col("sl"))
ext = clean.filter(~pl.col("sl") & (pl.col("label") == 0)).join(V1, on="s1").join(V2, on="a")
ns = cpdist([v_translit(x) for x in ext["name1"].to_list()], [v_translit(x) for x in ext["name2"].to_list()], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
hard = ext.filter(pl.Series(ns >= 85)).select("s1", "a", "label", "p", "r")
print(f"shortlist pairs {short.height:,} (pos {short['label'].sum():,}) | extra same-name hard negatives {hard.height:,}")

neg = pl.concat([short.filter(pl.col("label") == 0).select("s1", "a", "label", "p", "r"), hard])
pos = short.filter(pl.col("label") == 1).select("s1", "a", "label", "p", "r")
pos = pos.sample(min(pos.height, neg.height), seed=11)
tr = pl.concat([neg, pos]).join(V1, on="s1").join(V2, on="a")
e = pl.col("addr2") == ""
tr = tr.with_columns(addr1=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr1")), addr2=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr2")))
a0 = (pl.read_parquet(f"{ER_ROOT}/work/ce_tagged/artifacts/ce_candidates_tag.parquet")
        .filter(pl.col("ce_part") == "A0").select("s1", "a", "label", "name1", "addr1", "name2", "addr2"))
out = pl.concat([tr.select("s1", "a", "label", "name1", "addr1", "name2", "addr2"), a0.with_columns(pl.col("label").cast(tr["label"].dtype))])
out = out.unique(["s1", "a"]).sample(fraction=1.0, shuffle=True, seed=5).with_columns(ce_part=pl.lit("train"))
out.write_parquet("artifacts/train.parquet")
print(f"TRAIN {out.height:,} pairs | positives {out['label'].sum():,} ({out['label'].mean():.1%}) | from A0 {a0.height:,}")
assert out.join(eval_s1, on="s1", how="semi").height == 0 and out.join(hold_aux, on="a", how="semi").filter(~pl.col("s1").is_in(a0["s1"].implode())).height == 0
