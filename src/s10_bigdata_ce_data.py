"""Big clean training set for the mmBERT cross-encoder (same clean rule as s09): all clean non-matches incl. same-name look-alikes,
1.2M matches with name-suffix-variation matches (et fils, & co, sas/sarl, pvt ltd ...) up-weighted to ~35%, plus A0. -> artifacts/train.parquet"""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import hashlib, os, re, sys
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load  # noqa: E402
from textviews import v_translit  # noqa: E402

TAG, TARGET = "[MISSING]", 2_600_000
SUFFIX = {"sas", "sasu", "sarl", "sa", "eurl", "sci", "snc", "scop", "et", "fils", "cie", "compagnie", "co", "company", "inc", "incorporated", "llc",
          "ltd", "limited", "pvt", "private", "corp", "corporation", "lp", "llp", "pllc", "pc", "plc", "group", "groupe", "the", "and", "of",
          "enterprises", "enterprise", "services", "service", "associates", "partners", "france", "india", "usa", "sri", "shri", "m", "s", "ms"}
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
V1 = s1.select(s1="entity_id", name1="business_name", addr1="business_address")
V2 = aux.select(a="entity_id", name2="business_name", addr2="business_address")
print(f"clean dev S1 {clean['s1'].n_unique():,}", flush=True)

short = clean.filter(pl.col("sl")).join(V1, on="s1").join(V2, on="a")
ext = clean.filter(~pl.col("sl") & (pl.col("label") == 0)).join(V1, on="s1").join(V2, on="a")
ns = cpdist([v_translit(x) for x in ext["name1"].to_list()], [v_translit(x) for x in ext["name2"].to_list()], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
hard = ext.filter(pl.Series(ns >= 85))
neg = pl.concat([short.filter(pl.col("label") == 0), hard])
pos = short.filter(pl.col("label") == 1)
tok = lambda s: set(re.sub(r"[^\w\s]", " ", v_translit(s)).split())
is_sfx = [(lambda a, b: bool(a ^ b) and (a ^ b) <= SUFFIX and bool(a & b))(tok(x), tok(y)) for x, y in zip(pos["name1"].to_list(), pos["name2"].to_list())]
pos = pos.with_columns(sfx=pl.Series(is_sfx))
sfx = pos.filter(pl.col("sfx")); rest = pos.filter(~pl.col("sfx"))
a0 = (pl.read_parquet(f"{ER_ROOT}/work/ce_tagged/artifacts/ce_candidates_tag.parquet")
        .filter(pl.col("ce_part") == "A0").select("s1", "a", "label", "name1", "addr1", "name2", "addr2"))
N_POS = 1_200_000                                     # clean matches: suffix-variation up-weighted to ~35% (natural ~24%), rest ordinary
n_sfx = min(sfx.height, int(0.35 * N_POS)); n_rest = min(rest.height, N_POS - n_sfx)
cols = ["s1", "a", "label", "name1", "addr1", "name2", "addr2"]
tr = pl.concat([neg.select(cols), sfx.sample(n_sfx, seed=12).select(cols), rest.sample(n_rest, seed=11).select(cols)])
e = pl.col("addr2") == ""
tr = tr.with_columns(addr1=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr1")), addr2=pl.when(e).then(pl.lit(TAG)).otherwise(pl.col("addr2")))
out = pl.concat([tr, a0.with_columns(pl.col("label").cast(tr["label"].dtype))]).sample(fraction=1.0, shuffle=True, seed=5).with_columns(ce_part=pl.lit("train"))
assert out.join(eval_s1, on="s1", how="semi").height == 0
out.write_parquet("artifacts/train.parquet")
print(f"TRAIN {out.height:,} pairs | non-matches {neg.height:,} (same-name look-alikes {hard.height:,}) | suffix-variation matches {n_sfx:,} of {sfx.height:,} | "
      f"other matches {n_rest:,} | A0 {a0.height:,} | positives {out['label'].mean():.1%}")
