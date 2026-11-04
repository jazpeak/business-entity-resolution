"""Final TEST prediction: shortlist features + 4 cross-encoder scores (tagged mDeBERTa, name-only mDeBERTa, street-aware mDeBERTa,
mmBERT big-data) -> final XGBoost fusion (models/fusion/fusion_final.json) -> one-owner + margin decoding (tau/delta from the meta file)
-> output/matching_results.tsv + output/candidate_pairs.tsv (every blocker candidate the prefilter model scores), then the official validator.
Feature code is identical to the fusion training script, so test features match training features."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import json, math, os, re, subprocess, sys, time
import numpy as np
import polars as pl
import xgboost as xgb
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import rd  # noqa: E402
from textviews import v_norm, v_translit  # noqa: E402
from hn2 import compare as hn2_compare  # noqa: E402

T = os.environ.get("ER_TEST_DATA", f"{ER_ROOT}/data/test")
M1 = os.environ.get("ER_M1_TEST_ARTIFACTS", f"{ER_ROOT}/work/prefilter/artifacts_test")
B = os.environ.get("ER_M2_TEST_ARTIFACTS", f"{ER_ROOT}/work/ce_candidates/artifacts_test")
A = os.environ.get("ER_ARTIFACTS", f"{ER_ROOT}/work/test/artifacts")
A15 = f"{ER_ROOT}/work/test/artifacts"
META_A = os.environ.get("ER_TEST_META_ARTIFACTS", A)
SUB = f"{ER_ROOT}/output"; os.makedirs(SUB, exist_ok=True)
COUNTRIES = ["France", "India", "US"]
T0 = time.time(); log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
s1 = rd(f"{T}/test_source1.tsv")
aux = pl.concat([rd(f"{T}/test_source2.tsv"), rd(f"{T}/test_source3.tsv")])

# =============================================================== fusion base features 
F_OLD = ["p", "ce_logit", "ce_prob", "ce_rank", "p_x_ce", "p_minus_ce"]
F_PAIR = ["score", "gap_aux", "aux_margin", "is_aux_top1", "is_rev", "s1_n", "tr_tset", "nm_ratio", "nm_tset", "ad_tset",
          "num_state", "num1_sim", "num_jaccard", "empty_addr_i"]
F_M3NEW = ["s1_name_freq", "aux_name_freq", "s1_rarest_tok_df", "aux_rarest_tok_df", "near_name_cands", "name_margin", "n_cand_s1_of_aux"]
log("fusion base features")
S1n = s1.select(pl.col("entity_id").alias("s1"), "country", n1="business_name", a1="business_address").with_columns(
    nn=pl.col("n1").map_elements(v_norm, return_dtype=pl.Utf8))
Auxn = aux.select(pl.col("entity_id").alias("a"), "country", n2="business_name", a2="business_address").with_columns(
    nn=pl.col("n2").map_elements(v_norm, return_dtype=pl.Utf8))
name_freq = S1n.group_by("country", "nn").agg(freq=pl.len())
tok_df = (S1n.select("country", tok=pl.col("nn").str.split(" ")).explode("tok").filter(pl.col("tok") != "").group_by("country", "tok").agg(df=pl.len()))
def rarest(df, key):
    t = df.select(key, "country", tok=pl.col("nn").str.split(" ")).explode("tok").filter(pl.col("tok") != "") \
          .join(tok_df, on=["country", "tok"], how="left").with_columns(pl.col("df").fill_null(0))
    return t.group_by(key).agg(rarest=pl.col("df").min())
s1_feat = S1n.join(name_freq, on=["country", "nn"]).select("s1", s1_name_freq="freq").join(rarest(S1n, "s1").rename({"rarest": "s1_rarest_tok_df"}), on="s1", how="left")
aux_feat = (Auxn.join(name_freq, on=["country", "nn"], how="left").select("a", aux_name_freq=pl.col("freq").fill_null(0))
                .join(rarest(Auxn, "a").rename({"rarest": "aux_rarest_tok_df"}), on="a", how="left"))
P = pl.concat([pl.read_parquet(f"{M1}/pairs__{c}.parquet", columns=["s1", "a"] + F_PAIR) for c in COUNTRIES])  # test pairs
look = P.group_by("a").agg(near_name_cands=(pl.col("tr_tset") >= 90).sum(), n_cand_s1_of_aux=pl.len(),
                           best_tr=pl.col("tr_tset").max(), second_tr=pl.col("tr_tset").sort(descending=True).get(1, null_on_oob=True))
P = (P.join(look, on="a")
      .with_columns(name_margin=pl.when(pl.col("tr_tset") >= pl.col("best_tr")).then(pl.col("tr_tset") - pl.col("second_tr").fill_null(0))
                                  .otherwise(pl.col("tr_tset") - pl.col("best_tr")))
      .drop("best_tr", "second_tr").join(s1_feat, on="s1", how="left").join(aux_feat, on="a", how="left"))

base = pl.concat([pl.read_parquet(f"{B}/base__{c}.parquet") for c in COUNTRIES])
meta = pl.read_parquet(f"{META_A}/test_meta.parquet", columns=["s1", "a", "ce_rank"])
CE = pl.concat([pl.read_parquet(f"{A}/ce_test_scores_{k}.parquet") for k in range(2)])
NCE = pl.read_parquet(f"{A}/nce_test_scores.parquet").rename({"ce_logit": "nce_logit"})
assert CE.height == meta.height and CE.join(meta, on=["s1", "a"]).height == meta.height, "CE scores do not cover the shortlist"
D = CE.join(meta, on=["s1", "a"]).join(base, on=["s1", "a"]).with_columns(ce_prob=1 / (1 + (-pl.col("ce_logit")).exp()))
D = D.with_columns(p_x_ce=pl.col("p") * pl.col("ce_prob"), p_minus_ce=pl.col("p") - pl.col("ce_prob"))
D = D.join(P, on=["s1", "a"], how="left").join(NCE, on=["s1", "a"], how="left")
g = D.filter(pl.col("nce_logit").is_not_null()).group_by("a").agg(b1=pl.col("nce_logit").max(), b2=pl.col("nce_logit").sort(descending=True).get(1, null_on_oob=True))
D = D.join(g, on="a", how="left").with_columns(
    nce_margin=pl.when(pl.col("nce_logit") >= pl.col("b1")).then(pl.col("nce_logit") - pl.col("b2").fill_null(-10)).otherwise(pl.col("nce_logit") - pl.col("b1"))).drop("b1", "b2")
log(f"test shortlist {D.height:,} pairs, name-only scored {D['nce_logit'].is_not_null().sum():,}; blocker pairs {base.height:,}")
BIG = pl.concat([pl.read_parquet(f"{A15}/big_test_scores_{k}.parquet") for k in range(2)]).unique(["s1", "a"]).rename({"ce_logit": "big_logit"})
assert BIG.join(meta, on=["s1", "a"], how="semi").height == meta.height, "street-aware CE scores do not cover the shortlist"
D = D.join(BIG, on=["s1", "a"], how="left").with_columns(big_prob=1 / (1 + (-pl.col("big_logit")).exp()))
D = D.with_columns(p_x_big=pl.col("p") * pl.col("big_prob"), ce2_avg=(pl.col("ce_prob") + pl.col("big_prob")) / 2)
import glob as _glob
MMF = sorted(_glob.glob(f"{ER_ROOT}/work/ce_mmbert/test/scores_*.parquet"))
MM2 = pl.concat([pl.read_parquet(f).select("s1", "a", "ce_logit") for f in MMF]).unique(["s1", "a"]).rename({"ce_logit": "mm2_logit"})
assert MM2.join(meta, on=["s1", "a"], how="semi").height == meta.height, "mmBERT scores do not cover the shortlist"
D = D.join(MM2, on=["s1", "a"], how="left").with_columns(mm2_prob=1 / (1 + (-pl.col("mm2_logit")).exp()))
D = D.with_columns(p_x_mm2=pl.col("p") * pl.col("mm2_prob"))
A1, HB, COMP = D, D.head(0), D.head(0)          # names used by the verbatim feature block

# =============================================================== new features: parse addresses / names once per record
_HN = re.compile(r"(?<![\w])(\d+[a-z]?(?:[-/]\w+)*)", re.I)          # house token: 14593, 14593d, 1-d-206, 604/6
_DIG = re.compile(r"\d+")
_UNIT = re.compile(r"(?:unit|apt|apartment|suite|ste|flat|fl|floor|room|rm|#|no\.?|appartement|bureau)\s*[#:.\-]?\s*([a-z]?\d+[a-z]?)", re.I)
_STREET_TYPES = {"street", "st", "road", "rd", "avenue", "ave", "av", "drive", "dr", "lane", "ln", "court", "ct", "circle", "cir", "boulevard",
                 "blvd", "bd", "place", "pl", "way", "highway", "hwy", "parkway", "pkwy", "terrace", "trail", "rue", "r", "allee", "all",
                 "impasse", "imp", "route", "rte", "chemin", "che", "square", "sq", "quai", "cours", "nagar", "marg", "sector", "colony",
                 "block", "plot", "no", "house", "flat", "shop", "floor"}
def parse_addr(a):
    a = (a or "").replace("<NULL>", " ")
    low = a.lower()
    m = _HN.search(low)
    hn_full = m.group(1) if m else ""
    hn = (_DIG.search(hn_full).group() if hn_full else "")
    nums = sorted(set(_DIG.findall(low)))
    unit = (_UNIT.search(low).group(1) if _UNIT.search(low) else "")
    parts = [p.strip() for p in low.split(",") if p.strip()]
    j = next((k for k, p in enumerate(parts) if _DIG.search(p)), 0)
    clean = lambda x: [t for t in re.sub(r"[^\w\s]", " ", _DIG.sub(" ", x)).split() if t not in _STREET_TYPES and len(t) > 1]
    st = clean(parts[j]) if parts else []
    if len(st) < 2 and j + 1 < len(parts):             # India style '1-D-206, Eksar Road, ...': street is the next part
        st += clean(parts[j + 1])
    locs = "|".join(re.sub(r"[^\w\s]", " ", p).strip() for p in parts if not _DIG.search(p))   # locality parts (city, district, state...)
    return hn, hn_full, " ".join(nums), unit, " ".join(st), locs
def parse_many(addrs):
    return [parse_addr(x) for x in addrs]
def parsed(df, idcol, addrcol):
    xs = df[addrcol].to_list(); step = len(xs) // 64 + 1
    res = sum(Parallel(n_jobs=64)(delayed(parse_many)(xs[i:i + step]) for i in range(0, len(xs), step)), [])
    cols = list(zip(*res)) if res else [[]] * 6
    return df.select(idcol).with_columns(**{k: pl.Series(list(v)) for k, v in zip(["hn", "hn_full", "nums", "unit", "street", "city"], cols)})

pairs_all = pl.concat([A1.select("s1", "a"), HB.select("s1", "a"), COMP.select("s1", "a")]).unique()
aux_set = pairs_all.select("a").unique()
cand = P.select("s1", "a").join(aux_set, on="a", how="semi")          # all candidate S1 of these aux (for competition features)
log(f"parsing addresses: {cand['s1'].n_unique():,} S1, {aux_set.height:,} aux")
PS1 = parsed(S1n.join(cand.select("s1").unique(), on="s1", how="semi"), "s1", "a1")
PAX = parsed(Auxn.join(aux_set, on="a", how="semi"), "a", "a2")
# number rarity: in how many S1 addresses of the country does a number occur
num_df = (S1n.select("country", nums=pl.col("a1").str.extract_all(r"\d+").list.unique()).explode("nums").drop_nulls()
             .group_by("country", "nums").agg(ndf=pl.len()))
num_df_map = {(c, n): d for c, n, d in num_df.iter_rows()}
# IDF over S1 name tokens (translit, per country)
log("name IDF")
S1t = S1n.select("s1", "country", tt=pl.col("n1").map_elements(lambda x: " ".join(sorted(set(v_translit(x).split()))), return_dtype=pl.Utf8))
AXt = Auxn.join(aux_set, on="a", how="semi").select("a", tt=pl.col("n2").map_elements(lambda x: " ".join(sorted(set(v_translit(x).split()))), return_dtype=pl.Utf8))
ndoc = S1t.group_by("country").len()
tdf = S1t.select("country", t=pl.col("tt").str.split(" ")).explode("t").filter(pl.col("t") != "").group_by("country", "t").agg(df=pl.len())
N_c = dict(ndoc.iter_rows()); idf = {(c, t): math.log((N_c[c] + 1) / (d + 1)) + 1 for c, t, d in tdf.iter_rows()}

def pair_new_features(df):
    """df: s1, a, country + parsed columns (_1 for S1, _2 for aux) + tt_1, tt_2 -> HN, ST, IDF columns."""
    kw = dict(workers=-1, dtype=np.float32)
    hn1, hn2 = df["hn_1"].to_list(), df["hn_2"].to_list()
    hf1, hf2 = df["hn_full_1"].to_list(), df["hn_full_2"].to_list()
    n1s, n2s = df["nums_1"].to_list(), df["nums_2"].to_list(); u1, u2 = df["unit_1"].to_list(), df["unit_2"].to_list()
    ctry = df["country"].to_list(); t1s, t2s = df["tt_1"].to_list(), df["tt_2"].to_list()
    city1, a2full = df["city_1"].to_list(), df["a2_low"].to_list()
    out = {k: np.zeros(df.height, np.float32) for k in
           ["hn_eq", "hn_full_eq", "hn_logdiff", "hn_reldiff", "hn_lev", "hn_prefix", "hn_len_eq", "hn_missing", "unit_state",
            "long_shared", "nums_shared", "nums_only_s1", "nums_only_aux", "shared_num_rarity", "city_in_aux", "idf_overlap_s1", "idf_overlap_aux"]}
    for i in range(df.height):
        a, b = hn1[i], hn2[i]
        if a and b:
            ia, ib = int(a[:12]), int(b[:12])
            out["hn_eq"][i] = a == b; out["hn_full_eq"][i] = hf1[i] == hf2[i]
            out["hn_logdiff"][i] = math.log1p(abs(ia - ib)); out["hn_reldiff"][i] = abs(ia - ib) / max(ia, ib, 1)
            out["hn_lev"][i] = Levenshtein.normalized_similarity(a, b); out["hn_prefix"][i] = a != b and (a.startswith(b) or b.startswith(a))
            out["hn_len_eq"][i] = len(a) == len(b)
        else:
            out["hn_missing"][i] = 1 + (not a) + 2 * (not b)                      # 2: S1 missing, 3: aux missing, 4: both
            out["hn_logdiff"][i] = -1; out["hn_reldiff"][i] = -1; out["hn_lev"][i] = -1
        out["unit_state"][i] = (2 if (u1[i] and u2[i] and u1[i] == u2[i]) else 1 if (u1[i] and u2[i]) else 0)  # 2 agree, 1 conflict, 0 n/a
        A, B = set(n1s[i].split()), set(n2s[i].split())
        sh = A & B
        out["nums_shared"][i] = len(sh); out["nums_only_s1"][i] = len(A - B); out["nums_only_aux"][i] = len(B - A)
        out["long_shared"][i] = sum(1 for x in sh if len(x) >= 3)
        out["shared_num_rarity"][i] = min((math.log1p(num_df_map.get((ctry[i], x), 0)) for x in sh), default=-1)
        L = [x for x in city1[i].split("|") if x]
        out["city_in_aux"][i] = (sum(x in a2full[i] for x in L) / len(L)) if (L and a2full[i]) else -1   # locality overlap share
        T1, T2 = set(t1s[i].split()), set(t2s[i].split())
        w = lambda T: sum(idf.get((ctry[i], t), 12.0) for t in T)
        shw = w(T1 & T2)
        out["idf_overlap_s1"][i] = shw / max(w(T1), 1e-6); out["idf_overlap_aux"][i] = shw / max(w(T2), 1e-6)
    st = cpdist(df["street_1"].to_list(), df["street_2"].to_list(), scorer=fuzz.token_set_ratio, **kw)
    return df.select("s1", "a").with_columns(**{k: pl.Series(v) for k, v in out.items()}, street_tset=pl.Series(st))

def attach(df):
    return (df.join(S1n.select("s1", "country"), on="s1")
              .join(PS1.rename({k: f"{k}_1" for k in PS1.columns if k != "s1"}), on="s1", how="left")
              .join(PAX.rename({k: f"{k}_2" for k in PAX.columns if k != "a"}), on="a", how="left")
              .join(S1t.select("s1", tt_1="tt"), on="s1", how="left").join(AXt.select("a", tt_2="tt"), on="a", how="left")
              .join(Auxn.select("a", a2_low=pl.col("a2").str.to_lowercase()), on="a", how="left").fill_null(""))
def chunked_features(df, n=64):
    df = attach(df); step = df.height // n + 1
    parts = Parallel(n_jobs=n)(delayed(pair_new_features)(df.slice(i, step)) for i in range(0, df.height, step))
    return pl.concat(parts)
log(f"pair features for {cand.height:,} candidate pairs (competition needs all candidates of the aux)")
NF = chunked_features(cand)
# competition by address / number among the aux's candidates
C = NF.join(P.select("s1", "a", "ad_tset"), on=["s1", "a"], how="left")
agg = C.group_by("a").agg(cands_same_hn=pl.col("hn_eq").sum(), best_ad=pl.col("ad_tset").max(),
                          second_ad=pl.col("ad_tset").sort(descending=True).get(1, null_on_oob=True),
                          best_idf=pl.col("idf_overlap_s1").max(), second_idf=pl.col("idf_overlap_s1").sort(descending=True).get(1, null_on_oob=True))
C = (C.join(agg, on="a")
      .with_columns(addr_margin=pl.when(pl.col("ad_tset") >= pl.col("best_ad")).then(pl.col("ad_tset") - pl.col("second_ad").fill_null(0)).otherwise(pl.col("ad_tset") - pl.col("best_ad")),
                    idf_margin=pl.when(pl.col("idf_overlap_s1") >= pl.col("best_idf")).then(pl.col("idf_overlap_s1") - pl.col("second_idf").fill_null(0)).otherwise(pl.col("idf_overlap_s1") - pl.col("best_idf")))
      .drop("best_ad", "second_ad", "best_idf", "second_idf", "ad_tset"))
F_HN = ["hn_eq", "hn_full_eq", "hn_logdiff", "hn_reldiff", "hn_lev", "hn_prefix", "hn_len_eq", "hn_missing", "unit_state",
        "long_shared", "nums_shared", "nums_only_s1", "nums_only_aux", "shared_num_rarity"]
F_ST = ["street_tset", "city_in_aux"]
F_IDF = ["idf_overlap_s1", "idf_overlap_aux"]
F_CMP = ["cands_same_hn", "addr_margin", "idf_margin"]
def hn2_chunk(a1s, a2s):
    return [hn2_compare(x, y) for x, y in zip(a1s, a2s)]
HX = cand.join(S1n.select("s1", "a1"), on="s1").join(Auxn.select("a", "a2"), on="a")
step = HX.height // 64 + 1
log(f"house-number v2 features for {HX.height:,} candidate pairs")
res = Parallel(n_jobs=64)(delayed(hn2_chunk)(HX["a1"][i:i + step].to_list(), HX["a2"][i:i + step].to_list()) for i in range(0, HX.height, step))
H2 = pl.concat([HX.select("s1", "a"), pl.DataFrame(sum(res, []))], how="horizontal")
H2 = H2.join(H2.group_by("a").agg(cands_same_hn2=((pl.col("hn2_eq") == 1) | (pl.col("hn2_in_other") == 1)).sum()), on="a")
C = C.join(H2, on=["s1", "a"], how="left")
F_HN2 = ["hn2_state", "hn2_eq", "hn2_in_other", "hn2_mindiff", "hn2_trunc", "nums2_shared", "nums2_only_s1", "nums2_only_aux", "cands_same_hn2"]
A1 = A1.join(C, on=["s1", "a"], how="left"); HB = HB.join(C, on=["s1", "a"], how="left"); COMP = COMP.join(C, on=["s1", "a"], how="left")
log("features ready")


D = A1
F13 = f"{ER_ROOT}/models/fusion"
MF = json.load(open(f"{F13}/fusion_final_meta.json")); FEATS, TAU, DELTA = MF["features"], MF["tau"], MF["delta"]
bst = xgb.Booster(); bst.load_model(f"{F13}/fusion_final.json"); bst.set_param({"device": "cpu"})
missing = [f for f in FEATS if f not in D.columns]; assert not missing, missing
fused = D.select("s1", "a").with_columns(pf=pl.Series(bst.inplace_predict(D.select(FEATS).to_numpy().astype(np.float32))))
log(f"fused {fused.height:,} pairs (tau {TAU}, delta {DELTA})")
S = base.join(fused, on=["s1", "a"], how="left").with_columns(pl.col("pf").fill_null(pl.col("p")))
X = S.with_columns(r=pl.col("pf").rank("ordinal", descending=True).over("a"))
best = (X.filter(pl.col("r") == 1).join(X.filter(pl.col("r") == 2).select("a", p2="pf"), on="a", how="left").with_columns(pl.col("p2").fill_null(0.0))
          .filter((pl.col("pf") >= TAU) & (pl.col("pf") - pl.col("p2") >= DELTA)))
def write(pairs, col, path):
    lists = pairs.group_by("s1").agg(pl.col("a").unique().sort().str.join(",").alias(col))
    out = s1.select(source1_entity_id="entity_id").join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))
    out.write_csv(path, separator="\t", quote_style="never")
write(best.select("s1", "a"), "matched_entity_ids", f"{SUB}/matching_results.tsv")
write(S.select("s1", "a"), "candidate_entity_ids", f"{SUB}/candidate_pairs.tsv")
best.select("s1", "a", "pf", "p2").write_parquet(f"{A15}/test_predictions.parquet")
stats = (s1.select(s1="entity_id", country="country").join(best.group_by("s1").agg(k=pl.len()), on="s1", how="left")
           .join(S.group_by("s1").agg(n=pl.len()), on="s1", how="left").with_columns(pl.col("k").fill_null(0), pl.col("n").fill_null(0))
           .join(pl.concat([pl.read_parquet(f"{META_A}/test_meta.parquet", columns=["s1"])]).group_by("s1").agg(sl=pl.len()), on="s1", how="left")
           .group_by("country").agg(s1=pl.len(), cand_per_s1=pl.col("n").mean(), shortlist_per_s1=pl.col("sl").mean(), pred_per_s1=pl.col("k").mean(),
                                    pred_empty_pct=(pl.col("k") == 0).mean() * 100).sort("country"))
print(stats); stats.write_csv(f"{A15}/test_country_diagnostics.csv")
log(f"wrote {SUB}: {best.height:,} matches, {S.height:,} candidate pairs, {s1.height:,} S1")
r = subprocess.run([sys.executable, f"{ER_ROOT}/src/utils/validate_submission.py", "--matching", f"{SUB}/matching_results.tsv",
                    "--candidate", f"{SUB}/candidate_pairs.tsv", "--test-dir", T, "--check-ids"], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:]); raise SystemExit(r.returncode)
