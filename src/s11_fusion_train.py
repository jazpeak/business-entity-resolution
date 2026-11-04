"""Final fusion model (trained on A1a, decoder tuned on A1b, holdoutB reported once).

Inputs per shortlist pair: prefilter LightGBM probability p, the 4 cross-encoders
  - ce_logit   : mDeBERTa-v3 cross-encoder, [MISSING]-tag input (trained on A0)
  - nce_logit  : name-only mDeBERTa cross-encoder for empty-address records (+ nce_margin)
  - big_logit  : street-aware mDeBERTa cross-encoder (clean dev pool + same-name look-alike negatives)
  - mm2_logit  : mmBERT-base cross-encoder trained on the big clean set (2.9M pairs)
plus engineered name / address / house-number (parser v2) / competition features. XGBoost (CUDA) -> one-owner + margin decoding.
Validated: A1b 0.99079, holdoutB 0.99070. Writes models/fusion/fusion_final.json + fusion_final_meta.json.
"""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import hashlib, math, os, re, sys, time
from collections import Counter
import numpy as np
import polars as pl
import torch
import xgboost as xgb
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load, COUNTRIES  # noqa: E402
from textviews import v_norm, v_translit  # noqa: E402
from hn2 import compare as hn2_compare  # noqa: E402

M1 = f"{ER_ROOT}/work/prefilter/artifacts"
M2 = f"{ER_ROOT}/work/ce_candidates/artifacts"
OUT = "artifacts"; os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
T0 = time.time(); log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)

s1, aux, edges, split = load()

# =============================================================== parts (same hashing as tagged-CE/fusion)
def part_of(e, sp):
    if sp != "holdout": return sp
    if int(hashlib.sha1(e.encode()).hexdigest()[8:16], 16) % 2: return "holdoutB"
    return ["A0", "A1a", "A1b"][int(hashlib.sha1(e.encode()).hexdigest()[16:24], 16) % 3]
pop = split.select(pl.col("source1_entity_id").alias("s1"), "card", "country", "split")
pop = pop.with_columns(part=pl.Series([part_of(e, s) for e, s in zip(pop["s1"].to_list(), pop["split"].to_list())])).sort("s1")

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
P = pl.concat([pl.read_parquet(f"{M1}/pairs__{c}.parquet", columns=["s1", "a"] + F_PAIR) for c in COUNTRIES])
look = P.group_by("a").agg(near_name_cands=(pl.col("tr_tset") >= 90).sum(), n_cand_s1_of_aux=pl.len(),
                           best_tr=pl.col("tr_tset").max(), second_tr=pl.col("tr_tset").sort(descending=True).get(1, null_on_oob=True))
P = (P.join(look, on="a")
      .with_columns(name_margin=pl.when(pl.col("tr_tset") >= pl.col("best_tr")).then(pl.col("tr_tset") - pl.col("second_tr").fill_null(0))
                                  .otherwise(pl.col("tr_tset") - pl.col("best_tr")))
      .drop("best_tr", "second_tr").join(s1_feat, on="s1", how="left").join(aux_feat, on="a", how="left"))
base = pl.read_parquet(f"{M1}/scored_pairs.parquet").select("s1", "a", "p", "label")
meta = pl.read_parquet(f"{M2}/ce_candidates.parquet", columns=["s1", "a", "ce_rank"])
def build(ce_path):
    d = pl.read_parquet(ce_path).join(meta, on=["s1", "a"]).join(base, on=["s1", "a"]).with_columns(ce_prob=1 / (1 + (-pl.col("ce_logit")).exp()))
    d = d.with_columns(p_x_ce=pl.col("p") * pl.col("ce_prob"), p_minus_ce=pl.col("p") - pl.col("ce_prob"))
    return d.join(P, on=["s1", "a"], how="left")
M4 = f"{ER_ROOT}/work/ce_tagged/artifacts"; M8 = f"{ER_ROOT}/work/shortlist/artifacts"
meta = pl.read_parquet(f"{M8}/top10_meta.parquet", columns=["s1", "a", "ce_rank", "ce_part"])
NEW = pl.read_parquet(f"{M8}/ce_new_scores.parquet")
ALLCE = pl.concat([pl.read_parquet(f"{M4}/ce_A1.parquet").with_columns(old=pl.lit(True)), pl.read_parquet(f"{M4}/ce_holdoutB.parquet").with_columns(old=pl.lit(True)),
                   pl.read_parquet(f"{M4}/ce_dev_competitors_B_scores.parquet").with_columns(old=pl.lit(True)), NEW.with_columns(old=pl.lit(False))]).unique(["s1", "a"])
NCE = pl.read_parquet(f"{M8}/nce_scores.parquet").rename({"ce_logit": "nce_logit"})
def build_all():
    d = ALLCE.join(meta, on=["s1", "a"]).join(base, on=["s1", "a"]).with_columns(ce_prob=1 / (1 + (-pl.col("ce_logit")).exp()))
    d = d.with_columns(p_x_ce=pl.col("p") * pl.col("ce_prob"), p_minus_ce=pl.col("p") - pl.col("ce_prob"))
    return d.join(P, on=["s1", "a"], how="left").join(NCE, on=["s1", "a"], how="left")
D = build_all()
BIG = pl.concat([pl.read_parquet(f"{ER_ROOT}/work/ce_street/artifacts/big_scores_{k}.parquet") for k in range(2)]).unique(["s1", "a"]).rename({"ce_logit": "big_logit"})
D = D.join(BIG, on=["s1", "a"], how="left").with_columns(big_prob=1 / (1 + (-pl.col("big_logit")).exp()))
D = D.with_columns(p_x_big=pl.col("p") * pl.col("big_prob"), ce2_avg=(pl.col("ce_prob") + pl.col("big_prob")) / 2)
MM2 = pl.concat([pl.read_parquet(f"{ER_ROOT}/work/ce_mmbert/scores_{k}.parquet").select("s1", "a", "ce_logit") for k in range(2)]).unique(["s1", "a"]).rename({"ce_logit": "mm2_logit"})
D = D.join(MM2, on=["s1", "a"], how="left").with_columns(mm2_prob=1 / (1 + (-pl.col("mm2_logit")).exp()))
D = D.with_columns(p_x_mm2=pl.col("p") * pl.col("mm2_prob"))
log(f"mmBERT-more coverage {D['mm2_logit'].is_not_null().mean():.4f}")
log(f"big CE coverage {D['big_logit'].is_not_null().mean():.4f}")
A1 = D.filter(pl.col("ce_part").is_in(["A1a", "A1b"])).join(pop.select("s1", "part"), on="s1")
HB = D.filter(pl.col("ce_part") == "holdoutB"); COMP = D.filter(pl.col("ce_part") == "dev")
def nce_margin(df):
    """name-only logit minus the best OTHER name-only logit among the scored S1 of the same aux (competition on the name alone)."""
    g = df.filter(pl.col("nce_logit").is_not_null()).group_by("a").agg(b1=pl.col("nce_logit").max(), b2=pl.col("nce_logit").sort(descending=True).get(1, null_on_oob=True))
    return df.join(g, on="a", how="left").with_columns(
        nce_margin=pl.when(pl.col("nce_logit") >= pl.col("b1")).then(pl.col("nce_logit") - pl.col("b2").fill_null(-10)).otherwise(pl.col("nce_logit") - pl.col("b1"))).drop("b1", "b2")
A1, HB, COMP = nce_margin(A1), nce_margin(pl.concat([HB, COMP])), None
COMP = HB.filter(pl.col("ce_part") == "dev"); HB = HB.filter(pl.col("ce_part") == "holdoutB")
log(f"CE pairs old/new: A1 {A1['old'].sum():,}/{(~A1['old']).sum():,} | holdoutB {HB['old'].sum():,}/{(~HB['old']).sum():,} | competitors {COMP.height:,} | name-only scored {D['nce_logit'].is_not_null().sum():,}")
log(f"A1 {A1.height:,} | holdoutB {HB.height:,} | competitors {COMP.height:,}")

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

# =============================================================== training / decoding (GPU)
BASEF = F_OLD + F_PAIR + F_M3NEW
V5 = BASEF + F_HN + F_ST + F_IDF + F_CMP; F_NCE = ["nce_logit", "nce_margin"]
V1NUM = set(F_HN) | {"cands_same_hn", "num_state", "num1_sim", "num_jaccard"}
V5_HN2 = [f for f in V5 if f not in V1NUM] + F_HN2
F_BIG = ["big_logit", "big_prob", "p_x_big"]; F_MM2 = ["mm2_logit", "mm2_prob", "p_x_mm2"]
VARIANTS = {"final: final + mmBERT-more": (V5_HN2 + F_NCE + F_BIG + F_MM2 + ["ce2_avg"], False)}
def xgb_params(depth=8): return dict(objective="binary:logistic", eval_metric="logloss", device=DEV, tree_method="hist", eta=0.05,
                                     max_depth=depth, min_child_weight=50, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, seed=29)
emptyset = aux.filter(pl.col("empty_addr")).select(pl.col("entity_id").alias("a"))
def scores_for(fused):
    return base.select("s1", "a", "p").join(fused, on=["s1", "a"], how="left").with_columns(pl.col("pf").fill_null(pl.col("p")))
def winners(S):
    X = S.with_columns(r=pl.col("pf").rank("ordinal", descending=True).over("a"))
    return (X.filter(pl.col("r") == 1).join(X.filter(pl.col("r") == 2).select("a", p2="pf"), on="a", how="left")
             .with_columns(pl.col("p2").fill_null(0.0), empty=pl.col("a").is_in(emptyset["a"].implode())))
def gpu_tables(W, target):
    idx = target.select("s1").with_row_index("i")
    w = W.join(idx, on="s1").join(edges.with_columns(ok=pl.lit(True)), on=["s1", "a"], how="left").with_columns(pl.col("ok").fill_null(False))
    t = lambda c, dt=torch.float32: torch.tensor(w[c].to_numpy(), dtype=dt, device=DEV)
    te = edges.join(target.select("s1"), on="s1", how="semi").join(emptyset, on="a", how="semi").height
    return dict(i=t("i", torch.long), pf=t("pf"), p2=t("p2"), empty=t("empty", torch.bool), ok=t("ok", torch.bool),
                card=torch.tensor(target["card"].to_numpy(), dtype=torch.float32, device=DEV),
                us=torch.tensor((target["country"] == "US").to_numpy(), device=DEV), n=target.height, n_empty_true=te)
def summarize(acc, G):
    k = torch.zeros(acc.shape[0], G["n"], device=DEV).index_add_(1, G["i"], acc.float())
    tp = torch.zeros_like(k).index_add_(1, G["i"], (acc & G["ok"][None, :]).float()); c = G["card"][None, :]
    f = torch.where((c == 0) & (k == 0), 1.0, torch.where(tp == 0, 0.0, 1.25 * tp / (1.25 * tp + .25 * (c - tp) + (k - tp))))
    return [{"f05": f[s].mean().item(), "precision": (tp[s].sum() / k[s].sum().clamp(min=1)).item(), "recall": (tp[s].sum() / c.sum()).item(),
             "singleton_acc": f[s][c[0] == 0].mean().item(), "f05_US": f[s][G["us"]].mean().item(), "f05_India": f[s][~G["us"]].mean().item(),
             "empty_addr_found_pct": ((acc[s] & G["ok"] & G["empty"]).sum() / max(1, G["n_empty_true"]) * 100).item()} for s in range(acc.shape[0])]
def grid_eval(G, grid):
    tau = torch.tensor([g[0] for g in grid], device=DEV)[:, None]; dl = torch.tensor([g[1] for g in grid], device=DEV)[:, None]
    return summarize((G["pf"][None, :] >= tau) & ((G["pf"] - G["p2"])[None, :] >= dl), G)
def expf_accept(G, gamma):
    """Expected-F0.5 decoding: per S1, sort its (owner-winning) candidates by p and take the prefix k maximising the plug-in
    expected F0.5, with expected cardinality = gamma * sum(p) over its winners (gamma calibrates missing-candidate mass)."""
    i, p = G["i"].cpu().numpy(), (G["pf"] * (G["pf"] - G["p2"] > 0)).cpu().numpy()
    order = np.lexsort((-p, i)); acc = np.zeros(len(p), bool); start = 0
    while start < len(order):
        end = start
        while end < len(order) and i[order[end]] == i[order[start]]: end += 1
        idx = order[start:end]; ps = p[idx]; C = gamma * ps.sum()
        best_k, best = 0, float(np.prod(1 - ps))                              # k = 0: F = 1 only if no true match
        cum = 0.0
        for k in range(1, len(ps) + 1):
            cum += ps[k - 1]
            fk = 1.25 * cum / (1.25 * cum + 0.25 * max(C - cum, 0) + (k - cum))
            if fk > best: best, best_k = fk, k
        acc[idx[:best_k]] = True; start = end
    return torch.tensor(acc, device=DEV)[None, :]

grid = [(round(t, 2), round(d, 2)) for t in np.arange(.3, .86, .05) for d in np.arange(0, .46, .05)]
target_A1b = pop.filter(pl.col("part") == "A1b"); target_B = pop.filter(pl.col("part") == "holdoutB")
tr = A1.filter(pl.col("part") == "A1a")
results = []
def run_variant(name, feats, old_only, depth=8, rounds=600, expf=False):
    t = time.time()
    keep = (lambda d: d.filter(pl.col("old"))) if old_only else (lambda d: d)
    tr = keep(A1.filter(pl.col("part") == "A1a"))
    bst = xgb.train(xgb_params(depth), xgb.DMatrix(tr.select(feats).to_numpy().astype(np.float32), label=tr["label"].to_numpy(), feature_names=feats), rounds)
    pred = lambda d: d.select("s1", "a").with_columns(pf=pl.Series(bst.inplace_predict(d.select(feats).to_numpy().astype(np.float32))))
    GA = gpu_tables(winners(scores_for(pred(keep(A1)))), target_A1b)
    RA = grid_eval(GA, grid); j = int(np.argmax([r["f05"] for r in RA])); tau, delta = grid[j]
    GB = gpu_tables(winners(scores_for(pl.concat([pred(keep(HB)), pred(keep(COMP))]).unique(["s1", "a"]))), target_B)
    rb = grid_eval(GB, [(tau, delta)])[0]
    import json
    bst.save_model(f"{ER_ROOT}/models/fusion/fusion_final.json")
    json.dump({"features": feats, "tau": float(tau), "delta": float(delta), "A1b_f05": RA[j]["f05"], "holdoutB_f05": rb["f05"]}, open(f"{ER_ROOT}/models/fusion/fusion_final_meta.json", "w"), indent=1)
    results.append({"variant": name, "decoder": f"threshold tau={tau} delta={delta}", "A1b_f05": RA[j]["f05"], **{f"B_{k}": v for k, v in rb.items()}})
    log(f"{name}: A1b {RA[j]['f05']:.5f} | holdoutB {rb['f05']:.5f} | empty found {rb['empty_addr_found_pct']:.1f}% ({time.time()-t:.0f}s)")
    if expf:
        best_g, best_a = None, -1
        for g in [0.9, 1.0, 1.1, 1.2, 1.35, 1.5]:
            fa = summarize(expf_accept(GA, g), GA)[0]["f05"]
            if fa > best_a: best_a, best_g = fa, g
        rbe = summarize(expf_accept(GB, best_g), GB)[0]
        results.append({"variant": name, "decoder": f"expected-F0.5 gamma={best_g}", "A1b_f05": best_a, **{f"B_{k}": v for k, v in rbe.items()}})
        log(f"{name} expected-F0.5: A1b {best_a:.5f} | holdoutB {rbe['f05']:.5f}")
    return bst
for name, (feats, old_only) in VARIANTS.items():
    b = run_variant(name, feats, old_only)
g = b.get_score(importance_type="gain"); tot = sum(g.values())
imp = pl.DataFrame({"feature": list(g), "gain_pct": [v / tot * 100 for v in g.values()]}).sort("gain_pct", descending=True)
imp.write_csv(f"{ER_ROOT}/models/fusion/importance_final.csv"); print(imp.head(25))
R = pl.DataFrame(results).with_columns(pl.col(pl.Float64).round(5))
R.write_csv(f"{ER_ROOT}/models/fusion/results_final.csv")
print(R.select("variant", "decoder", "A1b_f05", "B_f05", "B_f05_US", "B_f05_India", "B_precision", "B_recall", "B_singleton_acc", "B_empty_addr_found_pct"))
log("done (fusion 0.98761)")
