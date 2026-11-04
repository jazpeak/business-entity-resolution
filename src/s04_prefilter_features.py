"""Pair features for every blocker candidate (dense score, ranks, reverse flags, aux competition gap/margin, name / transliterated / skeleton /
address string similarities, house-number agreement). ER_MODE=train (labelled train pairs) or ER_MODE=test."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import hashlib, os, re, sys, time
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load, COUNTRIES  # noqa: E402
from textviews import v_norm, v_translit, v_skel, v_addr_abbr  # noqa: E402

OUT = "artifacts"
REV_CAP = 10
T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
_NUM = re.compile(r"\d+")


def part_of(ids, split):
    """holdout S1 are split in half by a second hash: A = matcher training/tuning, B = final evaluation."""
    sp = dict(zip(split["source1_entity_id"].to_list(), split["split"].to_list()))
    out = []
    for e in ids:
        if sp[e] == "holdout":
            out.append("holdoutA" if int(hashlib.sha1(e.encode()).hexdigest()[8:16], 16) % 2 == 0 else "holdoutB")
        else:
            out.append("dev")
    return out


def _views(names, addrs):
    return ([v_norm(x) for x in names], [v_translit(x) for x in names], [v_skel(x) for x in names],
            [v_addr_abbr(x) for x in addrs])


def record_views(df):
    """Per-record text views, computed once per record (not per pair)."""
    n, a = df["business_name"].to_list(), df["business_address"].to_list()
    step = len(n) // 48 + 1
    res = Parallel(n_jobs=48)(delayed(_views)(n[i:i + step], a[i:i + step]) for i in range(0, len(n), step))
    cols = [sum((r[k] for r in res), []) for k in range(4)]
    first = [(m.group() if (m := _NUM.search(x)) else "") for x in a]
    return df.select("entity_id").with_columns(nn=pl.Series(cols[0]), tn=pl.Series(cols[1]), sk=pl.Series(cols[2]),
                                               ad=pl.Series(cols[3]), num1=pl.Series(first),
                                               nums=pl.Series([" ".join(sorted(set(_NUM.findall(x)))) for x in a]))


def string_features(P):
    kw = dict(workers=-1, dtype=np.float32)
    g = lambda c: P[c].to_list()
    f = {"nm_ratio": cpdist(g("nn_1"), g("nn_2"), scorer=fuzz.ratio, **kw),
         "nm_tset": cpdist(g("nn_1"), g("nn_2"), scorer=fuzz.token_set_ratio, **kw),
         "nm_tsort": cpdist(g("nn_1"), g("nn_2"), scorer=fuzz.token_sort_ratio, **kw),
         "nm_partial": cpdist(g("nn_1"), g("nn_2"), scorer=fuzz.partial_ratio, **kw),
         "tr_tset": cpdist(g("tn_1"), g("tn_2"), scorer=fuzz.token_set_ratio, **kw),
         "sk_ratio": cpdist(g("sk_1"), g("sk_2"), scorer=fuzz.ratio, **kw),
         "ad_ratio": cpdist(g("ad_1"), g("ad_2"), scorer=fuzz.ratio, **kw),
         "ad_tset": cpdist(g("ad_1"), g("ad_2"), scorer=fuzz.token_set_ratio, **kw),
         "ad_partial": cpdist(g("ad_1"), g("ad_2"), scorer=fuzz.partial_ratio, **kw),
         "num1_sim": cpdist(g("num1_1"), g("num1_2"), scorer=Levenshtein.normalized_similarity, **kw)}
    n1, n2 = g("num1_1"), g("num1_2")
    s1n, s2n = g("nums_1"), g("nums_2")
    jac = np.empty(len(n1), np.float32); pre = np.empty(len(n1), np.int8)
    for i, (x, y, a, b) in enumerate(zip(n1, n2, s1n, s2n)):
        A, B = set(a.split()), set(b.split())
        jac[i] = len(A & B) / len(A | B) if (A or B) else -1.0
        pre[i] = 1 if (x and y and x != y and (x.startswith(y) or y.startswith(x))) else 0
    num_state = np.where([not x or not y for x, y in zip(n1, n2)], 2, np.where([x == y for x, y in zip(n1, n2)], 1, 0)).astype(np.int8)
    return P.with_columns(**{k: pl.Series(v) for k, v in f.items()}, num_jaccard=pl.Series(jac), num_prefix=pl.Series(pre),
                          num_state=pl.Series(num_state))   # num_state: 1 equal, 0 conflict, 2 missing on a side


MODE = os.environ.get("ER_MODE", "train")          # "train" (labelled, holdout parts) or "test" (unlabelled)
if MODE == "test":
    from erdata import rd
    T = f"{ER_ROOT}/data/test"
    OUT = "artifacts_test"
    NONLATIN = r"[^\p{Latin}\p{Common}\p{Inherited}\s]"
    s1 = rd(f"{T}/test_source1.tsv")
    aux = pl.concat([rd(f"{T}/test_source2.tsv").with_columns(src=pl.lit("S2")), rd(f"{T}/test_source3.tsv").with_columns(src=pl.lit("S3"))])
    aux = aux.with_columns(native=pl.col("business_name").str.contains(NONLATIN), empty_addr=pl.col("business_address") == "")
    edges = pl.DataFrame(schema={"s1": pl.Utf8, "a": pl.Utf8})
    split = None
    COUNTRY_LIST = sorted(s1["country"].unique().to_list())   # open set: France included
    part_of = lambda ids, split: ["test"] * len(ids)
else:
    s1, aux, edges, split = load()
    COUNTRY_LIST = COUNTRIES
for c in COUNTRY_LIST:
    out = f"{OUT}/pairs__{c}.parquet"
    if os.path.exists(out):
        continue
    fwd, rev = pl.read_parquet(f"{OUT}/fwd__{c}.parquet"), pl.read_parquet(f"{OUT}/rev__{c}.parquet")
    # aux-level competition: its best and second-best S1 over the whole country
    top = rev.pivot(on="rank", index="a", values=["s1", "score"], aggregate_function="first")
    top = top.select("a", aux_top1_s1="s1_1", aux_top1_score="score_1", aux_top2_score=pl.col("score_2").fill_null(0.0))
    rcand = (rev.filter(pl.col("rank") == 1).with_columns(rr=pl.col("score").rank("ordinal", descending=True).over("s1"))
                .filter(pl.col("rr") <= REV_CAP).select("s1", "a", rev_score="score"))
    C = (fwd.rename({"rank": "fwd_rank", "score": "fwd_score"})
            .join(rcand, on=["s1", "a"], how="full", coalesce=True)
            .with_columns(score=pl.coalesce("fwd_score", "rev_score"), fwd_rank=pl.col("fwd_rank").fill_null(99).cast(pl.Int16),
                          is_rev=pl.col("rev_score").is_not_null().cast(pl.Int8))
            .drop("fwd_score", "rev_score")
            .join(top, on="a", how="left")
            .with_columns(is_aux_top1=(pl.col("aux_top1_s1") == pl.col("s1")).cast(pl.Int8),
                          gap_aux=pl.col("score") - pl.col("aux_top1_score"),
                          aux_margin=pl.col("aux_top1_score") - pl.col("aux_top2_score"),
                          s1_max=pl.col("score").max().over("s1"), s1_n=pl.len().over("s1"),
                          rank_in_s1=pl.col("score").rank("ordinal", descending=True).over("s1").cast(pl.Int16),
                          aux_fanout=pl.len().over("a"))
            .with_columns(gap_s1=pl.col("score") - pl.col("s1_max"))
            .drop("aux_top1_s1"))
    log(f"{c}: {C.height:,} candidate pairs for {C['s1'].n_unique():,} S1")
    V1 = record_views(s1.filter(pl.col("country") == c)); log(f"{c}: S1 views")
    A = aux.filter(pl.col("country") == c)
    V2 = record_views(A).join(A.select("entity_id", "src", "native", "empty_addr", nlen=pl.col("business_name").str.len_chars()), on="entity_id")
    log(f"{c}: aux views")
    C = (C.join(V1.rename({k: f"{k}_1" for k in V1.columns if k != "entity_id"}).rename({"entity_id": "s1"}), on="s1")
          .join(V2.rename({k: f"{k}_2" for k in ["nn", "tn", "sk", "ad", "num1", "nums"]}).rename({"entity_id": "a"}), on="a"))
    C = C.join(edges.with_columns(label=pl.lit(1, pl.Int8)), on=["s1", "a"], how="left").with_columns(pl.col("label").fill_null(0))
    ids = C["s1"].unique()
    C = C.join(pl.DataFrame({"s1": ids, "part": part_of(ids.to_list(), split)}), on="s1")
    chunks = []
    for i in range(0, C.height, 4_000_000):
        chunks.append(string_features(C.slice(i, 4_000_000)))
        log(f"{c}: string features {min(i + 4_000_000, C.height):,}/{C.height:,}")
    C = pl.concat(chunks).with_columns(is_s3=(pl.col("src") == "S3").cast(pl.Int8), native_i=pl.col("native").cast(pl.Int8),
                                       empty_addr_i=pl.col("empty_addr").cast(pl.Int8),
                                       len_ratio=pl.col("nn_1").str.len_chars().cast(pl.Float32) / pl.col("nn_2").str.len_chars().clip(1))
    C.drop([x for x in C.columns if x.endswith(("_1", "_2")) and x not in ("nm_1",)] + ["src", "native", "empty_addr"]).write_parquet(out)
    log(f"{c}: wrote {out}")
log("done")
