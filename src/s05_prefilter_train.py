"""LightGBM prefilter on the candidate features, trained on holdoutA with 2-fold out-of-fold scores; writes prefilter_lgbm.txt and the scored
candidate pairs (scored_pairs.parquet: s1, a, label, part, p) used by all later stages."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import hashlib, os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb

sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import load, COUNTRIES  # noqa: E402

OUT = "artifacts"
T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(220)
FEATURES = ["score", "fwd_rank", "is_rev", "is_aux_top1", "gap_aux", "aux_margin", "aux_top1_score", "aux_top2_score", "s1_max", "s1_n",
            "rank_in_s1", "aux_fanout", "gap_s1", "nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "tr_tset", "sk_ratio", "ad_ratio",
            "ad_tset", "ad_partial", "num1_sim", "num_jaccard", "num_prefix", "num_state", "is_s3", "native_i", "empty_addr_i",
            "len_ratio", "nlen"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=48, verbose=-1, seed=0)

s1, aux, edges, split = load()
card = split.select(pl.col("source1_entity_id").alias("s1"), "card", "country")
P = pl.concat([pl.read_parquet(f"{OUT}/pairs__{c}.parquet", columns=["s1", "a", "label", "part"] + FEATURES) for c in COUNTRIES])
log(f"pairs {P.height:,}; parts {P.group_by('part').len().to_dicts()}")

# ---- training: holdoutA, 2-fold OOF by S1 hash
A = P.filter(pl.col("part") == "holdoutA").with_columns(
    fold=pl.col("s1").map_elements(lambda e: int(hashlib.sha1(e.encode()).hexdigest()[16:24], 16) % 2, return_dtype=pl.Int8))
oof = np.zeros(A.height, np.float32)
X, y, f = A.select(FEATURES).to_numpy().astype(np.float32), A["label"].to_numpy(), A["fold"].to_numpy()
rounds = []
for k in [0, 1]:
    tr, va = f != k, f == k
    b = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), 4000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = b.predict(X[va], num_iteration=b.best_iteration); rounds.append(b.best_iteration)
    log(f"fold {k}: {b.best_iteration} trees")
final = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=FEATURES), int(np.mean(rounds) * 1.1))
final.save_model(f"{OUT}/prefilter_lgbm.txt")
imp = pl.DataFrame({"feature": FEATURES, "gain": final.feature_importance("gain")}).sort("gain", descending=True)
print(imp.head(15))
rest = P.filter(pl.col("part") != "holdoutA")
S = pl.concat([A.select("s1", "a", "label", "part").with_columns(p=pl.Series(oof)),
               rest.select("s1", "a", "label", "part").with_columns(
                   p=pl.Series(final.predict(rest.select(FEATURES).to_numpy().astype(np.float32)).astype(np.float32)))])
S.write_parquet(f"{OUT}/scored_pairs.parquet")
log("all pairs scored")


def decode(S, tau, mode, delta=0.0):
    """Returns accepted (s1, a) pairs. mode: 'indep' | 'owner' | 'owner_margin'."""
    X = S.filter(pl.col("p") >= tau) if mode == "indep" else S
    if mode == "indep":
        return X.select("s1", "a")
    X = X.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("a"))
    best = X.filter(pl.col("r") == 1)
    if mode == "owner_margin":
        second = X.filter(pl.col("r") == 2).select("a", p2="p")
        best = best.join(second, on="a", how="left").with_columns(pl.col("p2").fill_null(0.0)).filter(pl.col("p") - pl.col("p2") >= delta)
    return best.filter(pl.col("p") >= tau).select("s1", "a")


def f05(accepted, part):
    ids = card.join(S.filter(pl.col("part") == part).select("s1").unique(), on="s1", how="semi") \
        if False else card.join(P.filter(pl.col("part") == part).select("s1").unique(), on="s1", how="semi")
    # S1 of this part with no candidate at all still count (predicted empty)
    allp = split.filter(pl.col("split") == "holdout").select(pl.col("source1_entity_id").alias("s1"), "card", "country")
    allp = allp.join(pl.DataFrame({"s1": allp["s1"], "part": [("holdoutA" if int(hashlib.sha1(e.encode()).hexdigest()[8:16], 16) % 2 == 0 else "holdoutB")
                                                             for e in allp["s1"].to_list()]}), on="s1").filter(pl.col("part") == part)
    acc = accepted.join(allp.select("s1"), on="s1", how="semi")
    tp = acc.join(edges, on=["s1", "a"], how="semi").group_by("s1").agg(t=pl.len())
    k = acc.group_by("s1").agg(k=pl.len())
    per = (allp.join(tp, on="s1", how="left").join(k, on="s1", how="left").with_columns(pl.col("t").fill_null(0), pl.col("k").fill_null(0))
               .with_columns(f=pl.when((pl.col("card") == 0) & (pl.col("k") == 0)).then(1.0)
                                .when(pl.col("t") == 0).then(0.0)
                                .otherwise(1.25 * pl.col("t") / (1.25 * pl.col("t") + 0.25 * (pl.col("card") - pl.col("t")) + (pl.col("k") - pl.col("t"))))))
    r = {"f05": per["f"].mean()}
    for c in COUNTRIES:
        r[f"f05_{c}"] = per.filter(pl.col("country") == c)["f"].mean()
    r["singleton_acc"] = per.filter(pl.col("card") == 0)["f"].mean()
    r["precision"] = per["t"].sum() / max(1, per["k"].sum()); r["recall"] = per["t"].sum() / per["card"].sum()
    r["pred_per_s1"] = per["k"].mean(); r["true_per_s1"] = per["card"].mean()
    return r, per


# ---- tune on holdoutA (OOF scores, all S1 competing)
rows = []
for mode, deltas in [("indep", [0.0]), ("owner", [0.0]), ("owner_margin", [0.05, 0.1, 0.2, 0.3])]:
    for d in deltas:
        for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            r, _ = f05(decode(S, tau, mode, d), "holdoutA")
            rows.append({"mode": mode, "delta": d, "tau": tau, **r})
T = pl.DataFrame(rows).sort("f05", descending=True)
T.write_csv(f"{OUT}/tuning_holdoutA.csv")
print(T.head(15).with_columns(pl.col(pl.Float64).round(4)))
best = T.row(0, named=True)
log(f"best on holdoutA: {best['mode']} tau={best['tau']} delta={best['delta']} F0.5={best['f05']:.4f}")

# ---- final: holdoutB, parameters frozen
res = {}
for mode, tau, d in [(best["mode"], best["tau"], best["delta"]), ("indep", 0.5, 0.0), ("owner", 0.5, 0.0)]:
    r, per = f05(decode(S, tau, mode, d), "holdoutB")
    res[f"{mode} tau={tau} delta={d}"] = r
    if (mode, tau, d) == (best["mode"], best["tau"], best["delta"]):
        per.write_parquet(f"{OUT}/per_s1_holdoutB.parquet")
        print("holdoutB by true cardinality:")
        print(per.with_columns(cb=pl.col("card").clip(0, 6)).group_by("cb").agg(pl.len(), f05=pl.col("f").mean()).sort("cb"))
# candidate-set oracle for holdoutB
cands = P.filter(pl.col("part") == "holdoutB").select("s1", "a")
res["ORACLE (blocker ceiling)"] = f05(cands.join(edges, on=["s1", "a"], how="semi"), "holdoutB")[0]
F = pl.DataFrame([{"system": k, **v} for k, v in res.items()])
F.write_csv(f"{OUT}/final_holdoutB.csv")
print(F.with_columns(pl.col(pl.Float64).round(4)))
log("done")
