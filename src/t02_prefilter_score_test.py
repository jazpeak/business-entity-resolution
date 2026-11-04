"""Score every test blocker candidate with the LightGBM prefilter (p) -> artifacts_test/base__{country}.parquet."""
import os as _os
ER_ROOT = _os.environ.get("ER_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # package root (code/business_entity_resolution)

import os, sys
import lightgbm as lgb
import numpy as np
import polars as pl
sys.path.insert(0, f"{ER_ROOT}/src/common")
from erdata import rd

TEST=f"{ER_ROOT}/data/test"
P1="../prefilter/artifacts_test"
os.makedirs("artifacts_test",exist_ok=True)
model=lgb.Booster(model_file=f"{ER_ROOT}/models/prefilter_lgbm.txt"); features=model.feature_name()
s1=rd(f"{TEST}/test_source1.tsv")
aux=pl.concat([rd(f"{TEST}/test_source2.tsv"),rd(f"{TEST}/test_source3.tsv")])
for country in sorted(s1["country"].unique().to_list()):
    P=pl.read_parquet(f"{P1}/pairs__{country}.parquet",columns=["s1","a"]+features)
    B=P.select("s1","a").with_columns(p=pl.Series(model.predict(P.select(features).to_numpy().astype(np.float32)).astype(np.float32)))
    B.write_parquet(f"artifacts_test/base__{country}.parquet")
    C=(B.with_columns(ce_rank=pl.col("p").rank("ordinal",descending=True).over("s1")).filter(pl.col("ce_rank")<=5)
       .join(s1.filter(pl.col("country")==country).select(s1="entity_id",name1="business_name",addr1="business_address"),on="s1")
       .join(aux.filter(pl.col("country")==country).select(a="entity_id",name2="business_name",addr2="business_address"),on="a")
       .with_columns(label=pl.lit(0,pl.Int8),ce_part=pl.lit("test")))
    C.write_parquet(f"artifacts_test/ce_candidates__{country}.parquet",compression="zstd")
    print(country,B.height,C.height,flush=True)
