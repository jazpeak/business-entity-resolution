"""Dev S1 top-5 cross-encoder candidates whose aux is also a holdoutB top-5 candidate: the dev 'competitors' scored so that
holdoutB decoding sees every S1 competing for an aux, as on test (verified: reproduces 1,188,072 rows).
Run from work/ce_candidates.  -> artifacts/ce_dev_competitors_B.parquet"""
import polars as pl
C = pl.read_parquet("artifacts/ce_candidates.parquet")
hb_aux = C.filter(pl.col("ce_part") == "holdoutB").select("a").unique()
comp = C.filter(pl.col("ce_part") == "dev").join(hb_aux, on="a", how="semi")
comp.write_parquet("artifacts/ce_dev_competitors_B.parquet", compression="zstd"); print("dev competitors", comp.height)
