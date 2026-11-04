#!/bin/bash
# Full retraining of every model from the TRAIN data (needs data/train/*.tsv). Writes the trained models into models/.
# Run this first; then run_inference.sh. ~15-20 h on 1-2 A100 GPUs.
# Pretrained bases are downloaded once: intfloat/multilingual-e5-base (MIT), microsoft/mdeberta-v3-base (MIT), jhu-clsp/mmBERT-base (MIT).
set -euo pipefail
export ER_ROOT="${ER_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
PY="${PY:-python}"; S=$ER_ROOT/src; W=$ER_ROOT/work; M=$ER_ROOT/models
mkdir -p $W/splits $W/blocker $W/prefilter $W/ce_candidates $W/ce_tagged $W/shortlist $W/ce_street $W/ce_bigdata $W/ce_mmbert $M
[ -d $M/base_mdeberta_v3 ] || $PY -c "from huggingface_hub import snapshot_download as d; d('microsoft/mdeberta-v3-base', local_dir='$M/base_mdeberta_v3')"
echo "[1] frozen entity-level split (dev 80% / holdout 20% by sha1(entity_id))";  ( cd $W/splits && $PY $S/s00_make_split.py )
echo "[2] fine-tune the e5 bi-encoder blocker on dev pairs";                      ( cd $W/blocker && $PY $S/s01_train_blocker.py --out $M/blocker_e5 )
echo "[3] embed train S1/aux";  for c in US India; do ( cd $W/blocker && $PY $S/s02_embed_train.py --model $M/blocker_e5 --tag e5b_dev_v1 --country $c ); done
echo "[4] blocking for all train S1 (FAISS IVF)";                               ( cd $W/prefilter && $PY $S/s03_block_train.py )
echo "[5] prefilter features + LightGBM (holdoutA, 2-fold OOF)";                     ( cd $W/prefilter && ER_MODE=train $PY $S/s04_prefilter_features.py && $PY $S/s05_prefilter_train.py )
cp $W/prefilter/artifacts/prefilter_lgbm.txt $M/prefilter_lgbm.txt
echo "[6] cross-encoder candidates (A0/A1a/A1b/holdoutB/dev) + dev competitors"; ( cd $W/ce_candidates && $PY $S/s06_ce_candidates.py && $PY $S/s06b_dev_competitors.py )
echo "[7] [MISSING]-tag text + tagged cross-encoder (train A0, score A1/holdoutB/competitors)"
( cd $W/ce_tagged && $PY $S/s07_tag_missing.py )
T=$W/ce_tagged/artifacts
$PY $S/ce_mdeberta.py train --parts A0 --model $M/ce_tagged --data $T/ce_candidates_tag.parquet
$PY $S/ce_mdeberta.py score --parts A1a,A1b --model $M/ce_tagged --data $T/ce_candidates_tag.parquet --out $T/ce_A1.parquet --batch 256
$PY $S/ce_mdeberta.py score --parts holdoutB --model $M/ce_tagged --data $T/ce_candidates_tag.parquet --out $T/ce_holdoutB.parquet --batch 256
$PY $S/ce_mdeberta.py score --parts dev --model $M/ce_tagged --data $T/ce_dev_competitors_B_tag.parquet --out $T/ce_dev_competitors_B_scores.parquet --batch 256
echo "[8] wider shortlist + name-only cross-encoder (clean-pool empty-address data)"
( cd $W/shortlist && $PY $S/s08_shortlist_nameonly_data.py )
L=$W/shortlist/artifacts
$PY $S/ce_mdeberta.py score --parts A1a,A1b,holdoutB,dev --model $M/ce_tagged --data $L/ce_new_tag.parquet --out $L/ce_new_scores.parquet --batch 256
$PY $S/ce_mdeberta.py train --parts train --model $M/ce_nameonly --data $L/nce_train.parquet --batch 64
$PY $S/ce_mdeberta.py score --parts A1a,A1b,holdoutB,dev --model $M/ce_nameonly --data $L/nce_score.parquet --out $L/nce_scores.parquet --batch 512
echo "[9] street-aware cross-encoder (clean dev pool + same-name look-alike negatives)"
( cd $W/ce_street && $PY $S/s09_street_ce_data.py && $PY $S/s09b_score_inputs.py )
B=$W/ce_street/artifacts
$PY -c "import polars as pl; t=pl.read_parquet('$B/train.parquet'); m=pl.read_parquet('$T/ce_candidates_tag.parquet',columns=['s1','a','ce_part']).filter(pl.col('ce_part')=='A0'); \
x=t.join(m.select('s1','a'),on=['s1','a'],how='anti'); pos=x.filter(pl.col('label')==1).sample(300_000,seed=1); \
S=pl.read_parquet('$W/prefilter/artifacts/scored_pairs.parquet',columns=['s1','a','p']).join(t.select('s1').unique(),on='s1',how='semi').with_columns(r=pl.col('p').rank('ordinal',descending=True).over('s1')); \
S=S.with_columns(sl=(pl.col('r')<=5)|((pl.col('r')<=10)&(pl.col('p')>=0.01))).select('s1','a','sl'); hard=x.join(S,on=['s1','a'],how='left').filter((pl.col('label')==0)&~pl.col('sl')).drop('sl'); \
pl.concat([t.join(m.select('s1','a'),on=['s1','a'],how='semi'),hard,pos]).sample(fraction=1.0,shuffle=True,seed=5).write_parquet('$B/train_1m.parquet')"
$PY $S/ce_mdeberta.py train --parts train --model $M/ce_street --data $B/train_1m.parquet --batch 64
for k in 0 1; do $PY $S/ce_mdeberta.py score --parts A1a,A1b,holdoutB,dev --model $M/ce_street --data $B/score_in_$k.parquet --out $B/big_scores_$k.parquet --batch 256; done
echo "[10] mmBERT cross-encoder on the big clean set (2.9M pairs: all look-alike negatives, suffix-variation matches up-weighted)"
( cd $W/ce_bigdata && $PY $S/s10_bigdata_ce_data.py )
$PY $S/ce_mmbert.py train --parts train --data $W/ce_bigdata/artifacts/train.parquet --model $M/ce_mmbert --batch 64
for k in 0 1; do $PY $S/ce_mmbert.py score --parts A1a,A1b,holdoutB,dev --data $B/score_in_$k.parquet --model $M/ce_mmbert --out $W/ce_mmbert/scores_$k.parquet --batch 256 --offline; done
echo "[11] fusion (XGBoost on A1a) + decoder tuning (A1b) + holdoutB report -> models/fusion/"
mkdir -p $M/fusion; ( cd $W && $PY $S/s11_fusion_train.py )
