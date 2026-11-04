#!/bin/bash
# Regenerate output/matching_results.tsv + output/candidate_pairs.tsv from the TEST data with the trained models in models/ (created by run_training.sh).
# Requires: data/test/test_source{1,2,3}.tsv (see README), 1 NVIDIA GPU (>= 24 GB), ~6-8 h on one A100.
set -euo pipefail
export ER_ROOT="${ER_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY="${PY:-python}"; S=$ER_ROOT/src; W=$ER_ROOT/work; M=$ER_ROOT/models
mkdir -p $W/prefilter $W/ce_candidates $W/test $W/ce_mmbert/test $ER_ROOT/output
echo "[1/6] dense blocking (fine-tuned e5 + FAISS IVF): forward top-10/source + reverse top-1 (cap 10)"
( cd $W/prefilter && $PY $S/t01_block_test.py )
echo "[2/6] prefilter pair features for every blocker candidate"
( cd $W/prefilter && ER_MODE=test $PY $S/s04_prefilter_features.py )
echo "[3/6] LightGBM prefilter probability p for every candidate"
( cd $W/ce_candidates && $PY $S/t02_prefilter_score_test.py )
echo "[4/6] cross-encoder shortlist: prefilter top-5 + ranks 6-10 with p >= 0.01"
( cd $W/test && $PY $S/t03_prep_test.py )
echo "[5/6] cross-encoder scores (4 models)"
A=$W/test/artifacts
for k in 0 1; do
  $PY $S/ce_mdeberta.py score --parts test --model $M/ce_tagged --data $A/ce_test_$k.parquet --out $A/ce_test_scores_$k.parquet --batch 256
  $PY $S/ce_mdeberta.py score --parts test --model $M/ce_street --data $A/ce_test_$k.parquet --out $A/big_test_scores_$k.parquet --batch 256
  $PY $S/ce_mmbert.py   score --parts test --model $M/ce_mmbert --data $A/ce_test_$k.parquet --out $W/ce_mmbert/test/scores_$k.parquet --batch 256 --offline
done
$PY $S/ce_mdeberta.py score --parts test --model $M/ce_nameonly --data $A/nce_test.parquet --out $A/nce_test_scores.parquet --batch 512
echo "[6/6] features + XGBoost fusion + one-owner decoding -> output/ (+ official validator)"
$PY $S/t04_predict.py
