# M002: mDeBERTa-v3 Cross-Encoder with Learned LightGBM Fusion

Sep 27, 2026

## 1. Executive summary

M002 adds a multilingual cross-encoder above the M001 dense-blocking and LightGBM pipeline. It reranks the five strongest M001 candidates for each S1, combines the cross-encoder output with the original M001 probability through a learned LightGBM fusion model, then applies a frozen one-owner decoder.

On untouched holdoutB, M002 improves macro F0.5 from **0.971524 to 0.985680 (+0.014156)**. Both precision and recall improve, with similar gains in the US and India.

| System | Macro F0.5 | US | India | Precision | Recall | Singleton accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M001 | 0.971524 | 0.972753 | 0.969657 | 0.987850 | 0.944850 | 0.947356 |
| **M002** | **0.985680** | **0.985398** | **0.986108** | **0.994686** | **0.962690** | **0.991063** |

The submitted test configuration was hashed and frozen before test access. Test diagnostics were reporting-only and did not change the model, fusion, candidate count, or decoder thresholds.

## 2. End-to-end architecture

```text
M001 dense blocker
  -> about 20 candidates per S1
  -> M001 31-feature LightGBM probability p
  -> select top 5 candidates per S1 by p
  -> mDeBERTa-v3 cross-encoder logit
  -> learned LightGBM fusion probability pf
  -> non-top-5 candidates retain M001 probability p
  -> one owner per auxiliary record
  -> accept when pf >= 0.50 and pf - second_owner_pf >= 0.40
  -> matching_results.tsv
```

M002 does not replace M001 blocking. The cross-encoder only improves pair classification and ownership decisions inside the existing high-recall candidate graph. The complete M001 candidate graph is retained in `candidate_pairs.tsv`.

## 3. Leakage-safe data partitioning

The original frozen split is unchanged:

- `dev`: M001 dense-blocker training and ownership competitors.
- `holdoutA`: matcher development.
- `holdoutB`: final evaluation, untouched until all M002 choices were frozen.

M002 splits holdoutA deterministically by S1:

```python
int(sha1(s1_id)[16:24], 16) % 3
```

| Part | Purpose | Top-5 pairs | Positives |
| --- | --- | ---: | ---: |
| A0 | Train mDeBERTa cross-encoder | 367,075 | 239,529 |
| A1a | Train LightGBM fusion | 367,995 | 239,357 |
| A1b | Tune decoder thresholds | 368,815 | 240,974 |
| holdoutB | Final evaluation only | 1,101,595 | 718,124 |

This separation ensures:

1. A1a cross-encoder scores come from a model trained only on A0.
2. Fusion is trained only on A1a.
3. Decoder thresholds are selected only on A1b.
4. HoldoutB is evaluated once with everything frozen.
5. Test is accessed only after hashes of all frozen components are recorded.

## 4. Candidate construction

Input is M001's `artifacts/scored_pairs.parquet`, containing every dense-blocker candidate and its frozen M001 LightGBM probability `p`.

For each S1:

1. Rank candidates by `p` descending.
2. Retain ranks 1 through 5 for cross-encoder scoring.
3. Attach raw S1 and auxiliary business names and addresses.
4. Retain the ground-truth label for training and validation.

Pairs outside the top five retain their original M001 probability and continue participating in ownership decoding. The top-five choice limits transformer cost while focusing it on pairs most likely to affect final assignment.

Implementation: `experiments/m002_mdeberta/prepare.py`.

## 5. Cross-encoder

### Model

- Checkpoint: `microsoft/mdeberta-v3-base`
- Transformers class: `DebertaV2ForSequenceClassification`
- Parameters: 278,810,113
- License: MIT
- Classification head: one logit
- Tokenizer: mDeBERTa SentencePiece tokenizer
- Local base checkpoint: `experiments/m002_mdeberta/model_base/`

Training and inference use `local_files_only=True`.

### Pair representation

Each pair is passed as two tokenizer sequences:

```text
Sequence A: name: {S1 business_name} | address: {S1 business_address}
Sequence B: name: {aux business_name} | address: {aux business_address}
```

Raw text lets the transformer jointly read punctuation, scripts, token order, house numbers, street types, and address components.

### Training configuration

| Setting | Value |
| --- | --- |
| Training split | A0 only |
| Training pairs | 367,075 |
| Epochs | 1 |
| Batch size | 32 |
| Maximum sequence length | 128 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup | 5% of steps |
| Scheduler | Linear decay |
| Loss | Binary cross-entropy with logits |
| Precision | bfloat16 autocast |
| Gradient clipping | 1.0 |
| Seed | 17 |

The blocker candidates naturally supply hard negatives with similar names, addresses, scripts, or competing owners. The trained checkpoint is `experiments/m002_mdeberta/artifacts/model_A0/`.

Implementation: `experiments/m002_mdeberta/train_score.py`.

## 6. Learned fusion

A fixed weighted average was avoided. The A1a fusion model learns when to trust M001 versus the cross-encoder.

### Fusion features

| Feature | Meaning |
| --- | --- |
| `p` | Frozen M001 LightGBM probability |
| `ce_logit` | Raw mDeBERTa output logit |
| `ce_prob` | Sigmoid of the cross-encoder logit |
| `ce_rank` | Candidate rank within S1 by M001 probability |
| `p_x_ce` | `p * ce_prob` interaction |
| `p_minus_ce` | `p - ce_prob` disagreement |

### Fusion LightGBM

| Setting | Value |
| --- | --- |
| Objective | Binary |
| Trees | 500 |
| Learning rate | 0.03 |
| Leaves | 63 |
| Minimum data per leaf | 200 |
| Feature fraction | 0.9 |
| L2 regularization | 2 |
| Seed | 29 |
| Training data | A1a only |

The output is fused probability `pf`. Top-five pairs receive `pf`; other M001 pairs keep `p`.

Model: `experiments/m002_mdeberta/artifacts/lgbm_fusion_A1a.txt`.

Implementation: `experiments/m002_mdeberta/fuse_eval.py`.

## 7. Ownership decoding

For every auxiliary S2/S3 record:

1. Rank all candidate S1 owners by final score.
2. Keep the best owner.
3. Let `p1` be the best score and `p2` the second-best score, or zero if none exists.
4. Accept only when `p1 >= tau` and `p1 - p2 >= delta`.

The A1b grid searched:

- `tau`: 0.30 through 0.80 in increments of 0.05.
- `delta`: 0.00 through 0.40 in increments of 0.05.

The frozen optimum was:

```text
tau   = 0.50
delta = 0.40
```

A1b at this setting scored 0.985302 macro F0.5, 0.994430 precision, and 0.962086 recall. The decoder was then used unchanged on holdoutB and test.

## 8. HoldoutB evaluation

HoldoutB top-five pairs were scored with A0, fused with the A1a model, and decoded with the A1b thresholds. For realistic ownership competition, 1,188,072 dev candidate pairs directly competing for holdoutB auxiliary records were also cross-encoder scored. Remaining non-material competitors retained M001 scores.

| Metric | M001 | M002 | Change |
| --- | ---: | ---: | ---: |
| Macro F0.5 | 0.971524 | 0.985680 | +0.014156 |
| Precision | 0.987850 | 0.994686 | +0.006836 |
| Recall | 0.944850 | 0.962690 | +0.017840 |
| Singleton accuracy | 0.947356 | 0.991063 | +0.043708 |
| US F0.5 | 0.972753 | 0.985398 | +0.012645 |
| India F0.5 | 0.969657 | 0.986108 | +0.016451 |

The result shows that M002 addresses M001's main weakness—matcher and decoder errors rather than candidate recall. The large singleton improvement indicates much stronger rejection of plausible-looking false matches.

Implementation: `experiments/m002_mdeberta/final_holdoutB.py`.

## 9. Freeze before test

These hashes were recorded before test preparation:

| Component | SHA-256 |
| --- | --- |
| A0 weights | `c230f409910b86af846641e32c5f24947507313c779c55c3bb71bd90d06011f8` |
| A0 config | `605adb015f487666e52a281158001c26fd3115bad0f7e51b4d829b39de1d11a7` |
| A1a fusion | `e71c0668069214a83b9eac825ffc09fa2fe29476ee61c209545d8529d86d9ddd` |
| A1b tuning results | `2220f6274702bba8116e00984ebc7ea5ef46f876679167cd891d28c24d8d6f84` |
| HoldoutB results | `1528b3319044ac2426ef36478d761c630b681e7b10e4755351b9d1cbbef7ce57` |

All hashes were recomputed after test inference and remained unchanged. Full record: `experiments/m002_mdeberta/FROZEN_BEFORE_TEST.md`.

## 10. Test inference

### Candidate preparation

M001 LightGBM rescored all test candidate features. The top five candidates per S1 were selected and joined with raw test text.

| Country | All M001 pairs | mDeBERTa pairs |
| --- | ---: | ---: |
| France | 5,434,893 | 1,297,260 |
| India | 16,669,747 | 4,049,930 |
| US | 13,474,450 | 3,315,530 |
| **Total** | **35,579,090** | **8,662,720** |

Implementation: `experiments/m002_mdeberta/prepare_test.py`.

### Frozen scoring and output

Each country's top-five pairs were scored with the frozen A0 checkpoint. The frozen A1a fusion produced `pf`, and the frozen decoder assigned owners. The full 35.6M candidate graph was written to `candidate_pairs.tsv`.

Reporting-only diagnostics:

| Country | S1 | Predictions / S1 | Predicted empty |
| --- | ---: | ---: | ---: |
| France | 259,452 | 3.271 | 5.97% |
| India | 809,986 | 3.373 | 5.78% |
| US | 663,106 | 3.369 | 5.78% |

France predicts about 0.10 fewer matches per S1. This observation did not cause tuning or a country-specific rule.

Implementation: `experiments/m002_mdeberta/predict_test.py`.

### Submission validation

The official validator ran with `--check-ids`:

```text
required S1 entities:       1,732,544
valid S2/S3 match IDs:      9,969,589
matching_results.tsv rows:  1,732,544
candidate_pairs.tsv rows:   1,732,544
empty candidate rows:       0
result: PASS
```

Outputs:

- `submissions/m002/output/matching_results.tsv` (93 MB)
- `submissions/m002/output/candidate_pairs.tsv` (459 MB)

## 11. Reproduction

Run from `experiments/m002_mdeberta/`.

### Training and validation

```bash
python prepare.py

CUDA_VISIBLE_DEVICES=0 python train_score.py train \
  --parts A0 --model artifacts/model_A0 --batch 32

CUDA_VISIBLE_DEVICES=0 python train_score.py score \
  --parts A1a,A1b --model artifacts/model_A0 \
  --out artifacts/ce_A1.parquet --batch 128

python fuse_eval.py --ce artifacts/ce_A1.parquet

CUDA_VISIBLE_DEVICES=0 python train_score.py score \
  --parts holdoutB --model artifacts/model_A0 \
  --out artifacts/ce_holdoutB.parquet --batch 128

python final_holdoutB.py
```

A compatible local protobuf runtime was exposed where needed:

```bash
export PYTHONPATH=/home/arul24115/miniconda3/envs/py311/lib/python3.11/site-packages
```

### Frozen test run

```bash
python prepare_test.py

CUDA_VISIBLE_DEVICES=0 python train_score.py score \
  --data artifacts_test/ce_candidates__France.parquet \
  --parts test --model artifacts/model_A0 \
  --out artifacts_test/ce_scores__France.parquet --batch 128

CUDA_VISIBLE_DEVICES=0 python train_score.py score \
  --data artifacts_test/ce_candidates__India.parquet \
  --parts test --model artifacts/model_A0 \
  --out artifacts_test/ce_scores__India.parquet --batch 128

CUDA_VISIBLE_DEVICES=0 python train_score.py score \
  --data artifacts_test/ce_candidates__US.parquet \
  --parts test --model artifacts/model_A0 \
  --out artifacts_test/ce_scores__US.parquet --batch 128

python predict_test.py
```

Do not rerun fusion tuning, change thresholds, or select another checkpoint after inspecting test diagnostics.

## 12. Artifact map

```text
experiments/m002_mdeberta/
├── model_base/                    official mDeBERTa-v3-base checkpoint
├── prepare.py                     train candidate/text preparation
├── train_score.py                 cross-encoder training and inference
├── fuse_eval.py                   A1a fusion and A1b decoder tuning
├── final_holdoutB.py              frozen holdoutB evaluation
├── prepare_test.py                frozen test candidate preparation
├── predict_test.py                fusion, decoding, writing, validation
├── FROZEN_BEFORE_TEST.md          pre-test hashes and settings
├── artifacts/
│   ├── ce_candidates.parquet
│   ├── model_A0/
│   ├── ce_A1.parquet
│   ├── lgbm_fusion_A1a.txt
│   ├── tuning_A1b.csv
│   └── final_holdoutB.csv
└── artifacts_test/
    ├── base__{country}.parquet
    ├── ce_candidates__{country}.parquet
    ├── ce_scores__{country}.parquet
    └── country_diagnostics.csv
```

## 13. Limitations and next work

- France has no labeled validation data. Its test prediction count is lower, but thresholds must not be changed from this observation.
- Cross-encoder inference dominates runtime. Dynamic padding makes very large batches inefficient when address lengths vary; batch 128 was reliable.
- A0 uses one third of holdoutA. Training on more data would change calibration and require new leakage-safe fusion scores and a fresh isolated validation design.
- M002 cross-encodes only the top five. It cannot recover a true match outside M001's candidate graph, and lower-ranked pairs retain M001 scoring.
- The current fusion is intentionally small. Adding original M001 pair features could help but requires a new experiment.
- HoldoutB is now a reported result, not a future tuning set. Any M003 work should use new development partitions or cross-fitted predictions.

