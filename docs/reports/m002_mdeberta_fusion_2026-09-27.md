# M002 — mDeBERTa-v3 Cross-Encoder + LightGBM Fusion

## Result

M002 improves frozen holdoutB macro F0.5 from **0.971524 to 0.985680 (+0.014156)**.

| System | F0.5 | US | India | Precision | Recall | Singleton accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M001 | 0.971524 | 0.972753 | 0.969657 | 0.987850 | 0.944850 | 0.947356 |
| M002 mDeBERTa fusion | **0.985680** | **0.985398** | **0.986108** | **0.994686** | **0.962690** | **0.991063** |

This is the strongest local result so far. Both precision and recall improve substantially, and the gain is balanced across countries.

## Design

- Base model: `microsoft/mdeberta-v3-base`, 278.8M parameters, MIT license.
- Cross-encoder input: raw name and address from each S1/candidate pair.
- Candidate scope: top 5 M001 LightGBM candidates per S1.
- Maximum sequence length: 128; bf16; batch 32; one epoch; learning rate 2e-5.
- Cross-encoder training: 367,075 A0 pairs, including 239,529 positives and blocker-derived hard negatives.
- Learned fusion: LightGBM on A1a using M001 probability, cross-encoder logit/probability, rank, and interactions.
- Decoder tuning: disjoint A1b; optimum `tau=0.50`, `delta=0.40`, F0.5 0.985302.
- Final evaluation: holdoutB touched once after configuration freeze.

For holdoutB ownership, cross-encoder fusion was applied to holdoutB candidates and 1,188,072 directly competing dev candidates. Remaining competitors retain calibrated M001 probabilities. This covers the material full-density competition without scoring unrelated dev pairs.

## Recommendation

Proceed to M002 test inference with the frozen A0 cross-encoder, A1a fusion model, top-5-per-S1 scope, and `tau=0.50`, `delta=0.40`. Score all test top-5 candidates because every test S1 is in the target population. Check France score, acceptance, and empty-rate distributions before submission.

## Files

- `experiments/m002_mdeberta/prepare.py`
- `experiments/m002_mdeberta/train_score.py`
- `experiments/m002_mdeberta/fuse_eval.py`
- `experiments/m002_mdeberta/final_holdoutB.py`
- `experiments/m002_mdeberta/artifacts/model_A0/`
- `experiments/m002_mdeberta/artifacts/lgbm_fusion_A1a.txt`
- `experiments/m002_mdeberta/artifacts/tuning_A1b.csv`
- `experiments/m002_mdeberta/artifacts/final_holdoutB.csv`
