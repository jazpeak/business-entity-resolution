# M003 OOF3 LightGBM self-check — 2026-09-27

## Outcome

The three leakage-safe LightGBM fits completed and produced `p_new_oof3` for all 4,474,602 holdoutA candidate rows. Each held partition was score-only: its S1 IDs were excluded from fitting, and the models used the frozen M001 configuration and 766 trees without early stopping.

The mandatory stop conditions did not trigger:

- maximum fold mean shift versus old OOF2: 0.000181
- worst log-loss change versus old OOF2: -0.000545 (an improvement)
- maximum US/India delta difference: 0.000154
- maximum negative q99 increase: 0.001584
- full-graph owner churn versus old OOF2: 0.1843%

No K8 candidate generation or mDeBERTa training was started.

## Step 1 gate

Applying the saved final `lgbm_m001.txt` to A0 and A1a confirmed that the probability already stored in `scored_pairs.parquet` is materially different from final-model in-sample scoring.

| Fold | rows | MAD | median | p95 | p99 | Spearman | >0.01 | >0.05 | >0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1,488,007 | 0.004591 | 0.000015 | — | — | 0.989604 | 6.484% | 2.220% | 1.210% |
| A1a | 1,491,736 | 0.004571 | 0.000015 | — | — | 0.989550 | 6.488% | 2.219% | 1.199% |

Aggregate performance also diverged. On A0, old OOF2 had log loss 0.031504, ROC-AUC 0.998549 and AP 0.994415; final-model in-sample scoring had 0.020786, 0.999568 and 0.998038. This passed the specified Step 2 gate. It does not mean old `p` was in-sample: it confirms the old 2-fold OOF and final in-sample predictions are not equivalent.

## OOF3 fits

| Scored fold | validation rows | validation S1 | training rows | positives |
|---|---:|---:|---:|---:|
| A0 | 1,488,007 | 73,415 | 2,986,595 | 252,838 |
| A1a | 1,491,736 | 73,599 | 2,982,866 | 252,466 |
| A1b | 1,494,859 | 73,763 | 2,979,743 | 254,446 |

All train/validation S1 disjointness assertions passed. Total fit-and-score runtime was 387.3 seconds (6.46 minutes).

## Fold-level quality

| Fold | version | mean p | log loss | ROC-AUC | AP | fraction >0.5 | fraction >0.6 |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | new OOF3 | 0.169404 | 0.030887 | 0.998605 | 0.994620 | 0.166751 | 0.164634 |
| A0 | old OOF2 | 0.169342 | 0.031504 | 0.998549 | 0.994415 | 0.166763 | 0.164667 |
| A1a | new OOF3 | 0.169040 | 0.031011 | 0.998602 | 0.994542 | 0.166376 | 0.164217 |
| A1a | old OOF2 | 0.168859 | 0.031555 | 0.998546 | 0.994363 | 0.166274 | 0.164131 |
| A1b | new OOF3 | 0.169590 | 0.030984 | 0.998579 | 0.994576 | 0.166976 | 0.164898 |
| A1b | old OOF2 | 0.169578 | 0.031561 | 0.998524 | 0.994389 | 0.166980 | 0.164902 |

OOF3 is slightly better than OOF2 on all three folds without a large calibration shift. Final-model in-sample log loss is approximately 0.0207 and is, as expected, optimistically better than either OOF estimate.

## Paired differences across all rows

| Comparator | signed mean | MAD | median | p95 | p99 | max | Pearson | Spearman | >0.01 | >0.05 | >0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old OOF2 | 0.000085 | 0.003767 | 0.000015 | 0.015821 | 0.090396 | 0.688246 | 0.998985 | 0.988158 | 6.640% | 2.022% | 0.860% |
| old final in-sample | -0.000442 | 0.004173 | 0.000014 | 0.013563 | 0.107547 | 0.872252 | 0.997916 | 0.991139 | 6.029% | 2.019% | 1.083% |

The fold-level OOF3-versus-OOF2 MAD range is 0.003750–0.003791, and the fraction above 0.05 is 2.010–2.038%; no single fold dominates the change.

## Boundary, ranking, and owner behavior

- Per fold/country, directional crossings of 0.5 or 0.6 versus OOF2 are each 0.097–0.132% of rows.
- Versus OOF2, top-5 row-membership churn is 2.43–2.68%; top-8 churn is 4.51–6.05%. Between 23.7–24.9% of S1 top-5 sets and 41.8–50.8% of top-8 sets change at least one member. This is notable candidate-boundary movement even though most changed sets differ by few rows.
- On the complete 10,313,397-aux graph, owner churn versus OOF2 is only 0.1843%. Margin MAD is 0.001245; margin crossings are 0.1108% at 0.3 and 0.0947% at 0.4. M001/M002 acceptance decisions change for 0.0912%/0.0893% of aux rows.
- Versus final in-sample scoring, owner churn is 0.1873%, margin MAD 0.001424, and M001/M002 acceptance changes are 0.1237%/0.1280%.

Interpretation: the top-k lists are sensitive around their cutoffs, as expected when replacing fold assignments, but this does not propagate into major owner or decoder-boundary churn. The detailed country/label quantiles, threshold crossings and negative-tail measurements are retained in the CSV artifacts.

## Artifacts

All new outputs are under `experiments/m003_k8/`:

- `artifacts/oof3_probabilities.parquet` — 76 MB
- three fold model files — about 21 MB each
- `artifacts/step1_old_p_A0_A1a.parquet` — 40 MB
- `artifacts/step1_aggregate.csv`, `step1_paired.csv`
- `artifacts/selfcheck_distributions.csv`, `selfcheck_paired.csv`
- `artifacts/selfcheck_threshold_crossings.csv`, `selfcheck_topk_churn.csv`
- `artifacts/selfcheck_owner_margin.csv`, `selfcheck_flags.json`
- executable scripts: `step1_compare_old_p.py`, `train_oof3.py`, `self_check.py`

No M002-K5 artifact was modified.
