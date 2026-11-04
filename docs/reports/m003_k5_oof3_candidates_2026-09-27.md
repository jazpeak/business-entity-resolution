# M003 K5-on-OOF3 candidate checkpoint — 2026-09-27

## Result

K5 candidate membership was generated from the unchanged M001 candidate graph. No blocking or FAISS work was run. HoldoutA uses its fold-specific OOF3 probability; holdoutB uses the mean probability from the three frozen OOF3 fold models, all of which exclude holdoutB from fitting.

| Partition | M003 pairs | M002 actual | pair delta | M003 positives | M002 actual | positive delta | relative delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 367,075 | 367,075 | 0 | 239,553 | 239,529 | +24 | +0.0100% |
| A1a | 367,995 | 367,995 | 0 | 239,412 | 239,357 | +55 | +0.0230% |
| A1b | 368,815 | 368,815 | 0 | 240,967 | 240,974 | -7 | -0.0029% |
| holdoutB | 1,101,595 | 1,101,595 | 0 | 718,091 | 718,124 | -33 | -0.0046% |

All pair-count assertions passed. Every S1 has exactly five selected rows and `(s1, a)` membership is unique.

## Country check

| Partition | Country | pairs | M003 positives | M002 positives | delta |
|---|---|---:|---:|---:|---:|
| A0 | India | 146,160 | 95,453 | 95,456 | -3 |
| A0 | US | 220,915 | 144,100 | 144,073 | +27 |
| A1a | India | 147,990 | 96,086 | 96,081 | +5 |
| A1a | US | 220,005 | 143,326 | 143,276 | +50 |
| A1b | India | 147,850 | 96,582 | 96,559 | +23 |
| A1b | US | 220,965 | 144,385 | 144,415 | -30 |
| holdoutB | India | 437,340 | 285,716 | 285,723 | -7 |
| holdoutB | US | 664,255 | 432,375 | 432,401 | -26 |

The signs vary by fold/country, and every positive-count change is at most 0.023% of the prior partition total. This is far smaller than the previously observed top-5 membership churn and shows no systematic country bias. No problem is flagged.

Exact selected-row replacement rates relative to M002 are 4.95–5.41% within the holdoutA top-five sets (equivalent to the earlier 2.43–2.68% changed-row rate when measured over the full candidate pool) and 3.21–3.42% on holdoutB.

## Artifacts and runtime

- `experiments/m003_k8/artifacts/k5_oof3_candidates.parquet`
- `experiments/m003_k8/artifacts/k5_oof3_summary.csv`
- `experiments/m003_k8/artifacts/k5_oof3_country_summary.csv`
- `experiments/m003_k8/generate_k5_oof3.py`
- observed generation runtime: 29.4 seconds

No M001 or M002 artifact was written. Text/feature preparation and mDeBERTa training have not started. Future runtime comparisons will use M002-K5's actual logged timings.
