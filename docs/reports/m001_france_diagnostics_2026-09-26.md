# M001 France Diagnostics

## Conclusion

France is **not under-scored or over-rejected** by the M001 matcher. Its owner scores and acceptance rate are higher than US/India, while final predictions per S1 and empty rates remain aligned. There is no label-free evidence for lowering France thresholds or margins.

The clearest France-specific shift is weaker address token-set agreement. This supports trying the France-safe street/house-number features in lead #3, rather than country-specific calibration.

## Test comparison

| Metric | France | India | US |
| --- | ---: | ---: | ---: |
| S1 | 259,452 | 809,986 | 663,106 |
| Candidates / S1 | 20.95 | 20.58 | 20.32 |
| Owner score mean | **0.6345** | 0.5960 | 0.5996 |
| Owner margin mean | **0.6260** | 0.5922 | 0.5962 |
| Aux accepted | **61.24%** | 58.06% | 58.11% |
| Rejected: score below 0.6 | 38.49% | 41.78% | 41.73% |
| Rejected: margin below 0.3 | 0.27% | 0.17% | 0.16% |
| Predictions / S1 | 3.357 | 3.368 | 3.337 |
| Predicted empty | 5.29% | 5.53% | 5.71% |
| Dense retrieval top-1 is model owner | 93.67% | 94.98% | 96.11% |

The margin rule rejects very few records in every country. France's slightly higher margin-rejection rate is only about 0.11 percentage points above US and cannot explain a large leaderboard gap.

## Feature shifts on model-selected owners

| Feature | Quantile | France | India | US |
| --- | ---: | ---: | ---: | ---: |
| Address token-set similarity | p50 | **83.33** | 91.26 | 89.86 |
| Name token-sort similarity | p50 | 86.49 | 79.25 | 86.67 |
| Dense cosine | p50 | 0.835 | 0.835 | 0.861 |
| Number Jaccard | p50 | **1.00** | 0.50 | 0.33 |
| Matcher probability | p50 | **0.993** | 0.978 | 0.980 |

France address similarity is lower at the median despite much stronger numeric overlap: 59.10% of selected owners have equal first house numbers, versus 45.53% India and 44.92% US. This is consistent with the report's region/department and street-format observations. A street-only comparison plus explicit house-number features should separate useful local-address evidence from noisy trailing geography.

## Recommendation

1. Do not introduce France-specific thresholds from these diagnostics.
2. Implement lead #3 as country-agnostic features: street/address text with numeric tokens removed, number-aware address text, and house-number-conditioned street similarity.
3. Validate only through holdoutA tuning and one frozen holdoutB evaluation; inspect US/India separately for regressions.

Artifacts:

- `experiments/m001_pipeline/france_diagnostics.py`
- `experiments/m001_pipeline/artifacts/france_diagnostics/country_summary.csv`
- `experiments/m001_pipeline/artifacts/france_diagnostics/quantiles.csv`
