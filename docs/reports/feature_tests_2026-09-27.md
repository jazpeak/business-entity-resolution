# Matcher Feature Tests (M003–M006) — Results Report

Sep 27, 2026

## 1. Summary

The best configuration, **M006**, scores **0.98761 macro F0.5 on holdoutB**, up from 0.98568 for M002 (+0.0019) and 0.9715 for M001. It combines the missing-tag cross-encoder (M004) with the full engineered feature set (M005 V5) in the fusion model. Nearly all of the gain comes from **fewer false matches** (precision 0.9947 to 0.9963) and better singleton handling; the share of empty-address true matches found rose only from 43.7% to about 49%.

| System | Cross-encoder | Fusion features | holdoutB macro F0.5 | Precision | Recall | Singleton acc. | Empty-address matches found |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| M001 (public 0.961) | none | M001 LightGBM | 0.97152 | 0.9879 | 0.9449 | 0.947 | - |
| M002 (frozen) | mDeBERTa | 6 score features | 0.98568 | 0.99469 | 0.96269 | 0.991 | 43.7% |
| M003 | mDeBERTa | + missing flags, name rarity, look-alike margins | 0.98673 | 0.99581 | 0.96291 | 0.990 | 49.2% |
| M005 V5 | mDeBERTa | + house numbers, street/locality, IDF names, address/number competition | 0.98742 | 0.99620 | 0.96368 | 0.991 | 49.1% |
| **M006** | **mDeBERTa with `[MISSING]` tag** | **M005 V5 features** | **0.98761** | **0.99634** | **0.96375** | **0.992** | **49.2%** |

## 2. Protocol (identical for every test)

- Data split: frozen split v1. holdoutA is divided into A0 (cross-encoder training), A1a (fusion model training) and A1b (decoder tuning); holdoutB is only scored.
- Search and decoding conditions: full-density indexes, and every train S1 competes in one-owner decoding (dev S1 are scored only as competitors).
- Decoder: one-owner + margin (accept an aux record's best S1 if p >= tau and p - p_second >= delta); tau and delta chosen on A1b.
- Model selection: variants are compared on A1b; holdoutB is reported for information and never used to choose.
- Fusion learner: XGBoost on CUDA (GPU 0), depth 8, 600 rounds, eta 0.05. Re-training M002's 6 features with this learner reproduces M002 (0.98558 vs 0.98568), so differences come from features, not the learner.
- Empty-address metric: share of holdoutB true (S1, aux) edges whose aux record has an empty address and that are predicted.

## 3. Feature tests (M005, on M002 cross-encoder scores)

Each group is added on top of the M003 feature set (V0).

| Variant | Added features | A1b F0.5 | holdoutB F0.5 | Delta vs V0 (B) | Precision (B) | Empty found (B) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V0 | M003 set (baseline) | 0.98661 | 0.98673 | - | 0.99581 | 49.2% |
| V1 | House numbers: exact and full alphanumeric token (`1-D-206`, `14593d`), log/relative numeric difference, digit edit similarity, truncation, length match, missing pattern, unit/apt agree/conflict, shared long numbers, extra numbers per side, shared-number rarity | 0.98686 | 0.98688 | +0.00015 | 0.99602 | 48.9% |
| V2 | Street similarity (numbers and street types removed), locality overlap (share of the S1's non-numeric address parts present in the aux address) | 0.98662 | 0.98681 | +0.00008 | 0.99592 | 49.0% |
| V3 | IDF-weighted name overlap over S1 names of the country (term-frequency idea) | 0.98668 | 0.98682 | +0.00009 | 0.99584 | 49.3% |
| V4 | Address/number competition: candidates of the aux sharing the house number, address margin and IDF-name margin vs the best competing S1 | 0.98728 | 0.98726 | **+0.00053** | 0.99622 | 48.7% |
| **V5** | **All of the above** | **0.98749** | **0.98742** | **+0.00069** | 0.99620 | 49.1% |

Model-size and decoder checks on V5:

| V5 setting | A1b F0.5 | holdoutB F0.5 |
| --- | ---: | ---: |
| depth 8, 600 rounds, threshold decoder (chosen) | 0.98749 | 0.98742 |
| depth 6, 600 rounds | 0.98741 | 0.98740 |
| depth 10, 600 rounds | 0.98748 | 0.98746 |
| depth 8, 1,200 rounds | 0.98744 | 0.98749 |
| Expected-F0.5 decoding per S1 (gamma 1.5) | 0.98722 | 0.98728 |

- Competition features (V4) help most: they tell the model whether another S1 has the same house number or a better address match for this record.
- Model size does not matter (all within 0.0001).
- Expected-F0.5 decoding finds more matches (recall 0.9647, empty found 50.3%) but loses precision and singleton accuracy (0.985), so it scores lower than the tuned threshold.

Top V5 fusion features by gain: `p_x_ce` 47.8%, `ce_prob` 29.9%, `ce_logit` 15.9%, then `hn_missing` 0.73%, `addr_margin` 0.66%, `name_margin` 0.63%, `gap_aux` 0.60%, `s1_name_freq` 0.26%, locality overlap 0.21%, `ad_tset` 0.20%.

## 4. Missing-tag cross-encoder (M004)

When the aux address is empty, both records were given to the cross-encoder as `name: ... | address: [MISSING]` (the S1 address hidden too), so it compares names only. The cross-encoder was retrained on A0 with M002's exact settings; only the input changed.

| Fusion features | M002 cross-encoder | M004 (missing tag) cross-encoder |
| --- | ---: | ---: |
| M002's 6 score features | 0.98568 (empty found 43.7%) | 0.98564 (empty found 43.7%) |
| M003 features | 0.98673 (49.2%) | 0.98686 (49.7%) |
| M005 V5 features | 0.98742 (49.1%) | **0.98761** (49.2%) |

The tag does not by itself recover more empty-address matches: the cross-encoder was already judging these records mostly by name. It gives a small additional gain (+0.0002) once combined with the V5 features.

## 5. Empty-address records: status

- 33,886 holdoutB true edges (4.4%) have an empty aux address. M006 finds about 49%, versus 98.7% for records with an address.
- Diagnostics (see `experiments/m002_mdeberta/artifacts/empty_addr_report.txt`):
  - 60% of the misses have an owner whose exact normalized name is shared by 2 or more S1 in the country.
  - A sibling test on dev data shows that nothing else in the record separates these twins: the owner's other records are no more similar than the twin's (51-55% win rate; coin flip = 50%).
  - Guessing on those cases lowers the expected F0.5, so abstaining is correct.
- The recoverable part is unique-name records with near-exact names, about 5,400 edges. None of the tested features moved them, because the fusion model's decisions are dominated by the cross-encoder scores (about 94% of gain).

## 6. Issue found and fixed during the tests

In the `entity` environment (transformers 5.17), the mDeBERTa checkpoint loads as **float16** by default, and AdamW turns the weights into NaN after the first update (loss `nan` from step 1). Loading with `dtype=torch.float32` fixes it; the fixed script is `experiments/m004_missing_tag/train_score_fp32.py`. The `protobuf` package was also installed so the slow DeBERTa tokenizer loads. Any cross-encoder training run started with the original `train_score.py` in this environment (for example `m003_k8`) should be checked for `nan` losses.

## 7. Recommendation

| Option | holdoutB F0.5 | What is needed for a test submission |
| --- | ---: | --- |
| M006 (best) | 0.98761 | Score test with the M004 cross-encoder (top-5 candidates, about 11M pairs), compute V5 features on test, fusion + decoding |
| M005 V5 | 0.98742 | M002 cross-encoder test scores already exist; only V5 features on test + fusion + decoding (faster) |

- Submit **M005 V5** first if time is short (only features and fusion need to run on test), then **M006** once the M004 cross-encoder has scored test.
- Expect the public score to sit below holdoutB, as for M001 (0.9715 local, 0.961 public), mainly because of France (no labels) and the denser test set.

## 8. What to try next

1. Wider cross-encoder shortlist (top 8-10 instead of top 5): about 10,000 holdoutB true edges (1.3%) are never scored by the cross-encoder, worth up to +0.0023 if all were recovered.
2. A name-only cross-encoder trained on name pairs with hard same-name negatives, applied only to empty-address records, since the fusion model cannot move them without a stronger name signal.
3. Retrieval for empty-address records from a name-only index, so that unique-name owners always reach the shortlist (about 3,000 unique-name edges are lost before the cross-encoder).

## Artifacts

| Item | Path (server, `/home/arul24115/Entity/experiments/`) |
| --- | --- |
| M003 fusion v2 | `m003_fusion2/fusion2.py`, results `m003_fusion2/artifacts/final_holdoutB.csv` |
| M004 missing-tag cross-encoder | `m004_missing_tag/` (`make_tagged.py`, `train_score_fp32.py`, `run_m004.sh`, `fusion_tag.py`, model `artifacts/model_A0`) |
| M005 feature suite | `m005_feature_suite/feature_suite.py`, results `artifacts/results.csv`, importance `artifacts/importance_V5.csv` |
| M006 combined | `m006_combined/combined.py`, results `artifacts/results.csv` |
| Error analyses | `m002_mdeberta/analyze_holdoutB.py`, `analyze_empty_addr.py`, `sibling_signal_check.py` |
