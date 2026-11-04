# M010 error analysis and next-step experiments

## Executive summary

M010 is already close to the useful ceiling of its present candidate graph, but it is recall-limited: on holdoutB it has 0.99778 precision and 0.97087 recall. A global threshold reduction is not the best next move because F0.5 heavily rewards precision and the decoder threshold was selected on A1b.

The completed M011 experiment adds a genuinely independent K7 cross-encoder to the M010 fusion model. It improves both selection and holdout:

| Model | A1b F0.5 | HoldoutB F0.5 | Precision | Recall | Singleton accuracy | Empty-address found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M010 | 0.98918 | 0.98925 | 0.99778 | 0.97087 | 0.99196 | 52.93% |
| M011, M010 + K7 CE | **0.98951** | **0.98950** | **0.99825** | 0.97030 | **0.99415** | 52.07% |

The +0.00025 holdout gain is consistent with the +0.00033 A1b gain. It comes from precision and singleton rejection, not from recovering underpredicted matches. K7-derived features contribute about 82% of XGBoost gain importance, so this is a real complementary ensemble rather than a cosmetic average.

Recommended order:

1. Materialize an M011 test submission as the low-risk ensemble branch, but retain M010 because M011 trades about 0.00057 recall for precision and France has no labels.
2. Test a small lexical safety net before training another large model. Existing artifacts recover 2,529 of the 4,989 current blocker misses.
3. Build an empty-address specialist only for unique-name cases. Do not force matches for exact-name twins: most of those records are non-identifiable from the supplied fields.
4. Treat France as a scoring/competition domain-shift problem, not simply a shortage of candidates.

## Remaining error composition

M011 has the same candidate graph and decoder form as M010 and differs only by the K7 score features, so its persisted scores provide the cleanest detailed view of the remaining M010-family errors.

HoldoutB totals:

- 763,014 true edges
- 741,652 predicted edges
- 740,354 true positives
- 22,660 false negatives
- 1,298 false positives
- macro F0.5 0.989499

| Failure stage | Count | Share of FN | Oracle gain if every case were fixed |
| --- | ---: | ---: | ---: |
| Wrong S1 owner scores highest | 8,342 | 36.8% | +0.00354 |
| Winning true pair scores below tau | 7,501 | 33.1% | +0.00311 |
| True pair absent from blocker graph | 4,989 | 22.0% | +0.00193 |
| True pair outside CE shortlist | 1,775 | 7.8% | +0.00060 |
| Winner margin below delta | 53 | 0.2% | +0.00004 |

The margin threshold is no longer a bottleneck. Almost all remaining loss is owner ranking, absolute confidence, or candidate generation.

False positives are already scarce: 816 attach an actually unmatched auxiliary record and 482 assign an auxiliary record to the wrong S1. Fixing every FP is worth only +0.00153, versus +0.00898 for every FN. This confirms that the family is underpredicting locally, but it does not justify lowering the threshold globally because the attainable FN fixes have very different risk.

## Empty-address ceiling

Empty-address records account for 16,240 of 22,660 misses (71.7%). The cross-tab explains why:

| Empty-address owner-name status | True edges | Missed | Found |
| --- | ---: | ---: | ---: |
| Unique exact-normalized S1 name | 22,115 | 4,528 | 79.5% |
| Name shared by 2-3 S1 owners | 4,907 | 4,852 | 1.1% |
| Name shared by 4+ S1 owners | 6,864 | 6,860 | 0.06% |

Thus 11,712 empty-address misses are exact-name twins. With no address and the same normalized business name, a forced assignment mostly creates owner swaps. This is why M011's stronger rejection improves F0.5 while slightly reducing empty-address recall.

The recoverable target is the 4,528 unique-name empty-address misses. Their stage split includes 1,930 below-threshold cases, 1,350 wrong-owner cases, 890 blocker misses, and 354 outside the CE shortlist. A specialist should therefore use name uniqueness/fanout as a hard context feature and be trained with same-name competing owners. It should not share one threshold with ambiguous twins.

## Candidate-generation opportunity

The 4,989 blocker misses split into 2,742 empty-address and 2,247 address-present cases. Reusing existing lexical artifacts gives the following truth-recovery diagnostic:

- Number-aware address retrieval, forward top-3 plus reverse top-1: 1,623 recovered, all with an address.
- Normalized-name reverse top-3: 1,004 recovered, including 876 empty-address cases.
- Their union: **2,529 / 4,989 recovered (50.7%)**, comprising 876 empty and 1,653 address-present misses.

This is candidate recall only, not an achieved F0.5 gain. The full oracle value is at most roughly half of the blocker-miss +0.00193 headroom, and the matcher must still reject added negatives. Because the streams were selected after viewing holdoutB errors, the union must be evaluated on A1b or a new untouched split before promotion.

A practical M012 candidate branch is:

```text
dense M010 graph
UNION number-boosted address forward top-3/source
UNION number-boosted address reverse top-1
UNION normalized-name reverse top-3 only for empty-address aux
-> deduplicate
-> cap/rerank lexical-only additions
-> score additions with both CEs
-> retrain fusion on A1a, select on A1b
```

This targets measured misses and avoids expanding name-only candidates for every non-empty record.

## France and the public-test gap

The reported public score is 0.984 versus 0.98925 local. France is 15% of test S1 and is absent from labelled train, so it remains the main plausible source of the gap, although the hidden labels prevent proof.

M010 label-free test diagnostics:

| Country | Candidates/S1 | CE shortlist/S1 | Predictions/S1 | Aux assigned | Close CE twins |
| --- | ---: | ---: | ---: | ---: | ---: |
| France | 20.95 | 6.19 | 3.268 | 59.09% | 3.96% |
| India | 20.58 | 6.00 | 3.376 | 57.96% | 1.34% |
| US | 20.32 | 6.10 | 3.382 | 58.75% | 0.95% |

France has at least as many candidates and shortlisted pairs, but about 3.2% fewer predictions per S1 and substantially more close owner twins. Its empty-address assignment rate is 46.7%, versus 54.8% for India and 58.8% for the US. This points to score calibration and owner ambiguity rather than insufficient candidate volume.

The overall 58-59% auxiliary assignment rate should not by itself be compared with the 74% matched-aux prevalence in train: test has a different auxiliary/S1 density. Predictions per S1 are the safer label-free comparison.

Earlier dense-proxy checks do not show a total France retrieval collapse: median top-1 score and top-1/top-2 margin are comparable with US/India, and 98.2% of sampled France dense top-1 pairs are mutual best. The next France experiment should therefore add country-agnostic robustness transformations to the CE/fusion training—French-style street abbreviation expansion, diacritic folding, house-number dropping, and locality/department perturbation—and promote them only if they improve US/India A1b without damaging holdoutB. A France-only threshold chosen from unlabeled test distributions would be much riskier.

## What to run next

### 1. M011 test ensemble

Score the existing 10.5M M010 test shortlist with the K7 model and apply the saved `fusion_1.json` with tau 0.75 and delta 0.05. This is the only completed model change with consistent A1b and holdout gains. Keep M010 as a fallback because the ensemble improves precision while slightly reducing recall, and its behavior on France is unlabelled.

### 2. M012 lexical safety net

Run the targeted union above, score only newly introduced pairs, and retrain the fusion. This has more plausible recall upside than widening the CE shortlist again: shortlist misses are only 1,775, while the proposed streams recover 2,529 currently absent true pairs.

### 3. Unique-name empty-address gate

Train or calibrate a separate decision head for `empty_addr && normalized_name_fanout == 1`. Use the name-only CE logit, name margin, rarity, reverse-name rank/score, and both general CEs. Do not relax ambiguous-name records. Select the specialist threshold on A1b; do not tune it on holdoutB.

### 4. Diverse ensemble rather than more of the same

If a third model is trained, make its errors structurally different: name-only/character model, augmented address model, or lexical reciprocal-rank model. Another seed of the same CE is less attractive than the proven raw-address K7 plus `[MISSING]`-aware M004 pairing.

## Produced artifacts

- `experiments/m011_ensemble/artifacts/results_m011.csv`
- `experiments/m011_ensemble/artifacts/error_summary_m011.csv`
- `experiments/m011_ensemble/artifacts/error_stages_m011.csv`
- `experiments/m011_ensemble/artifacts/error_slices_m011.csv`
- `experiments/m011_ensemble/artifacts/error_transition_m006_m011.csv`
- `experiments/m011_ensemble/artifacts/blocker_miss_lexical_recovery.csv`
- `experiments/m011_ensemble/artifacts/blocker_miss_name_recovery.csv`
- `experiments/m011_ensemble/artifacts/blocker_miss_combined_recovery.csv`

Reproducible diagnostics are in `experiments/m011_ensemble/analyze_errors.py`, `post_analysis.py`, and `blocker_miss_recovery.py`.
