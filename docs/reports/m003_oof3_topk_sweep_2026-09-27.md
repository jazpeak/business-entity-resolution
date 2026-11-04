# M003: Leakage-Safe OOF3 Scoring and Cross-Encoder Top-K Sweep

September 27, 2026

## 1. Executive summary

M003 tests two changes to the M002 entity-resolution pipeline while preserving its candidate graph and model architecture:

1. replace the existing two-fold out-of-fold LightGBM probabilities with stricter three-fold, S1-disjoint probabilities (`p_new_oof3`); and
2. expand the mDeBERTa cross-encoder shortlist from five candidates per S1 to six and seven.

The investigation began with a correction to the experimental premise. The probability already stored by M001 was **not** an in-sample final-model score: it was already a two-fold out-of-fold estimate. Applying the saved final M001 model to the same rows produced materially different, substantially more optimistic predictions. M003 therefore compares OOF3 with an already leakage-safe OOF2 baseline, not with an in-sample baseline.

OOF3 passed its calibration, country-divergence, negative-tail, and full-graph ownership checks. A K5-on-OOF3 control then isolated the effect of fold construction from candidate count: its holdoutB macro F0.5 was 0.985566, effectively flat to M002's 0.985680. The improvement comes from widening the cross-encoder shortlist, not from OOF3 alone.

At a common decoder operating point (`tau=0.45`, `delta=0.40`), K7 reaches **0.987716 macro F0.5**, with 0.996820 precision and 0.967817 recall. This is +0.002149 over the K5-on-OOF3 control and only 0.000034 below K7's A1b-selected threshold result. The gain therefore survives a shared, non-cherry-picked threshold.

K7 changes 1.0739% of accepted auxiliary ownership decisions relative to K5, narrowly above the original 1% diagnostic limit. Directional decomposition shows that this is predominantly useful change: 49.05% of churned records move from incorrect or unassigned to correct, versus 12.33% moving from correct to incorrect or unassigned—a 3.98:1 beneficial-to-harmful ratio and +2,935 net corrections.

**Recommendation:** advance **K7-on-OOF3 with `tau=0.45`, `delta=0.40`** as the M003 configuration. Treat the 1% churn gate as a conservative diagnostic alert, not a rejection, because the fixed-threshold gain is large relative to the observed OOF noise and the churn is strongly truth-improving. Retain the K5 and K6 results as controls and preserve the full churn audit for review.

## 2. Baseline and experimental question

M002 reranks the five strongest candidates per S1 from the existing M001 candidate graph with an mDeBERTa-v3 cross-encoder. It then learns a six-feature LightGBM fusion on A1a and selects a one-owner decoder on A1b. M003 leaves that architecture unchanged and asks:

- Does a stricter three-fold OOF estimate improve or destabilize the base LightGBM probabilities?
- After controlling for that probability change, does cross-encoding more than five candidates recover useful matches?
- Are any gains robust to a common decoder threshold rather than an unusually permissive A1b selection?
- Does additional owner churn represent error correction or destabilization?

The dense candidate graph was reused unchanged. No reblocking was performed.

## 3. Correction: M001 probabilities were already OOF

The initial concern was that the probability in `scored_pairs.parquet` might be the saved final M001 model evaluated in-sample. A direct paired comparison disproved that hypothesis.

The saved final `lgbm_m001.txt` was applied to A0 and A1a and compared row-for-row with the stored probability. On A0, the mean absolute difference was 0.004591, Spearman correlation was 0.989604, 6.484% of rows differed by more than 0.01, and 1.210% differed by more than 0.10. A1a was similar: mean absolute difference 0.004571, Spearman 0.989550, 6.488% above 0.01, and 1.199% above 0.10.

The aggregate quality difference was also too large for the two scores to be the same estimator. On A0, the stored OOF2 score had log loss 0.031504, ROC-AUC 0.998549, and average precision 0.994415; final-model in-sample scoring had log loss 0.020786, ROC-AUC 0.999568, and average precision 0.998038.

This established the correct interpretation:

- the stored M001 probability is an existing two-fold out-of-fold score;
- the saved final-model score is an optimistic in-sample comparator; and
- OOF3 is a fold-construction refinement, not the first leakage correction.

## 4. Three-fold OOF3 construction

HoldoutA S1 records were assigned deterministically to A0, A1a, or A1b. Three frozen-configuration LightGBM models were trained, each excluding the complete S1 partition it scored. Train and validation S1 sets were asserted disjoint.

| Scored partition | Validation rows | Validation S1 | Training rows | Training positives |
|---|---:|---:|---:|---:|
| A0 | 1,488,007 | 73,415 | 2,986,595 | 252,838 |
| A1a | 1,491,736 | 73,599 | 2,982,866 | 252,466 |
| A1b | 1,494,859 | 73,763 | 2,979,743 | 254,446 |

The models use the frozen M001 feature set and 766 trees. Together they produced `p_new_oof3` for all 4,474,602 holdoutA candidate rows. HoldoutB, unseen by all three fits, is scored by averaging the three model probabilities.

### 4.1 Fold-level quality

| Partition | Version | Mean p | Log loss | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|
| A0 | OOF3 | 0.169404 | 0.030887 | 0.998605 | 0.994620 |
| A0 | OOF2 | 0.169342 | 0.031504 | 0.998549 | 0.994415 |
| A1a | OOF3 | 0.169040 | 0.031011 | 0.998602 | 0.994542 |
| A1a | OOF2 | 0.168859 | 0.031555 | 0.998546 | 0.994363 |
| A1b | OOF3 | 0.169590 | 0.030984 | 0.998579 | 0.994576 |
| A1b | OOF2 | 0.169578 | 0.031561 | 0.998524 | 0.994389 |

OOF3 improves log loss, ROC-AUC, and average precision slightly in every fold without a material mean shift.

### 4.2 Paired-difference and calibration audit

Across all rows, OOF3 versus OOF2 has signed mean difference 0.000085, mean absolute difference 0.003767, p95 absolute difference 0.015821, p99 0.090396, Pearson correlation 0.998985, and Spearman correlation 0.988158. The fractions with absolute changes above 0.01, 0.05, and 0.10 are 6.640%, 2.022%, and 0.860%.

The mandatory OOF3 stop conditions all passed:

| Check | Observed | Stop limit | Result |
|---|---:|---:|---|
| Maximum fold mean shift | 0.000181 | 0.010 | Pass |
| Worst log-loss change | -0.000545 | +0.005 | Pass; improved |
| Maximum US/India delta difference | 0.000154 | 0.005 | Pass |
| Maximum negative q99 increase | 0.001584 | 0.030 | Pass |
| Full-graph owner churn vs OOF2 | 0.1843% | 1.000% | Pass |

OOF3 does move rows near shortlist boundaries. Versus OOF2, top-5 row-membership churn is 2.43–2.68% and top-8 row-membership churn is 4.51–6.05%; 23.7–24.9% of S1 top-5 sets and 41.8–50.8% of top-8 sets change at least one member. These local changes do not propagate into large full-graph ownership movement: owner churn remains 0.1843%, margin mean absolute difference is 0.001245, and M001/M002 acceptance changes are 0.0912%/0.0893%.

## 5. K5-on-OOF3 control

Before changing K, the complete M002 pipeline was rerun with top five candidates selected by OOF3. This control uses the same mDeBERTa configuration, A1a fusion, A1b threshold grid, and holdoutB ownership competition as the K sweep.

The K5 candidate counts exactly matched the M002 K5 counts: 367,075 A0 rows, 367,995 A1a rows, 368,815 A1b rows, and 1,101,595 holdoutB rows. Every S1 had five unique candidates, text joins were complete, A1 inference produced 736,810 non-null scores, and holdoutB inference produced 1,101,595 scores. The selected decoder was `tau=0.45`, `delta=0.40`.

| System | Macro F0.5 | Precision | Recall | US F0.5 | India F0.5 | Singleton accuracy |
|---|---:|---:|---:|---:|---:|---:|
| M002 K5 + OOF2 | 0.985680 | 0.994686 | 0.962690 | 0.985398 | 0.986108 | 0.991063 |
| M003 K5 + OOF3 | 0.985566 | 0.993935 | 0.963971 | 0.985362 | 0.985877 | 0.988058 |

OOF3 alone changes the precision/recall balance slightly but does not improve macro F0.5: the control is 0.000114 below M002. All subsequent K comparisons therefore use 0.985566 as the causal control for shortlist width.

## 6. K6/K7 shortlist sweep

K6 and K7 reuse `p_new_oof3` and the existing graph. For each K, the pipeline is:

```text
OOF3 top-K candidates
  -> raw-text preparation
  -> mDeBERTa A0 training
  -> A1a fusion training
  -> A1b tau/delta tuning
  -> frozen holdoutB candidate and competitor inference
  -> one-owner decoding
```

The cross-encoder configuration remains the M002 configuration: one epoch, batch 32, maximum sequence length 128, AdamW at `2e-5`, bf16 autocast, and seed 17. Fusion retains the same six features and 500-tree LightGBM configuration.

### 6.1 Candidate and preparation gates

Candidate counts were exactly proportional to K5 in every partition; there were no candidate-cap shortfalls or decreases.

| K | A0 | A1a | A1b | holdoutB | Ratio vs K5 |
|---|---:|---:|---:|---:|---:|
| 5 | 367,075 | 367,995 | 368,815 | 1,101,595 | 1.0x |
| 6 | 440,490 | 441,594 | 442,578 | 1,321,914 | 1.2x |
| 7 | 513,905 | 515,193 | 516,341 | 1,542,233 | 1.4x |

K6 text preparation produced 2,646,576 candidate rows and 2,265,798 holdoutB dev-competitor rows. K7 produced 3,087,672 candidate rows and 3,593,220 competitor rows. All checked joins had no missing text, null scores, or duplicate `(s1, a)` pairs.

### 6.2 A1b selections and holdoutB results

K6 selected `tau=0.45`, `delta=0.40`, with A1b macro F0.5 0.986779. Its holdoutB macro F0.5 is 0.986879, a +0.001313 gain over K5 OOF3; this clearly exceeds the 0.0001–0.0002 OOF-only noise band and promoted K7.

K7's deterministic A1b sort selected `tau=0.30`, `delta=0.40` from a tie spanning tau 0.30–0.40. A1b macro F0.5 was 0.987234. HoldoutB at the selected threshold scored 0.987749.

| System | Macro F0.5 | Precision | Recall | US F0.5 | India F0.5 | Singleton accuracy | Tau / delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| K5 OOF3 control | 0.985566 | 0.993935 | 0.963971 | 0.985362 | 0.985877 | 0.988058 | 0.45 / 0.40 |
| K6 OOF3 | 0.986879 | 0.995752 | 0.966453 | 0.986589 | 0.987320 | 0.988789 | 0.45 / 0.40 |
| K7 OOF3, A1b-selected | 0.987749 | 0.996544 | 0.968563 | 0.987188 | 0.988602 | 0.988951 | 0.30 / 0.40 |
| M002 | 0.985680 | 0.994686 | 0.962690 | 0.985398 | 0.986108 | 0.991063 | 0.50 / 0.40 |

## 7. Fixed-threshold K7 diagnostic

Because K7 selected a lower tau than K5 and K6, the already-computed K7 holdoutB fused scores were decoded again at the common operating point `tau=0.45`, `delta=0.40`. No candidate generation, model fitting, or inference was repeated.

| System | Macro F0.5 | Precision | Recall | Singleton accuracy | Owner churn vs K5 |
|---|---:|---:|---:|---:|---:|
| K5 OOF3 control | 0.985566 | 0.993935 | 0.963971 | 0.988058 | — |
| K7 OOF3, fixed threshold | **0.987716** | **0.996820** | **0.967817** | **0.989845** | 1.0739% |

The fixed-threshold K7 result is +0.002149 over K5 and only -0.000034 below the A1b-selected K7 result. The improvement is therefore not an artifact of the lower selected tau. It includes simultaneous gains in precision, recall, and singleton accuracy.

The original K7 owner-churn check at tau 0.30 was 1.1134%, slightly above the 1% diagnostic limit. Fixing tau at 0.45 reduces churn to 1.0739% but does not move it below 1%. This motivated a truth-direction audit rather than treating all churn as equally harmful.

## 8. Owner-churn decomposition

At the fixed operating point, K5 accepted 740,011 auxiliary ownership assignments and K7 accepted 740,814. Their union contains 744,409 auxiliary records, of which 7,994 have a changed owner or assignment state.

| K5 state -> K7 state | Count | Share of churn |
|---|---:|---:|
| Incorrect or unassigned -> correct | 3,921 | 49.05% |
| Correct -> incorrect or unassigned | 986 | 12.33% |
| Neither assignment correct | 3,087 | 38.62% |

The detailed transitions are:

- unassigned -> correct: 3,920;
- incorrect owner -> correct: 1;
- correct -> unassigned: 986;
- incorrect owner -> unassigned: 2,609; and
- unassigned -> incorrect owner: 478.

Beneficial churn outnumbers harmful churn by **3.98:1**, yielding **+2,935 net corrected assignments**. The dominant improvement is recovery of true matches K5 left unassigned. The dominant harm is increased abstention on records K5 assigned correctly, not reassignment to a competing wrong owner.

### 8.1 Country split of harmful transitions

| Transition | India | US | Total |
|---|---:|---:|---:|
| Correct -> unassigned | 428 (43.41%) | 558 (56.59%) | 986 |
| Unassigned -> incorrect owner | 196 (41.00%) | 282 (59.00%) | 478 |

Neither harmful transition is isolated to one country.

## 9. Relationship to the M006 error taxonomy

The existing M006 FN/FP parquets and the published M006 A1–A5/E1–E2 taxonomy were joined to the two harmful K7 transitions where identifiers were compatible.

For the 986 correct-to-unassigned cases, 338 (34.28%) are exact true-pair false negatives in M006. Of those overlaps, 172 fall in A1–A5 and 166 in E1/E2. The overlap is 225/428 (52.57%) in India and 203/558 (36.38%) in the US. This shows that a substantial portion of K7's lost K5 assignments belongs to the same hard-match population identified independently in M006.

For the 478 unassigned-to-incorrect-owner cases, only nine (1.88%) have an M006 FN for the same auxiliary record: two A1, four E1, and three E2. Eighty-four (17.57%) reproduce an exact M006 false-positive pair; these belong to M006's separate FP taxonomy rather than A1–A5/E1–E2.

The M006 FN categories mean:

- **E1:** empty address with the normalized name shared by multiple S1 records; considered intrinsically ambiguous and generally best handled by abstention;
- **E2:** empty address with a unique normalized name;
- **A1:** alias or unrelated-looking name;
- **A2:** conflicting house number after zero-padding normalization;
- **A3:** house number missing on one side;
- **A4:** heavily changed name or address; and
- **A5:** near-identical name/address where model scoring or ownership competition still fails.

### 9.1 What K7 does fix

The strongest supported conclusion is that widening from K5 to K7 recovers true matches that were already present in the unchanged candidate graph but fell outside the cross-encoder's top-five scope. The 3,920 unassigned-to-correct transitions, higher recall, higher precision, and higher singleton accuracy all support this interpretation. The result is consistent across US and India at holdoutB aggregate level.

### 9.2 What K7 does not establish or solve

K7 does not change the blocker and therefore cannot recover true pairs absent from the existing graph. It also does not resolve intrinsically ambiguous E1 records; the M006 analysis indicates abstention is appropriate when an empty-address name is shared by multiple S1 records. The current diagnostics do not show category-specific K7 recall gains for every A1–A5/E2 group, so no claim is made that K7 systematically fixes those categories.

K7 also introduces 986 lost correct assignments and 478 newly wrong assignments. Although these are outweighed by corrections, they remain real regressions. The M006 overlap suggests that many losses are hard or ambiguous records, but this does not eliminate their cost.

## 10. Final recommendation

Adopt the following as the M003 candidate configuration:

```text
Base probability:  p_new_oof3
Candidate graph:   existing M001 graph; no reblocking
Cross-encoder:     top 7 per S1
Training:          M002 mDeBERTa configuration, A0 only
Fusion:            M002 six-feature LightGBM, A1a only
Decoder:           tau = 0.45, delta = 0.40
HoldoutB macro F0.5: 0.987716
```

Reasons:

1. OOF3 is leakage-safe and passes all probability-distribution and full-graph stability checks.
2. The K5 control shows OOF3 alone is flat, so the K7 gain is attributable to shortlist width rather than fold noise.
3. K6 and K7 improve monotonically over K5 on holdoutB.
4. K7's gain survives the shared K5/K6 operating point and is almost identical to its A1b-selected result.
5. Owner churn is slightly above the conservative 1% alert, but truth decomposition shows a 3.98:1 beneficial-to-harmful ratio and +2,935 net corrections.

The recommended score is a validation result, not authorization to retune on holdoutB. The fixed threshold is chosen as a robustness and comparability decision, not by optimizing holdoutB.

## 11. Open next steps

1. **Freeze and package the K7 fixed-threshold configuration.** Record hashes for the OOF3 models, K7 cross-encoder, fusion model, decoder, and inference code before any test access.
2. **Audit the 986 lost correct assignments.** Separate score-threshold failures from margin-competition failures and quantify E1/E2 versus A1–A5 at full resolution.
3. **Audit the 478 new wrong assignments with the M006 FP taxonomy.** The exact M006 FP overlap is 84 records; determine whether a targeted abstention or competition feature can remove these without sacrificing the 3,920 recovered matches.
4. **Measure category-specific K7 deltas.** Reuse the M006 reporting framework to compare K5 and K7 FN/FP composition directly, especially E2 and A1–A5. This is diagnostic work and should not change the frozen decoder without a new development split.
5. **Keep E1 abstention explicit.** Wider K does not create identifying evidence for genuinely ambiguous empty-address/name-twin records.
6. **Consider wider K only as a new experiment.** K7 already raises competitor inference substantially; any K8 study should require a development-side promotion criterion and the same fixed-threshold/churn audit.

## 12. Reproducibility and artifact isolation

All M003 outputs are namespaced under:

- `experiments/m003_k8/artifacts/` for OOF3 and K5-control artifacts;
- `experiments/m003_k8/artifacts/k6_oof3/`; and
- `experiments/m003_k8/artifacts/k7_oof3/`.

The fixed-threshold and churn analyses were read-only diagnostics over saved probabilities, cross-encoder scores, fusion models, truth edges, and M006 error parquets. They did not generate candidates, retrain a model, or overwrite an artifact.

**No M001 or M002 artifact was modified during M003.** M001/M002 files were read only as frozen inputs and controls.
