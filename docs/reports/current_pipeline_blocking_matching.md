# Current Blocking and Matching Pipeline

This document summarizes the current business entity resolution pipeline in `final_submission_improved`, with emphasis on blocking, candidate generation, matching models, features, and final assignment.

## End-To-End Flow

The active pipeline is run from:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/run_all.sh`

It executes:

```text
normalize
-> key blocking
-> ANN / embedding retrieval
-> merge candidates
-> pair features
-> TF-IDF + siamese features
-> LightGBM stage 1
-> LightGBM stage 2
-> BERT on uncertain pairs
-> logistic combiner
-> thresholded one-to-one assignment
```

The current script enables:

```bash
ER_USE_ANN=1
ER_STAGE2=1
ER_BERT_DIR=bert_ce2
```

So the current pipeline uses embedding retrieval, the stage-2 cluster-support model, and the shipped BERT v2 cross-encoder.

## 1. Normalization

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/normalize.py`

For names, the pipeline performs:

- Mojibake repair
- ASCII transliteration using `Unidecode`
- Indic-script transliteration handling
- Domain extraction, for example `foo.com -> foo`
- Social handle detection, for example `@business`
- DBA / formerly / AKA alternative-name extraction
- Business abbreviation expansion, such as `pvt -> private`, `ltd -> limited`
- Legal-form stripping for core names
- Stopword removal
- Sorted core-name generation
- Concatenated-name generation
- Crude phonetic consonant skeleton generation for transliteration noise
- Non-Latin name flag

For addresses, the pipeline performs:

- Mojibake repair
- Country-aware state normalization for US and India
- Native Indian state-name mapping
- Address abbreviation expansion, such as `rd -> road`, `st -> street`
- Noise token removal
- House-number extraction
- Postcode extraction
- Address word extraction
- Address number extraction
- Empty-address flagging

These normalized fields feed both blocking and matching.

## 2. Key-Based Blocking

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/blocking.py`

Blocking is performed from each S2/S3 record to S1 records. The code assumes each S2/S3 record can match at most one S1.

Candidate generation:

- Build country-scoped blocking keys.
- Build an inverted index over S1 records.
- Skip overly common keys with `MAX_POSTING = 1000`.
- Score shared keys using IDF-like weights.
- Keep `PRE_K = 50` S1 candidates per S2/S3 by IDF score.
- Re-rank those candidates using RapidFuzz name/address similarity.
- Keep `TOPK_PER_RECORD = 8` at the raw blocking stage.

Blocking key families:

- `n`: name core tokens plus alternative-name tokens
- `b`: unordered pairs of first core-name tokens
- `f`: full sorted core name
- `k`: phonetic skeleton tokens
- `c`: first 8 characters of spaceless core name
- `a`: address word tokens
- `h`: house number + state
- `w`: house number + first address word
- `p`: postcode
- `x`: name token + state
- `y`: name token + house number

Blocking re-rank score:

```text
blk_rr = fuzzy_name_token_set + fuzzy_address_token_set + 2 * blk_score
```

For modeling, the pipeline normally keeps blocking top-4 with `TOPK_MODEL = 4`.

## 3. Embedding Retrieval / ANN Candidate Expansion

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/ann.py`

This is the second candidate source.

Model:

- Siamese retriever network named `siamese_R.pt`
- Similar architecture to the main siamese matcher
- Trained on train entities disjoint from dev and extra training entities
- Uses hard negatives from blocking candidates plus true pairs
- Embeds each record into a 128-dimensional representation
- Searches exact cosine similarity on GPU, country by country
- Adds top `K = 3` S1 records for every S2/S3 record

The merged candidate set is:

```text
blocking top-4 union retrieval top-3
```

For retrieval-only pairs, the pipeline computes fuzzy re-rank scores but sets blocking IDF evidence to zero.

Reported candidate recall:

```text
blocking top-4: 96.7% true-pair recall
blocking + retrieval top-3: 98.4% true-pair recall
```

## 4. Candidate Competition Features

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/features.py`

After merging, the pipeline recomputes competition features.

For each S2/S3 record:

- `rr_gap2`
- `score_gap2`
- `rr_rel2`
- `n_cand2`

For each S1 entity:

- `rr_gap1`
- `rr_rank1`
- `n_cand1`

These tell the model how clearly a pair wins against nearby alternatives.

## 5. Pairwise Matching Features

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/features.py`

Name features:

- RapidFuzz ratio
- RapidFuzz token sort ratio
- RapidFuzz token set ratio
- RapidFuzz partial ratio
- Jaro-Winkler on concatenated names
- Concatenated-name ratio
- Concatenated-name partial ratio
- Skeleton token-set similarity
- Alternative-name similarity
- Clean-name ratio
- Exact core-name match
- Token Jaccard
- Token counts
- Legal-form agreement
- Domain flag
- Handle flag
- Non-Latin flag
- S1/S2 name frequency

Address features:

- RapidFuzz ratio
- Token set ratio
- Token sort ratio
- Partial ratio
- Address-word token set similarity
- Address-word Jaccard
- House-number agreement
- Numeric-token Jaccard
- Number overlap count
- Postcode agreement
- State agreement
- Empty-address flags
- Address length ratio

Other features:

- Source id, S2 vs S3
- Blocking scores and ranks
- Competition features
- ANN cosine and ANN rank

## 6. TF-IDF Similarity Features

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/textsim.py`

Uses scikit-learn:

- `HashingVectorizer`
- `TfidfTransformer`
- sublinear TF
- hashed sparse vectors
- IDF learned from records in the current split

TF-IDF cosine features:

- `tf_nm_char`: name char 3-gram cosine
- `tf_nm_word`: name word cosine
- `tf_ad_char`: address char 3-gram cosine
- `tf_ad_word`: address word cosine
- `tf_all_char`: name + address char 3-gram cosine

Also adds gap features:

- `tf_nm_char_gap1`
- `tf_ad_char_gap1`
- `tf_all_char_gap1`

## 7. Siamese Matching Model

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/siamese.py`

This is separate from the retriever, though it uses a related architecture.

Architecture:

- Hashed character 3-grams plus word tokens
- Separate name and address `EmbeddingBag`s
- 64-dimensional name embedding
- 64-dimensional address embedding
- MLP projection to a 128-dimensional record embedding
- Match head over:
  - `z1 * z2`
  - `abs(z1 - z2)`
  - `name1 * name2`
  - `addr1 * addr2`

Training:

- Trained from scratch
- Binary cross-entropy loss on candidate pairs
- In-batch InfoNCE loss on true pairs
- Two cross-fitted models: `siamese_A.pt`, `siamese_B.pt`
- Extra training entities split into halves
- Dev gets averaged predictions

Outputs used as features:

- `siam_p`
- `siam_cos`
- `siam_cos_name`
- `siam_cos_addr`
- `siam_p_gap1`
- `siam_cos_gap1`

## 8. Stage-1 Matcher: LightGBM

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/final.py`

Model:

- LightGBM binary classifier
- Objective: binary
- `num_leaves = 255`
- `learning_rate = 0.05`
- `min_data_in_leaf = 200`
- `lambda_l2 = 1.0`
- `feature_fraction = 0.8`
- `bagging_fraction = 0.8`
- 3-fold `GroupKFold` by S1 entity
- Extra entities included in every training fold
- Final model saved as `models/lgb_final.txt`

From `models/meta_final.json`:

```text
train pairs: 9,239,155
dev pairs: 3,081,752
stage-1 OOF macro F0.5 without BERT: 0.98134
stage-1 threshold without BERT: 0.70
final stage-1 trees: 1357
```

## 9. Stage-2 Matcher: Cluster-Support LightGBM

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/stage2.py`

This model asks whether a candidate record fits the other records already confidently assigned to the same S1.

Cluster members are candidates whose first-stage probability is at least `0.5` and are the best S1 for that S2/S3 record.

Stage-2 features:

- `p1`
- `p1_rank1`
- `p1_gap2`
- `n_mem`
- `sum_p_mem`
- `mem_nm_max`
- `mem_nm_mean`
- `mem_ad_max`
- `mem_ad_mean`
- `mem_both_max`
- `mem_house_any`
- `mem_same_src`

Model:

- LightGBM binary classifier
- `num_leaves = 127`
- `learning_rate = 0.05`
- `min_data_in_leaf = 100`
- Saved as `models/lgb_stage2.txt`

Metadata:

```text
stage-2 OOF macro F0.5 without BERT: 0.98299
stage-2 threshold without BERT: 0.69
stage-2 trees: 221
```

## 10. BERT Cross-Encoder

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/crossenc.py`

Model:

- `google-bert/bert-base-multilingual-cased`
- 178M parameters
- Fine-tuned locally
- Current shipped model: `models/bert_ce2/`
- Stored in float16
- No runtime downloads required

Input format:

```text
[CLS] name | address | country [SEP] name | address | country [SEP]
```

BERT receives raw text, so it can use native scripts and raw name/address signals.

Scoring is only done on uncertain pairs:

```text
0.01 < p < 0.99
```

Metadata:

```text
dev uncertain pairs rescored by BERT: 77,370
```

## 11. Probability Combiner

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/final.py`

A scikit-learn logistic regression combines:

- LightGBM / stage-2 probability logit
- BERT probability logit
- Interaction term between the two logits

Saved combiner:

```text
coef: [0.7236, 0.3049, -0.00248]
intercept: -0.04239
```

Final validation:

```text
OOF macro F0.5 with BERT: 0.98624
final threshold: 0.68
```

## 12. Final Assignment / Matching

Implemented in:

`D:/Downloads/AmazonML/final_submission_improved/code/business_entity_resolution/src/er/postprocess.py`

Final rule:

- Sort all scored pairs by probability.
- For each S2/S3 record, keep only its best S1 candidate.
- Accept the assignment only if probability is at least `0.68`.
- Group accepted S2/S3 ids under each S1 id.
- S1 records without accepted matches get an empty list.

So final matching is one-to-one from each S2/S3 record to at most one S1, while each S1 can receive multiple S2/S3 matches.

## Outputs

Packaged outputs:

- `D:/Downloads/AmazonML/final_submission_improved/output/candidate_pairs.tsv`
- `D:/Downloads/AmazonML/final_submission_improved/output/matching_results.tsv`

`candidate_pairs.tsv` contains generated/scored candidate ids per S1.

`matching_results.tsv` contains only final accepted matches after assignment and thresholding.

## Short Summary

The current pipeline is a hybrid entity-resolution system:

```text
rule-heavy normalization
-> IDF/fuzzy blocking
-> neural embedding retrieval
-> rich string, TF-IDF, siamese, blocking, competition, and ANN features
-> LightGBM stage 1
-> cluster-support LightGBM stage 2
-> BERT cross-encoder on uncertain pairs
-> logistic calibration
-> thresholded final assignment
```
