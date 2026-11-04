# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** iiit_fishes  
**Team Members:** Jazl Anwar (team leader), Satvik Mittal, Arul Thiruvasagam A, Abhay Trehan  
**Submission Date:** 29 September 2026

---

## 1. Executive Summary

We use a scalable pipeline: dense blocking → LightGBM prefilter → four cross-encoders → XGBoost fusion → one-owner decoding.

- **Blocking:** a fine-tuned multilingual-e5 retriever, searched with FAISS, keeps 99.8% of the achievable F0.5 at about 20 candidates per record.
- **Matching:** four transformer cross-encoders read both records jointly. They are three mDeBERTa-v3 models (one of them street-aware) and one mmBERT model, trained on a bias-free pool of hard examples. An XGBoost fusion combines their scores with name, address and house-number features.
- **Decoding:** a one-owner rule gives each Source 2/3 record to at most one Source 1 entity.
- **Validation:** macro F0.5 = **0.99070** on a frozen, test-density holdout.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Data structure.**
  - Each S1 entity owns 0 to many S2/S3 records (mean 3.46); about 5.5% of S1 have no match.
  - All records of an entity are noisy copies of it, with the same noise patterns in all three countries (US, India, France).
- **Name noise.**
  - typos, character swaps, accent changes, casing
  - added or removed legal forms and suffixes (`LLC`, `Pvt Ltd`, `SAS`/`SARL`, `et Fils`, `& Co`)
  - domain-style names, native-script (Indic) names
  - completely replaced "alias" brand names at the same address
- **Address noise.**
  - reordered components, street-type abbreviations (`Rd`, `R.`, `AV`, `BD`), region/department swaps
  - added unit, PO box or floor numbers
  - **house numbers zero-padded, truncated, shifted by a few, or dropped**
  - 4.4% of aux records have an **empty address**
- **Look-alikes.** Many records belonging to no S1, or to another S1, are near-copies with one field changed (house number or street). These drive most false matches.
- **Competition and density matter.** Many S1 entities share names or addresses, so a record's correct owner must beat every other S1 that could claim it. Validation therefore searches the full-size index and lets every S1 compete for each aux record, exactly as on test.
- **Unresolvable cases.** When an aux record has no address and 2+ S1 share its exact name, nothing else in the records separates them: the owner's other records are no more similar than a twin's (51-55% pairwise win rate). Abstaining is F0.5-optimal there.

### 2.2 Solution Strategy

**Approach Type:** Blocking + stacked classifier ensemble (dense retrieval → gradient boosting → transformer cross-encoders → gradient-boosting fusion → constrained decoding)

**Core Innovation:**
1. **Test-density validation.** A frozen entity-level split where blocking searches the full index and decoding lets all 2.2M train S1 compete, so local scores reflect test conditions.
2. **Adaptive cross-encoder shortlist.** Top-5, plus ranks 6-10 only when the prefilter probability is ≥ 0.01. This covers ~99% of true pairs at ~6 pairs per S1.
3. **Bias-free hard-example training.** Cross-encoder training data comes only from S1 entities that never compete with evaluation entities. It includes 598k same-name / different-street-or-number non-matches and up-weighted name-suffix-variation matches, which directly target the look-alike and legal-form failure modes.
4. **One-owner + margin decoding** tuned for macro F0.5.

### 2.3 Scalability

Every stage is linear in the number of records, or in a fixed per-record budget. No stage compares all S1 with all S2/S3 records.

| Stage | Cost | Test size |
| --- | --- | --- |
| Dense blocking | FAISS IVF approximate search, ~1% of the index scanned per query; fixed top-k per source; reverse candidates capped at 10 per S1 | 35.6M candidate pairs (~20.5 per S1) |
| LightGBM prefilter | one prediction per candidate pair | 35.6M pairs, minutes on CPU |
| Cross-encoders | only the adaptive shortlist | 10.5M pairs (~6.1 per S1); name-only model on 1.6M empty-address pairs |
| Features + fusion | per-pair features; competition features are group-bys per aux record; name/number statistics are single passes over S1 | linear |
| One-owner decoding | group by aux record, best S1 + margin check | linear |

- **Hardware:** the full test inference runs on a single NVIDIA A100 in about 6-8 h.
- **Where the time goes:** the cross-encoders dominate. Their work splits trivially across GPUs by pair shards, and adding GPUs reduces wall time linearly.
- **Memory:** indexes are built per country and source, so memory stays bounded as data grows. Larger data can shard the index or use FAISS IVF-PQ.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - Dense embeddings of `name | address` from `intfloat/multilingual-e5-base`, fine-tuned as a bi-encoder on dev-split pairs.
  - Searched with **FAISS IVF-Flat** (nprobe 64, ~1% of the index scanned per query), separately per country and source.
  - **Forward** search: top-10 S2 and top-10 S3 per S1.
  - **Reverse** search: each S2/S3 record proposes its top-1 S1, capped at 10 per S1 to avoid hub entities.
- **Candidate pairs generated (test):** **35,579,090** pairs for 1,732,544 S1 (~20.5 per S1, versus ~9.97M S2/S3 records per full comparison). All are listed in `candidate_pairs.tsv`, because the prefilter model scores every one.
- **How we ensured true matches were not lost:**
  - Measured the *oracle* macro F0.5 (a perfect matcher restricted to the candidates) on a frozen holdout, searching the full-size train index: **0.9982** at 20.3 candidates per S1.
  - The same retrieval with exact search reaches 0.9988, so approximate search costs only 0.0006.
  - The reverse channel recovers records whose own nearest neighbours are dominated by look-alikes.
  - The multilingual encoder handles transliterated and native-script names, and reads house numbers inside the address text.

---

## 4. Matching Model

**Features used:**
- **Name features:**
  - rapidfuzz token-set ratio / ratio on raw, normalised, transliterated (Indic → Latin) and skeleton names
  - name frequency and rarest-token document frequency across S1
  - IDF-weighted token overlap
  - near-name competitor counts and name margins versus the aux record's other candidates
- **Address features:**
  - token-set similarity of full address and street part
  - locality overlap
  - address margin versus competing S1
  - empty-address and missing-field flags
- **House numbers:** a dedicated parser (zero padding stripped, unit/floor/PO box/ordinal/postal numbers excluded, order-free matching) giving:
  - equality, "same number present elsewhere", minimum numeric difference, truncation
  - shared / unshared number counts
  - number of candidates sharing the house number
- **Other:**
  - dense-retrieval score, ranks, reverse-search flags, score gap to the aux record's next candidate
  - **four cross-encoder probabilities**, each fine-tuned from its base model on `name: … | address: …` pairs:
    1. mDeBERTa-v3, with both addresses replaced by `[MISSING]` when the aux address is empty
    2. a name-only mDeBERTa-v3 for empty-address records, with same-name twins as hard negatives
    3. a street-aware mDeBERTa-v3 trained with 598k same-name / different-street-or-number non-matches
    4. **mmBERT-base trained on 2.9M clean pairs**, with name-suffix-variation matches up-weighted
  - products of the cross-encoder probabilities with the LightGBM prefilter probability

**Model type:**
- **Prefilter:** LightGBM on 31 features, which ranks all ~20 candidates.
- **Cross-encoders:** transformers (278M-307M parameters each; MIT licence) on the shortlist.
- **Fusion:** XGBoost (depth 8, 600 rounds) combining everything.
- **Decoding:** one-owner rule. Each S2/S3 record goes to its highest-scoring S1 when the fused probability is ≥ τ and exceeds the runner-up by ≥ δ.

**Threshold selection method:** F0.5 optimisation on a validation set. HoldoutA (entity-disjoint from training) is split into:
- **A0:** trains the first cross-encoder
- **A1a:** trains the fusion
- **A1b:** chooses τ and δ by grid search maximising macro F0.5, with all train S1 competing

The final values are τ = 0.75 and δ = 0.0. HoldoutB is used only for final evaluation.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.99070** on the frozen holdoutB (220k S1; A1b 0.99079). Other measures:
  - precision 0.9988, recall 0.9724
  - accuracy on S1 with no true match 0.9959
  - US 0.9902, India 0.9915
- **Common false positives (wrong merges):** about 0.1-0.2% of predictions. They come from look-alike records:
  - same name on the same street with a house number off by one or truncated (`144` vs `145 Elcano Dr`, `1800` vs `180 Edgecliff Rd`)
  - same name with a different street in dense cities
  - same address with a different business name
  - empty-address records whose name matches several entities
- **Common false negatives (missed matches):** recall is the main remaining loss.
  - ~50% are **empty-address records whose exact name is shared by 2+ S1**: unresolvable, so we abstain.
  - ~20% are empty-address records with truncated or changed names ("Corporate Medical" for "Corporate Medical Alliance").
  - ~16% are **alias names** (an unrelated brand name at the S1's exact address).
  - ~8% have genuinely changed house numbers, which cannot be told apart from look-alikes.

---

## 6. Conclusion

- **Validation had to match test conditions:** full-density blocking and all entities competing in decoding.
- **Hard, clean training data for the cross-encoders was the key to accuracy:** look-alikes, suffix variations and empty addresses.
- The system reaches **0.9907** macro F0.5 on holdout with a fully scalable pipeline (~20 candidates and ~6 cross-encoder pairs per record).
- **Remaining losses** are mostly information-theoretic (identical-name records without addresses).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- **`src/`**: all source code. `s*` are training stages, `t*` are test stages, `ce_mdeberta.py` / `ce_mmbert.py` train and score the cross-encoders.
  - `common/`: data loading, text normalisation / transliteration, house-number parser
  - `utils/validate_submission.py`: the official validator
- **`models/`**: the trained LightGBM prefilter and XGBoost fusion. The e5 blocker and the 4 cross-encoders (~5.5 GB, above the 1 GB upload limit) are rebuilt from the training data by `run_training.sh`.
- **`README.md`, `requirements.txt`**: pinned versions (Python 3.11, CUDA 12.4).
- **Entry points:**
  - **`bash run_inference.sh`**: test data → `output/matching_results.tsv` + `output/candidate_pairs.tsv`, then the validator.
  - **`bash run_training.sh`**: trains every model from the train data (run first).

### B. Additional Results

| Holdout metric | Value |
| --- | ---: |
| Blocking oracle macro F0.5 (candidates only) | 0.9982 |
| Candidates per S1 | 20.3 |
| Cross-encoder shortlist pairs per S1 | 5.8 |
| True pairs covered by the shortlist | ~99% |
| Final macro F0.5 | 0.99070 |
| Precision / recall | 0.9988 / 0.9724 |
| Accuracy on S1 with no true match | 0.9959 |
