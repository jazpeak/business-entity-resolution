# Scalable Blocking for Business Entity Resolution — Results & Recommendations

Sep 26, 2026

## 1. Executive summary

The recommended blocker is a fine-tuned multilingual dense retriever searched with an approximate (FAISS IVF) index. On a leak-free, test-density holdout it reaches **oracle macro F0.5 0.9982 at ~20.3 candidates per S1**, versus 0.9894 at 79.5 for the previous best design (B2).

| Blocker (all numbers: frozen holdout, full-density index) | Candidates / S1 | Oracle macro F0.5 | Scalable? |
| --- | ---: | ---: | --- |
| Teammate K10 lexical blocker (recreated) | 37.9 | 0.9778 | yes, with selective postings |
| Teammate B2 (lexical K20 + reverse top-1, recreated) | 79.5 | 0.9894 | yes, with selective postings |
| Best lexical (ours): norm name + number-aware address K10 + reverse top-2 | 45.9 | 0.9913 | only with an inverted-index budget |
| Lexical pool + LightGBM re-ranker, keep top-10 | 10 | 0.9931 | yes |
| **Recommended: dense e5 + FAISS IVF (nprobe 64), top-10/source + reverse top-1 capped at 10** | **20.3** | **0.9982** | **yes (~1% of index scanned per query)** |
| Same dense model, exact brute-force search (measurement only) | 20.2 | 0.9988 | no |

Key takeaways:

- Validation matters more than any single trick: the earlier 20%-index validation inflated oracle by ~0.011 (0.9775 to 0.9887 for the same blocker).
- Dense retrieval dominates lexical retrieval; approximate search costs only 0.0006 oracle once top-10 is retrieved.
- Reverse search (aux to S1) is the strongest lexical idea but creates hub S1 with thousands of candidates; a cap of 10 removes them at zero cost.
- France's lexical address search was weak because char n-grams ignore short house numbers; a number-aware address view fixed it (agreement 0.40 to 0.80). The dense model already handles numbers (88% same-number top-1 on France).

## 2. Validation protocol for blocking

All blocking numbers in this report come from a frozen S1 holdout, with every query searched against the full-size train index.

**Split.** File `experiments/splits/s1_split_v1.tsv` (sha256 `0eb6497f...`). Holdout = train S1 whose `int(sha1(entity_id)[:8], 16) % 5 == 0` (20%: 265,228 US + 175,868 India); the rest is `dev`. Each S1 keeps all its true S2/S3 matches (components are one-S1 stars). Singleton share (5.5-5.6%) and mean true matches (3.46) are equal across holdout and dev.

**Evaluation sample.** 20,000 US + 20,000 India holdout S1, stratified by true-match cardinality (0..7+). Design work and example browsing used dev only; learned components (dense model, re-ranker) were trained on dev only.

**Index density (the important part).**

- Forward search: each holdout S1 searches ALL train S2/S3 records of its country (US 3.0M + 3.2M, India 2.0M + 2.1M).
- Reverse search: every train S2/S3 record searches ALL train S1 of its country.
- This is leak-free for blocking (no labels are involved; other entities' records act as distractors exactly as on test) and is close to test density (test US 1.87M + 1.95M, India 2.31M + 2.41M, France 0.70M + 0.73M).

**Why the earlier validation numbers were optimistic.** The previous B2 validation searched a ~20% world index (the population's own true matches plus ~20% of unmatched records). The same K10 blocker scores 0.9887 there versus 0.9775 on the full index, and B2 scores 0.9946 versus 0.9888. Separately, the earlier 10k-cohort decoding only let 10k S1 compete for each aux record, which hid the one-owner/margin problem.

**Metrics.**

- Candidates / S1: mean candidate count (reverse candidates included), plus p99 and max to expose hubs.
- Edge recall: share of true (S1, aux) edges present in the candidate set.
- Oracle macro F0.5: F0.5 of a perfect matcher restricted to the candidates, per S1 then averaged. Precision is 1; recall = found / true; true singletons score 1. It is the ceiling the matcher can reach.
- Averages are train-weighted (60% US / 40% India). France has no labels (Section 7).

## 3. Baselines recreated and head-to-head comparison

Both teammate blockers were recreated with exact search and reproduce the reported numbers on the 20% world index, so the comparisons below are like-for-like.

**Recreated designs.**

- K10 blocker: same-country; four char TF-IDF streams (S1 name to S2 names, S1 name to S3 names with char 3-5-grams; S1 address to S2 / S3 addresses with char 3-grams); top-10 per stream; union.
- B2: the same four streams at top-20, plus reverse search (every S2/S3 record proposes its top-1 S1 by name and by address).

**Reproduction check.**

| Blocker | Index | Oracle macro F0.5 | Reported by teammate |
| --- | --- | ---: | ---: |
| K10 | 20% world | 0.9887 | 0.98746 |
| K10 | full train | 0.9775 | - |
| B2 | 20% world | 0.9946 | ~0.995 (gate) |
| B2 | full train | 0.9888 | - |

**Head-to-head on the frozen holdout, full-density index.**

| Blocker | Candidates / S1 | Oracle macro F0.5 | US oracle | India oracle | vs teammate K10 | vs teammate B2 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Teammate K10 | 37.9 | 0.9778 | 0.9835 | 0.9692 | - | - |
| Teammate B2 | 79.5 | 0.9894 | 0.9940 | 0.9826 | - | - |
| Lexical ours: K3 + reverse top-1 | 14.8 | 0.9791 | 0.9834 | 0.9727 | +0.0013 with 2.6x fewer | - |
| Lexical ours: K7 + reverse top-2 | 35.4 | 0.9880 | 0.9919 | 0.9821 | +0.0102, fewer candidates | - |
| Lexical ours: K10 + reverse top-2 | 46.1 | 0.9896 | 0.9934 | 0.9839 | +0.0118 | same oracle, 42% fewer |
| Lexical ours + number-aware address, K10 + reverse top-2 | 45.9 | 0.9913 | 0.9940 | 0.9873 | +0.0135 | +0.0019, 42% fewer |
| Dense (approx), top-3/source + reverse top-1 | 6.8 | 0.9945 | 0.9944 | 0.9947 | +0.0167 with 5.6x fewer | +0.0051 with 12x fewer |
| **Dense (approx), top-10/source + reverse top-1, cap 10** | **20.3** | **0.9982** | **0.9982** | **0.9981** | **+0.0204** | **+0.0088 with 3.9x fewer** |

"Lexical ours" = normalized name streams + raw-address streams + native-script skeleton stream (top-1) + reverse search on name, address and skeleton. Dense rows use the FAISS IVF index at nprobe 64 (Section 4).

## 4. Recommended scalable blocker (blocker v1)

Blocker v1 = fine-tuned multilingual-e5-base embeddings, searched with FAISS IVF-Flat at nprobe 64, top-10 per source forward plus reverse top-1 capped at 10 per S1.

**Model.**

- Base: `intfloat/multilingual-e5-base` (278M parameters, MIT license; within the MIT/Apache-2.0, 8B limit).
- Fine-tuned as a bi-encoder with MultipleNegativesRankingLoss (in-batch negatives), no-duplicate batches, 1 epoch, batch 256, lr 3e-5, max_seq_length 64, bf16. About 30 minutes on one A100.
- Training data: 800,000 true (S1, aux) pairs sampled from **dev** S1 only. Holdout S1 and their records were never seen.
- Input text for every record: `query: {business_name} | {business_address}`. Embeddings are L2-normalized, so inner product = cosine.

**Search (all within literal country equality).**

1. Embed every S1, S2 and S3 record once.
2. Per country and source, build a FAISS IVF-Flat index (inverted lists = power of two near 4 x sqrt(N), e.g. 8,192 for 3M records; k-means trained on a 50-per-list sample).
3. Forward: each S1 takes its top-10 S2 and top-10 S3 records (nprobe 64, about 0.8-1.6% of an index scanned per query).
4. Reverse: each S2/S3 record finds its top-1 S1 (IVF index over the country's S1, nprobe 64); the record becomes a candidate of that S1.
5. Cap reverse candidates at 10 per S1 (highest similarity first). Union with forward candidates.

**Results (frozen holdout, full density).**

| Setting | Candidates / S1 | p99 | max | Oracle macro F0.5 |
| --- | ---: | ---: | ---: | ---: |
| Approx, top-3/source + reverse top-1, cap 10 | 6.8 | 10 | 14 | 0.9945 |
| Approx, top-5/source | 10.0 | 10 | 10 | 0.9957 |
| **Approx, top-10/source + reverse top-1, cap 10 (v1)** | **20.3** | **23** | **28** | **0.9982** |
| Exact, top-10/source + reverse top-1 (not scalable, reference) | 20.2 | 22 | 66 | 0.9988 |

Dense retrieval recovers 3,422 of the 4,367 true edges that the best lexical blocker (K10 + reverse top-2) misses, including 1,240 of 1,268 native-script ones. Its weakest case is S2/S3 records with an empty address (about 94% recall at top-10 exact).

**Cost (one A100 40GB; test = 11.7M records).**

| Stage | Measured | Test-set estimate |
| --- | --- | --- |
| Embedding | 3,500-6,100 records/s | ~40-55 min on one GPU, half on two |
| Index build | 5-25 s per index (GPU FAISS) | a few minutes in total |
| Forward search | seconds per 20k queries | minutes |
| Reverse search | ~29,000 queries/s at nprobe 64 | ~6 min for 10M records |
| Memory | 1.5 KB per vector (fp16 768-d) | ~18 GB; 96 B per vector with IVF-PQ |

## 5. Scalable vs non-scalable components

Only approximate-index dense search, budgeted inverted-index lexical search and per-candidate models are submission-grade. Exact top-k search was used to measure ceilings and must not ship, because it scores every query against every same-country record.

| Component | Exploration version (NOT scalable) | Submission version (scalable) | Cost per query |
| --- | --- | --- | --- |
| Dense retrieval | Exact GPU brute force (B005): every S1 x every same-country record | FAISS IVF-Flat, nprobe 64 (B008) | ~1% of the index; sublinear in index size |
| Lexical char TF-IDF | Exact sparse top-k (B001-B007): touches every record sharing any n-gram | Inverted index with a posting budget (teammate's selective postings R006B; measured equal to exact: 0.9878 vs 0.9875) or BM25 on an inverted index | bounded posting work per query |
| Reverse search | Same as the search it uses | Same index type as above, plus a per-S1 cap (10) | one query per aux record |
| Re-ranker (LightGBM) | - | Scores only the candidates it receives | linear in candidates |
| Embedding | - | Each record encoded once | linear in records |

**Why exact search is not acceptable.** For the test set, exact forward dense search is 663k x 3.82M (US) + 810k x 4.72M (India) + 259k x 1.43M (France), about 6.7 trillion vector comparisons, with a similar amount for reverse. An A100 can finish that for this test size, but it grows quadratically and is exactly the "compare every record with every other one" the problem statement rules out.

**Why IVF is.** Records are grouped into ~8,000 clusters; each query checks only the 64 nearest clusters, about 250x fewer comparisons than exact at this size. The scanned fraction shrinks as the index grows, which is how billion-vector search runs in production. With product quantization (IVF-PQ, 96 bytes per vector instead of 1,536) one billion records fit in about 96 GB, at a measured oracle cost of 0.002-0.004 (Section 6, B008).

**Other scalability properties.**

- Country partition: every index and search stays inside one country (0 of 7.64M true train edges cross countries).
- Candidate count is bounded: top-k per source plus a capped reverse contribution.
- No component needs pairwise comparison of whole sources.

## 6. Experiment log

Each experiment below was measured with the protocol in Section 2 unless noted; B001 and the B2 check used a random 20k US + 20k India train sample before the holdout split was frozen.

| ID | Question | Result | Decision |
| --- | --- | --- | --- |
| B001 | Do transliterated / phonetic name views recover native-script matches? | 13-24% of India S2/S3 names are in Indic scripts; S1 is 100% Latin. Normalized names + **raw** addresses: 0.9775 to 0.9798 at equal budget (stripping punctuation from India addresses hurts). Native-only phonetic skeleton stream: +0.0005 (k1, +0.8 cand) to +0.0010 (k3, +2.4 cand); native-edge recall 82.7% to 85.6%. Replacing the name stream with the skeleton hurts (0.9751). | Keep norm names, raw addresses, native skeleton stream |
| B2 check | How good is B2 at realistic density? | 0.9946 on the 20% world vs 0.9888 on the full index; forward K10 + reverse top-1 = 0.9860 at ~40 cand, better than forward K20 alone (0.9839 at 78). | Reverse search is the best lexical idea |
| B002 | Combine preprocessing + skeleton + reverse; sweep budgets | Reverse top-2 beats a larger forward K at equal volume everywhere (K5 rev2 0.9866 @28.5 vs K7 rev1 0.9857 @29.4). Asymmetric name/address K: no gain. Transliterated vs normalized names: tie. | Lexical Pareto curve established |
| B003 | Exact reverse search for all 10.3M train aux | Confirms B002 estimates (K10 rev2 0.9896 @46.1). Finds hub S1: max 5,800-7,900 candidates for one S1, p99 100-160. Replacing torch.topk with k passes of max+mask made reverse search 33x faster (identical results). | Reverse needs a per-S1 cap |
| B004 | LightGBM re-ranker over the lexical pool (fwd K20 + rev top-3, ~105 cand, oracle 0.9932), trained on dev | Top-6: 0.9908; top-10: 0.9931; p >= 0.02: 0.9910 at 4.45 cand (p95 8). Top features: aux reverse top-1 flag, address reverse rank, house-number overlap. | Re-ranker cut works; reuse on the final pool |
| B005 | Fine-tuned multilingual-e5-base dense retrieval (exact search) | Top-3/source + rev1: 0.9960 @6.7; top-5: 0.9967 @10; top-10 + rev1: 0.9988 @20.2; lexical K5 rev2 + dense fwd3 rev1: 0.9981 @30.4. | Dense becomes the core |
| B006 | France label-free analysis | Section 7 | Address search needs number awareness |
| B007 | Number-aware and abbreviation address views (tested separately) | Number-boost: 0.9896 to 0.9913 (India +0.0034). Abbreviation table: 0.9893 (slightly negative on US/India). Number-key extra stream: +0.0003-0.0005 for 2-5 extra cand. France agreement in Section 7. | Keep number-boost (+ abbreviations only for France gain) |
| B008 | Replace exact dense search with FAISS approximate search | IVF-Flat nprobe 64: 0.9982 @20.3 (exact 0.9988); nprobe 32: 0.9973; nprobe 8: 0.9918; IVF-PQ96: 0.9959. Reverse cap 10: max candidates 63 to 14 at zero oracle cost. | Blocker v1 = IVF-Flat nprobe 64 |

All exact oracle numbers used GPU char TF-IDF (hashed n-grams, sublinear TF, smoothed IDF) verified against scikit-learn + sparse_dot_topn; differences were only ties at the rank-K boundary.

## 7. France: house numbers and label-free checks

France has no labels, so it was checked with label-free signals on the real test records, calibrated on US/India where labels exist. The lexical address search was weak on France because char n-grams barely see short house numbers; a number-aware address view fixed it, and the dense model already handles numbers.

**What the French test data looks like (B006).** The noise comes from the same generator as US/India (S2 addresses partly uppercase, `t/a` only in S3, ~2.7% domain names, accents added/removed). France-specific patterns:

| Pattern | S1 | S2 / S3 |
| --- | ---: | ---: |
| Address contains the word `Rue` | 65.8% | 38-39% (+12% written `R` / `R.`) |
| Address starts with a house number | 86.0% | 69-70% |
| Region present (e.g. `Nouvelle-Aquitaine`) | always | often dropped or replaced by a department (`Gironde`, `Nord`, `Pas-de-Calais`) |
| Name contains `SARL` | 28.3% | 19.7-20.0% |

Cities are very dense (test S1: Bordeaux 37,215; Nantes 32,250; Lille 29,592) and names are generic (`Lille Club`, `Ecole`, `Centre Hospitalier`).

**Why char n-grams fail on French addresses.** `25 rue des tanneurs, lille, hauts-de-france` has 41 distinct char 3-grams, of which only 2 contain a digit. Cosine to candidates: `2 rue des tanneurs...` 0.963; `rue des tanneurs...` (no number) 0.963; `17 rue des tanneurs...` 0.951; the true match `25 r des tanneurs...` only 0.925. US house numbers are long (e.g. `14593`, 5 digit 3-grams), so the US is much less affected.

**Label-free signals on test data (3,000 S1 per country).** Agreement = the S1's top-1 address match names that same S1 as its own top-1. Right number = the top-1 match has the same first house number.

| Address view (lexical) | France agreement | France right number | US agreement | US right number | India agreement | India right number |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw (baseline) | 0.40 | 53% | 0.89 | 75% | 0.93 | 72% |
| Abbreviation table only | 0.48 | 63% | 0.90 | 76% | 0.93 | 71% |
| Number-boost only | 0.78 | 95% | 0.92 | 90% | 0.93 | 80% |
| Abbreviations + number-boost | 0.80 | 97% | 0.92 | 90% | 0.93 | 80% |
| Number-key (extra stream) | 0.84 | 96% | 0.88 | 95% | 0.80 | 94% |

- Number-boost = the lowercase address plus each house number appended twice as a tagged token (`#25#`), so `25` vs `2` becomes a clear mismatch.
- Abbreviation table = one country-agnostic table mapping to short forms (`street`/`st`, `rue`/`r`, `avenue`/`av`, `nagar`/`ngr`, `saint`/`st`; `N°`, `No.`, `H.No.` markers dropped). No country branching.
- On US/India labels (B007), number-boost improves oracle (0.9896 to 0.9913); abbreviations are neutral to slightly negative.

**Dense model on France (trained on US/India only).**

|  | Top-1 cosine (median) | Top-1 minus top-2 | Mutual best match | Top-1 same house number | Top-1 is a true match |
| --- | ---: | ---: | ---: | ---: | ---: |
| US (train, labelled) | 0.932 | 0.026 | 99.9% | 81% | 92.2% |
| India (train, labelled) | 0.939 | 0.032 | 99.8% | 74% | 97.2% |
| France (test, unlabelled) | 0.936 | 0.034 | 98.2% | 88% | unknown |

No sign of a France collapse: scores and ambiguity match US/India, and the house number agrees more often than for US/India. Mutual best match is slightly lower (98.2% vs 99.8%). Limitation: for dense, mutual best match is almost always true even when top-1 is wrong, so it can catch a collapse but cannot give a France recall figure.

**Remaining dense-street cases the matcher must handle:** number missing on one side (31% of French S2/S3 addresses lack a leading number), slightly changed numbers (3-6% of true matches on US/India), and several businesses at one address. House-number agreement, conflict and missing flags are therefore matcher features, not a hard filter.

## 8. Other scalable ideas to try

The highest-value next step is a hybrid pool (approximate dense + budgeted lexical) cut by a re-ranker to about 5-10 candidates per S1; the rest are ranked by expected value.

| Idea | What it targets | How it stays scalable | Expected value |
| --- | --- | --- | --- |
| Hybrid pool + re-ranker cut | Smaller candidate set with the union's recall | Both searches are indexed; re-ranker is linear in candidates | High: B004 kept the full pool ceiling (0.9932) with 10 of ~105 candidates |
| Budgeted lexical safety net (selective postings, number-aware address view, normalized names) | Empty-address and French address cases, model-independent recall | Posting-work budget per query (teammate's R006B) | Medium |
| BM25 on word tokens (names and addresses) | Rare words and house numbers as whole tokens (`25` is not `2`) | Classic inverted index (Lucene-style) | Medium, mainly as the lexical safety net |
| HNSW instead of IVF | Higher recall at the same cost; avoids k-means training | Graph search, logarithmic hops | Low-medium (IVF already loses only 0.0006 at top-10) |
| Dense hard-negative fine-tuning | Same street + different number, same name + different address, competing S1 | Training-only change | Medium, helps France and dense streets |
| Name abbreviation / legal-form table (`Pvt`, `Ltd`, `LLC`, `Corp`, `SARL`, `SAS`) | Organizer-listed name noise | Text normalization, linear | Low-medium, not yet tested |
| Adaptive K per S1 (small K when the top score has a clear margin) | Fewer candidates for easy S1 | Uses scores already computed | Medium for the candidate-count ranking |
| IVF-PQ / scalar quantization | Memory at billion scale | 96 B per vector | Only needed beyond this dataset (costs 0.002-0.004 oracle here) |
| Record-to-record (S2 to S3) linking | A clean S3 record pulls in its noisy S2 sibling | One extra indexed search per aux record | Low-medium, untested |

## 9. Caveats, open issues and reproducibility

The blocker is ready to use, but four things are unmeasured or open.

**Caveats.**

- France recall is unknown (no labels); only label-free signals were checked.
- Test is denser than train per S1 (5.75 S2/S3 records per S1 vs 4.68), and the test India index is ~15% larger than train India, so test recall may be slightly lower.
- Oracle F0.5 is a ceiling, not a final score; more candidates can make the matcher's job harder.
- The dense model was trained on dev pairs. Any matcher trained on blocker output must use S1 the blocker never saw (we use half of the holdout for matcher training and the other half for final evaluation).

**Artifacts (server `/home/arul24115/Entity/experiments/`).**

| Item | Path |
| --- | --- |
| Frozen split | `splits/s1_split_v1.tsv` (script `splits/make_split.py`) |
| Shared data loader | `common/erdata.py` |
| GPU char TF-IDF search, text and address views | `b001_native_script/b001_lib.py` (`GpuIndex`, `v_norm`, `v_translit`, `v_skel`, `v_addr_abbr`, `v_addr_numboost`) |
| Native-script experiment | `b001_native_script/b001_native_script.ipynb` |
| Combined lexical experiment | `b002_combined/b002_combined.ipynb` |
| Exact reverse search, all aux | `b003_reverse_all/rev_all.py` |
| Re-ranker | `b004_reranker/pool.py`, `run_b004.py`, `union_curves_fixed.py` |
| Dense model (training, embedding, eval) | `b005_dense/train_dense.py`, `embed_search.py`, `eval_b005.py`; model `b005_dense/models/e5b_dev_v1` |
| France analysis | `b006_france/france_text_stats.py`, `france_neighbors.py`; `b008_ann/france_dense_check.py` |
| Address views | `b007_address/run_streams.py`, `eval_b007.py`, `france_proxy.py` |
| Approximate dense search | `b008_ann/ann_eval.py` (results `b008_ann/artifacts/ann_oracle.csv`, `ann_recall.csv`) |
| Blocker v1 on all train S1 | `m001_pipeline/blocking_all.py` |

**Environment.** Conda env `entity` (Python 3.11): PyTorch 2.6 (CUDA 12.4), sentence-transformers 6.1, faiss-gpu-cu12 1.14.1, polars 1.44, rapidfuzz 3.14, LightGBM. Run with `HF_HUB_OFFLINE=1` (Hugging Face calls hang on this server) and `CUDA_DEVICE_ORDER=PCI_BUS_ID` so GPU numbers match `nvidia-smi`.
