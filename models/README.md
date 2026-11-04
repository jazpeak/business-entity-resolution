# Models

This directory holds the trained models for the pipeline.

## Included in the repo

| Path | Model | Size |
| --- | --- | ---: |
| `prefilter_lgbm.txt` | LightGBM candidate prefilter | ~21 MB |
| `fusion/fusion_final.json` | XGBoost fusion model | small |
| `fusion/fusion_final_meta.json` | fusion feature list + decoder thresholds (`tau=0.75`, `delta=0.0`) and holdout scores | small |
| `blocker_e5/`, `ce_tagged/`, `ce_nameonly/`, `ce_street/`, `ce_mmbert/` | tokenizer + config **only** (no weights) | small |

## Not in the repo: the five transformer weights (~5.5 GB)

The fine-tuned transformer weights are too large for git and are published on the Hugging Face Hub instead. Each `models/<name>/` directory already contains the tokenizer and `config.json`; you only need to drop the weight files (`model.safetensors`) in next to them.

| Directory | Base model (MIT) | Role |
| --- | --- | --- |
| `blocker_e5/` | `intfloat/multilingual-e5-base` | dense blocking bi-encoder |
| `ce_tagged/` | `microsoft/mdeberta-v3-base` | main cross-encoder (`[MISSING]`-tagged empty addresses) |
| `ce_nameonly/` | `microsoft/mdeberta-v3-base` | empty-address, name-only cross-encoder |
| `ce_street/` | `microsoft/mdeberta-v3-base` | street-aware cross-encoder |
| `ce_mmbert/` | `jhu-clsp/mmBERT-base` | look-alike cross-encoder (2.9M-pair training) |

### Option A — download the published weights

```bash
python ../scripts/download_models_from_hf.py --repo jazl26/business-entity-resolution-models --dest .
```

### Option B — rebuild from the training data

From the repo root:

```bash
bash run_training.sh
```

This fine-tunes the blocker and all four cross-encoders from the pretrained bases (downloaded once from Hugging Face) and writes them here. ~15–20 h on 1–2 A100 GPUs. GPU training is not bit-deterministic, so rebuilt weights reproduce the published outputs up to tiny differences.
