---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:800000
- loss:MultipleNegativesRankingLoss
base_model: intfloat/multilingual-e5-base
widget:
- source_sentence: 'query: Miller Cap | ME, Saint George, 32 Glenmere Road'
  sentences:
  - 'query: Surgical Clinic LP | '
  - 'query: miller cap | '
  - 'query: Paz and Frazier Llc | '
- source_sentence: 'query: Christ Chapel | 10424 9th Avenue, Seattle, WA'
  sentences:
  - 'query: Vision Anchor | '
  - 'query: Christ (Chapel) | '
  - 'query: cook.com | 7053 400, <NULL>, ORLEANS, IN'
- source_sentence: 'query: Lynch Union | 6746 Putting Green Drive, Dallas, TX'
  sentences:
  - 'query: Delacruz-Best Bond | '
  - 'query: Chat | 95 HILLSIDE AVE, FREEPORT, NY'
  - 'query: LYNCH UNION | '
- source_sentence: 'query: Rosaon | KY, 11106 89, Mckee'
  sentences:
  - 'query: Rosaon | 11106-11108 89, Mckee, Kentucky'
  - 'query: QUFLOSOFT LP  ASSETS | 17, ROSCOE, NY'
  - 'query: STOBER DIAMOND GREAT INC | '
- source_sentence: 'query: TK Egh Inc | 9400 Pope Court, Lipan, TX'
  sentences:
  - 'query: TK Egh | POPE COURT, LIPAN, TX'
  - 'query: Faircloth Holdings | '
  - 'query: Inc Atlanta | Ault Rd, Belmont, Ohio'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on intfloat/multilingual-e5-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base). It maps inputs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) <!-- at revision d128750597153bb5987e10b1c3493a34e5a4502a -->
- **Maximum Sequence Length:** 64 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'query: TK Egh Inc | 9400 Pope Court, Lipan, TX',
    'query: TK Egh | POPE COURT, LIPAN, TX',
    'query: Faircloth Holdings | ',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.8758, -0.0072],
#         [ 0.8758,  1.0000, -0.0167],
#         [-0.0072, -0.0167,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 800,000 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                             | positive                                                                           |
  |:---------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type     | string                                                                             | string                                                                             |
  | modality | text                                                                               | text                                                                               |
  | details  | <ul><li>min: 17 tokens</li><li>mean: 30.08 tokens</li><li>max: 61 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 33.28 tokens</li><li>max: 64 tokens</li></ul> |
* Samples:
  | anchor                                                                                      | positive                                                                                   |
  |:--------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------|
  | <code>query: Olszewski Holding Company LLC \| 218 Shoal Creek, James City County, VA</code> | <code>query: Olszewski Holding Company LLC LLC \| 218 SHOAL CREEK, WILLIAMSBURG, VA</code> |
  | <code>query: Eye Valley Medicine Inc \| 39 Main Street, Savona, NY</code>                   | <code>query: Eye Valley Medicine Incorporated \| 39 MAIN STREET, SAVONA, NY</code>         |
  | <code>query: Mccloud's Trading \| 75 Palmer Street, Unit 2 FRONT, Fall River, MA</code>     | <code>query: MT \| 25 PALMER SAINT, FALL RIVER, MA</code>                                  |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 256
- `num_train_epochs`: 1.0
- `learning_rate`: 3e-05
- `warmup_steps`: 0.05
- `bf16`: True
- `seed`: 0
- `dataloader_num_workers`: 4
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 256
- `num_train_epochs`: 1.0
- `max_steps`: -1
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.05
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: True
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 0
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 4
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 0.032 | 100  | 0.9413        |
| 0.064 | 200  | 0.0163        |
| 0.096 | 300  | 0.0075        |
| 0.128 | 400  | 0.0075        |
| 0.16  | 500  | 0.0063        |
| 0.192 | 600  | 0.0051        |
| 0.224 | 700  | 0.0048        |
| 0.256 | 800  | 0.0044        |
| 0.288 | 900  | 0.0044        |
| 0.32  | 1000 | 0.0035        |
| 0.352 | 1100 | 0.0031        |
| 0.384 | 1200 | 0.0032        |
| 0.416 | 1300 | 0.0028        |
| 0.448 | 1400 | 0.0029        |
| 0.48  | 1500 | 0.0034        |
| 0.512 | 1600 | 0.0032        |
| 0.544 | 1700 | 0.0024        |
| 0.576 | 1800 | 0.0026        |
| 0.608 | 1900 | 0.0020        |
| 0.64  | 2000 | 0.0024        |
| 0.672 | 2100 | 0.0022        |
| 0.704 | 2200 | 0.0019        |
| 0.736 | 2300 | 0.0023        |
| 0.768 | 2400 | 0.0021        |
| 0.8   | 2500 | 0.0016        |
| 0.832 | 2600 | 0.0018        |
| 0.864 | 2700 | 0.0019        |
| 0.896 | 2800 | 0.0016        |
| 0.928 | 2900 | 0.0018        |
| 0.96  | 3000 | 0.0016        |
| 0.992 | 3100 | 0.0015        |


### Training Time
- **Training**: 29.8 minutes

### Framework Versions
- Python: 3.11.16
- Sentence Transformers: 6.1.0
- Transformers: 5.17.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.15.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->