# Data

The ML Challenge 2026 dataset is **not** redistributed in this repository. Obtain it from the challenge organisers and place the tab-separated files here, keeping the original folder names:

```
data/
├── train/
│   ├── train_source1.tsv       # Source 1 (deduplicated reference) training records
│   ├── train_source2.tsv       # Source 2 training records
│   ├── train_source3.tsv       # Source 3 training records
│   └── train_ground_truth.tsv  # source1_entity_id -> matched_entity_ids (comma list)
└── test/
    ├── test_source1.tsv        # generate matches for every entity here
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Format notes

- **All files are tab-separated (`.tsv`).** Addresses and the ID-list columns contain commas, so read with an explicit tab separator:

  ```python
  import pandas as pd
  df = pd.read_csv("data/train/train_source1.tsv", sep="\t")
  ```

  Reading without `sep="\t"` silently collapses each line into a single column.

- **Columns** of each source file: `entity_id` (prefix `S1-`/`S2-`/`S3-` indicates the source), `business_name`, `business_address`, `country`.
- **Countries:** training covers `US` and `India`; the test set also contains `France`, which does **not** appear in training. `country` is treated as an open set of string labels — the pipeline never hard-codes, filters, or one-hot-encodes to `{US, India}`.
- **Ground truth** (`train_ground_truth.tsv`): columns `source1_entity_id` and `matched_entity_ids` (comma-separated `S2-`/`S3-` IDs, empty for singletons).

`data/` is git-ignored (see [`../.gitignore`](../.gitignore)), so these files never get committed.
