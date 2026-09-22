# Step 2 — Deduplication Report

**pHash threshold:** 5 (Hamming)

## Per source

| Source | Raw files | After augmentation-stripping |
|---|---|---|
| mendeley | 1024 | 1024 |
| roboflow | 7014 | 2013 |

## Deduplication

| Metric | Value |
|---|---|
| Source images in | 3035 |
| Exact duplicates (SHA-256) | 5 |
| **Cross-source exact duplicates** | **0** |
| Conflicting labels | 3 |
| Near-duplicate clusters | 162 |
| Images inside a cluster | 394 |
| **Cross-source near-dup clusters** | **75** |
| **TRUE dataset size (groups)** | **2798** |
| Lesion groups | 1833 |
| Normal groups | 965 |
| Imbalance | 0.53:1 |
| Groups with mixed labels | 0 |

Report `2798` as the dataset size. Never the file count.

`group_id` in `ai/data/manifest.csv` goes to `StratifiedGroupKFold(groups=...)`.
An entire group belongs to exactly one fold.
