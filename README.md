# TEMPO — Sign Language Recognition

1st Place Solution for MSLR 2026 Track 1

**Competition:** MSLR 2026 @ CVPR 2026 — Track 1: Signer-Independent Continuous Arabic Sign Language Recognition

| Track | Dev WER | Test WER |
|:---|:---:|:---:|
| **Track 1 — Signer Independent** | **9.76%** | **5.68%** |

---

## Solution Summary

Cosign-TempDP-CSLR is a skeleton-based Continuous Sign Language Recognition (CSLR) system for signer-independent Arabic sign language recognition. The backbone is a two-stream CoSign-2s Spatial-Temporal GCN that processes static and motion skeleton features in parallel. On top of the GCN we stack three temporal modules:

- **TAPE** — a lightweight positional adapter that introduces frame-level temporal context into the skeleton feature stream without heavy computational overhead.
- **MS-TCN + LiftPool** — a multi-scale temporal convolutional network with a learnable pooling layer for capturing gesture dynamics across multiple timescales.
- **Hybrid BiLSTM + Transformer** — a sequential backend that combines the local sequential modeling of BiLSTM with the global attention of a Transformer encoder.

Training uses a multi-term CTC loss with cross-stream consistency regularization across static, motion, and fusion branches. At inference, a dynamic-programming post-processing step applies a Levenshtein+Jaccard lexicon correction pass over the raw CTC output to further reduce WER.

---

## Requirements

```bash
pip install torch torchvision tqdm pyyaml numpy
```

- Python 3.8+, PyTorch ≥ 1.9, CUDA GPU

---

## Project Structure

```
TEMPO/
├── run.py                          # Unified CLI entry point
├── temposlr/                       # Main package
│   ├── model.py                    # TwoStream_Cosign model
│   ├── extract.py                  # BoundaryExtractor (train/dev/test)
│   ├── runner.py                   # SLRProcessor (training loop)
│   ├── loops.py                    # seq_train, seq_eval
│   ├── data/                       # Ground truth CSVs
│   │   ├── train.csv               # 19500 samples
│   │   ├── dev.csv                 # 1950 samples
│   │   └── test.csv                # 7800 samples
│   ├── datasets/                   # SkeletonFeeder + info JSONs
│   ├── modules/                    # Visual extractor, STGCN, BiLSTM, etc.
│   ├── utils/                      # Decode, boundary extraction, etc.
│   └── evaluation/                 # WER computation
├── configs/
│   ├── train.yaml                  # Training config
│   └── test.yaml                   # Eval/inference config
└── scripts/
    └── generate_dataset_files.py   # Dataset generation
```

---

## Quick Start

### Train

```bash
python run.py train --config configs/train.yaml
```

### Evaluate (dev set)

```bash
python run.py eval --config configs/test.yaml --split dev
```

### Extract Gloss Boundaries

Extract start/end frame for each predicted gloss from CTC logits:

```bash
# single sample (dev split)
python run.py extract 02_0001

# multiple samples
python run.py extract 02_0001 02_0002 02_0003

# first N samples from dev set
python run.py extract --split dev --all --limit 10

# train split
python run.py extract --split train --all --limit 5

# JSON output
python run.py extract 02_0001 --json

# CSV output
python run.py extract 02_0001 02_0002 --csv
```

### Predict (glosses only)

```bash
python run.py predict --split dev 02_0001 02_0002 02_0003
```

### As a Python module

```bash
python -m temposlr.extract --split dev 02_0001 02_0002
```

---

## Library Usage

```python
from temposlr.extract import BoundaryExtractor

# Initialize (loads model once)
ext = BoundaryExtractor(split="dev")

# Extract boundaries for one sample
result = ext("02_0001")
# {"sample_id": "02_0001", "t_orig": 32, "feat_len": 11,
#  "segments": [{"gloss": "سوال", "label_id": 585,
#                "start": 0, "end": 13,
#                "start_feat": 0, "end_feat": 5}]}

# Batch extract
results = ext.batch(["02_0001", "02_0002"])

# Get all sample IDs for a split
all_ids = ext.all_sample_ids()
```

### Output Fields

| Field | Description |
|:---|:---|
| `gloss` | Predicted gloss name (Arabic) |
| `label_id` | Class index (1–1113) |
| `start` / `end` | Start/end frame in original video (inclusive) |
| `start_feat` / `end_feat` | Start/end frame in model feature space |

Config lives at `configs/test.yaml`. Edit `load_checkpoints` to point to your checkpoint.

---

*Work done by **Ahmed Hassan** (ahmed.hasan@ejust.edu.eg) and **Nadine Alsayad** (nadine.alsayad@ejust.edu.eg)*
