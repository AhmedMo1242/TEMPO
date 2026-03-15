# Cosign-TempDP-CSLR

1st Place Solution for MSLR 2026 Track 1

**Competition:** MSLR 2026 @ CVPR 2026 — Track 1: Signer-Independent Continuous Arabic Sign Language Recognition

| Track | Dev WER | Test WER |
|:---|:---:|:---:|
| **Track 1 — Signer Independent** | **9.76%** | **5.68%** |

---

## Solution Summary

Cosign-TempDP-CSLR is our skeleton-based Continuous Sign Language Recognition (CSLR) system for signer-independent Arabic sign language recognition. It transfers temporal modeling techniques from RGB-based methods to a pure skeleton pipeline. The backbone is a two-stream CoSign-2s Spatial-Temporal GCN that processes static and motion skeleton features in parallel. On top of the GCN we stack three temporal modules transferred from RGB CSLR literature:

- **TAPE** — a lightweight positional adapter that introduces frame-level temporal context into the skeleton feature stream without heavy computational overhead.
- **MS-TCN + LiftPool** — a multi-scale temporal convolutional network with a learnable pooling layer for capturing gesture dynamics across multiple timescales.
- **Hybrid BiLSTM + Transformer** — a sequential backend that combines the local sequential modeling of BiLSTM with the global attention of a Transformer encoder.

Training uses a multi-term CTC loss with cross-stream consistency regularization across static, motion, and fusion branches. At inference, a dynamic-programming post-processing step applies a Levenshtein+Jaccard lexicon correction pass over the raw CTC output to further reduce WER.

The full system (TAPE + MS-TCN + Transformer) achieves **5.68% WER** on the test set, a significant improvement over the CoSign-2s baseline (~18% WER). Two additional regularization modules — Stochastic Sequence Depth (SSD) and Cross-Stream Consistency (CSC) — are explored as ablation variants.

---

## Requirements

```bash
pip install torch torchvision tqdm pyyaml numpy
```

- Python 3.8+, PyTorch ≥ 2.0, CUDA GPU

---

## Running

### 1. Preprocess

```bash
python preprocess_data.py \
    --pkl_path /path/to/skeleton_data_SI.pkl \
    --train_csv /path/to/train.csv \
    --dev_csv /path/to/dev.csv \
    --output_dir datasets/mslr2026 \
    --setting si
```

### 2. Train

```bash
# train
python main.py --config configs/name.yaml
```

### 3. Post-process predictions

```bash
python postprocess.py \
    --input  work_dir/test.csv \
    --output submission/test.csv \
    --train  /path/to/train.csv \
    --alpha  0.5
```


---

*Work done by **Ahmed Hassan** (ahmed.hasan@ejust.edu.eg) and **Nadine Alsayad** (nadine.alsayad@ejust.edu.eg)*
