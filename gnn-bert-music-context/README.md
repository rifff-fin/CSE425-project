# GNN-BERT Music Context

A modular PyTorch project for music understanding that fuses graph-based audio representations with BERT-based textual context modeling. The repository supports four task formulations:

1. Multi-label tag classification with BERT
2. Graph-based audio encoding with GNNs
3. Joint GNN-BERT fusion for tag and emotion prediction
4. Contrastive audio-text retrieval with dual encoders
## Group 27
| Name | Student ID | Email | Section |
|---|---:|---|---:|
| Abdullah Al Rifat | 22201979 | `abdullah.al.rifat1@g.bracu.ac.bd` | 6 |
| Farhanul Haque Mridul | 22201964 | `farhanul.haque.mridul@g.bracu.ac.bd` | 6 |
| Sajid Safiullah | 23201686 | `sajid.safiullah@g.bracu.ac.bd` | 4 |
| Ariful Islam Naeem | 24141141 | `ariful.islam.naeem@g.bracu.ac.bd` | 6 |

## Repository and project layout
The Git repository root is `D:\\425 project`. This directory contains the course specification and the implementation directory. The **implementation project root** is `D:\\425 project\\gnn-bert-music-context`; run Python commands from there.

```text
D:\\425 project/                         # Git repository root
├── README.md                              # repository overview
├── CSE425_Project_GNN_BERT_Music_Context.md
├── CSE425_Project_GNN_BERT_Music_Context (1) (4).pdf
├── .gitignore
├── drive musics/                          # tracked demo audio files
└── gnn-bert-music-context/                 # implementation project root
    ├── README.md
    ├── requirements.txt
    ├── config.yaml
    ├── .gitignore
    ├── CURRENT_PROJECT_STATUS.md
    ├── PROJECT_REQUIREMENTS_AUDIT.md
    ├── TASK_1_3_SUBMISSION_CHECKLIST.md
    ├── TASK4_PROGRESS.md
    ├── TASK_4_EXPLANATION.md
    ├── data/
    │   ├── raw/                            # downloaded datasets; excluded from Git
    │   ├── processed/                       # graph fixtures and generated features
    │   ├── processed_full/                  # full processed FMA artifacts
    │   └── splits/                          # train/validation/test manifests
    ├── notebooks/
    │   ├── eda.ipynb
    │   ├── demo_context.ipynb
    │   └── demo_context.html
    ├── src/
    │   ├── audio_features.py               # mel, chroma, MFCC, segmentation
    │   ├── graph_builder.py                 # segment graph construction
    │   ├── preprocess_fma.py
    │   ├── bert_encoder.py                  # BERT/DistilBERT encoder
    │   ├── gnn_model.py                     # GraphSAGE/GAT encoder
    │   ├── fusion_model.py                  # cross-attention GNN-BERT
    │   ├── contrastive.py                   # Task 4 InfoNCE
    │   ├── train.py
    │   └── evaluate.py
    ├── results/
    │   ├── metrics.json
    │   ├── plots/
    │   └── retrieval_examples/
    ├── checkpoints/                         # local checkpoints; excluded from Git
    └── report/
        ├── final_report.tex
        └── final_report.md
```

Example Windows PowerShell setup:

```powershell
Set-Location 'D:\\425 project\\gnn-bert-music-context'
pip install -r requirements.txt
python src/train.py --task task1
```

## Environment setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

If PyG backend extensions are required for your platform, install the matching extra packages based on your CUDA version. The project is compatible with standard CPU or CUDA-enabled PyTorch builds.

## Configuration

The global training configuration lives in `config.yaml` and controls:

- audio sample rate: 22050 Hz
- log-mel representation: 128 bins
- chroma representation: 12 bins
- segment window size: 5 seconds
- graph similarity threshold: 0.7
- BERT backbone: `distilbert-base-uncased`
- batch size and learning rate
- graph and fusion capacity settings

## Dataset preparation

The project is designed for common audio-text/music datasets that support contextual tags and emotion labels.

> Strict verification note: the repository now includes a real MusicCaps caption-to-audio path. `src/download_musiccaps.py` downloads timestamped YouTube clips from the official MusicCaps CSV, and `src/preprocess_musiccaps.py` creates graph manifests only for audio files that exist locally. The checked-in retrieval result uses 95 verified pairs. Task 1 also provides a MusicCaps caption-to-tag proxy manifest under `data/splits/musiccaps_task1_proxy`; its labels are deterministic lexical caption matches, not independent human annotations.

### 1) FMA (Free Music Archive)
- Download the full FMA dataset or a curated subset.
- Place audio files in `data/raw/fma/`.
- Build metadata tables with track IDs, artist IDs, and genre labels when needed.
- Use the audio files for segment extraction and graph construction.

### 2) MagnaTagATune
- Use this dataset as an optional future source for multi-label music tagging.
- The current repo does not include a verified MagnaTagATune audio manifest or audio-aligned retrieval benchmark.
- Keep file-level metadata in `data/raw/metadata.csv` only when a valid audio mapping is established.

### 3) DEAM (Database for Emotional Analysis of Music)
- Use continuous valence and arousal labels from the DEAM annotations.
- Align emotion targets to each track or segment.
- Normalize labels to the expected regression range before training.

### 4) MusicCaps
- Download timestamped clips and captions with `src/download_musiccaps.py`.
- Build graph/audio features and leakage-safe train/val/test manifests with `src/preprocess_musiccaps.py`.
- Train and evaluate Task 4 against the MusicCaps manifest, not FMA metadata text:

```bash
python src/download_musiccaps.py --limit 100
python src/preprocess_musiccaps.py --limit 100
python src/train.py --task task4 --manifest-root data/splits/musiccaps --epochs 20 --batch-size 16 --lr 1e-4
python src/evaluate_fma_multimodal.py --task task4 --manifest-root data/splits/musiccaps --checkpoint checkpoints/task4_epoch_20.pt --output results/musiccaps_task4_metrics.json
```

## Training commands

Run the training script for each task with the appropriate CLI argument.

### Task 1: BERT tag classification

```bash
python src/train.py --task task1 --epochs 10 --batch-size 16 --lr 2e-5
```

### Task 2: GNN audio encoder

```bash
python src/train.py --task task2 --epochs 15 --batch-size 16 --lr 1e-4
```

### Task 3: GNN-BERT fusion with emotion regression

```bash
python src/train.py --task task3 --epochs 20 --batch-size 8 --lr 2e-5
```

### Task 4: Contrastive audio-text retrieval

```bash
python src/train.py --task task4 --epochs 20 --batch-size 16 --lr 1e-4
```

For the required MusicCaps experiment, use `--manifest-root data/splits/musiccaps` and the MusicCaps-specific preparation commands above.

## Evaluation commands

```bash
python src/evaluate.py
```

The evaluation module computes:

- Macro-F1 and Micro-F1 for multilabel tags
- AUC-PR for tagging quality
- MAE for valence/arousal prediction
- Retrieval Recall@1, Recall@5, and Recall@10 for contrastive retrieval

## Notebook usage

Open the real-data demo notebook:

```bash
jupyter notebook notebooks/demo_context.ipynb
```

The notebook runs a real held-out FMA example through the stored Task 3 checkpoint, visualizes its PyG graph, and reports predictions from real metadata text. Verified DEAM audio/emotion manifests are also summarized separately. Synthetic smoke data is still available only through the explicit `--synthetic` flag in `src/train.py`; it is not the default project path.

## Core modeling components

- `src/audio_features.py`: Librosa-based audio preprocessing and segment extraction
- `src/graph_builder.py`: PyG graph construction from segment descriptors
- `src/bert_encoder.py`: BERT token sequence and pooled representation encoder
- `src/gnn_model.py`: GraphSAGE/GAT-style graph encoder
- `src/fusion_model.py`: Cross-attention fusion for tag and emotion prediction
- `src/contrastive.py`: InfoNCE dual-encoder retrieval module
- `src/train.py`: unified training orchestration
- `src/evaluate.py`: metric computation and validation reporting

## Notes

This project is designed as a research-oriented and modular starter for multi-modal music understanding, with emphasis on clear code organization, tensor validation, and reproducible experiment workflows.
