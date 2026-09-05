# GNN-BERT Music Context

A modular PyTorch project for music understanding that fuses graph-based audio representations with BERT-based textual context modeling. The repository supports four task formulations:

1. Multi-label tag classification with BERT
2. Graph-based audio encoding with GNNs
3. Joint GNN-BERT fusion for tag and emotion prediction
4. Contrastive audio-text retrieval with dual encoders

## Repository layout

```text
gnn-bert-music-context/
├── config.yaml
├── requirements.txt
├── README.md
├── src/
│   ├── audio_features.py
│   ├── graph_builder.py
│   ├── preprocess_fma.py
│   ├── enrich_fma_metadata.py
│   ├── fma_dataset.py
│   ├── bert_encoder.py
│   ├── gnn_model.py
│   ├── fusion_model.py
│   ├── contrastive.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── demo_context.ipynb
└── data/
    ├── raw/              # downloaded datasets; excluded from Git
    ├── processed/        # graph fixtures and generated features
    └── splits/            # train/validation/test manifests
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

> Strict verification note: the repository now includes a real MusicCaps caption-to-audio path. `src/download_musiccaps.py` downloads timestamped YouTube clips from the official MusicCaps CSV, and `src/preprocess_musiccaps.py` creates graph manifests only for audio files that exist locally. The checked-in smoke result uses three verified pairs; a full benchmark requires running the same commands with a larger limit.

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
