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
│   ├── bert_encoder.py
│   ├── gnn_model.py
│   ├── fusion_model.py
│   ├── contrastive.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── demo_context.ipynb
└── data/
    ├── audio/
    ├── metadata.csv
    └── captions.json
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

### 1) FMA (Free Music Archive)
- Download the full FMA dataset or a curated subset.
- Place audio files in `data/audio/`.
- Build metadata tables with track IDs, artist IDs, and genre labels when needed.
- Use the audio files for segment extraction and graph construction.

### 2) MagnaTagATune
- Use MIDI-like or audio labels for multi-label tagging.
- Map annotations to a fixed label vocabulary for classification tasks.
- Keep file-level metadata in `data/metadata.csv`.

### 3) DEAM (Database for Emotional Analysis of Music)
- Use continuous valence and arousal labels from the DEAM annotations.
- Align emotion targets to each track or segment.
- Normalize labels to the expected regression range before training.

### 4) MusicCaps
- Use the MusicCaps captioning data as text supervision.
- Store text descriptions or prompt strings in `data/captions.json`.
- Pair each caption with the corresponding audio track or segment.

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

Open the demo notebook:

```bash
jupyter notebook notebooks/demo_context.ipynb
```

The notebook simulates a short synthetic 10-second music clip, converts it into a PyG graph, passes it through the fusion model, and visualizes the graph structure and predictions.

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
