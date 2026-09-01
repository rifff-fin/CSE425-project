# GNN-BERT Music Context

This repository contains a modular PyTorch project for music context understanding using a multi-task fusion of graph-based audio encoders and BERT-based text encoders.

## Tasks

1. BERT multi-label tag classification
2. GNN audio encoder on segment/chord graphs
3. Multi-task GNN-BERT fusion for tag classification and emotion regression
4. Dual-encoder contrastive retrieval for audio-text matching

## Repository structure

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
└── notebooks/
    └── demo_context.ipynb
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --task 1
```

## Configuration

The project is configured through `config.yaml` and can be adjusted for:
- audio sampling rate (22050 Hz)
- log-mel/chroma extraction
- graph construction parameters
- BERT architecture settings
- GNN hidden dimensions
- multi-task weighting coefficients `alpha` and `beta`

## Notes

This is a research-oriented starter template with placeholders for training logic, object-oriented modules, and tensor shape validation.
