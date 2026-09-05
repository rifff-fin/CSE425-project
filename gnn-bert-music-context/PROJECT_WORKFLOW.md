# GNN-BERT Music Context Project Workflow and Run Guide

This document provides the complete workflow for setting up, running, validating, and extending the GNN-BERT music context project.

## 1. Project overview

This repository targets a multimodal music understanding pipeline that combines:

- Audio signal processing with Librosa
- Graph construction with PyTorch Geometric (PyG)
- Text encoding with Hugging Face Transformers
- Cross-attention fusion for tag + emotion prediction
- Contrastive audio-text retrieval for multimodal matching

The project is organized around four task tracks using the verified real-data pipeline currently supported by the repo:

1. Task 1: BERT multi-label music tag classification
2. Task 2: GNN-based music audio encoding
3. Task 3: GNN-BERT fusion for tag and emotion prediction
4. Task 4: contrastive MusicCaps audio-text retrieval using timestamped clips and natural-language captions

Synthetic smoke data remains optional only via `--synthetic` and is not the default path used by the notebook or final validation workflow. MusicCaps manifests are accepted only after local audio files are downloaded and graph-processed.

---

## 2. Repository structure

```text
gnn-bert-music-context/
├── config.yaml
├── requirements.txt
├── README.md
├── PROJECT_WORKFLOW.md
├── src/
│   ├── audio_features.py
│   ├── graph_builder.py
│   ├── preprocess_fma.py
│   ├── bert_encoder.py
│   ├── gnn_model.py
│   ├── fusion_model.py
│   ├── contrastive.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── demo_context.ipynb
├── data/
│   ├── audio/
│   ├── metadata.csv
│   └── captions.json
├── checkpoints/
├── outputs/
└── .venv/
```

---

## 3. Prerequisites

### Required

- Windows 10/11, macOS, or Linux
- Python 3.10+
- pip
- Git
- A CUDA-capable GPU is optional; CPU execution works for development and demos

### Check Python availability

On Windows PowerShell:

```powershell
python --version
where.exe python
```

If Python is missing, install Python from python.org or Microsoft Store, then reopen the terminal and ensure it is in PATH.

> In the current environment used for this run, Python was not available on PATH, so the project could not be executed from this shell. The commands below are the exact steps to run it on a machine with Python installed.

---

## 4. Environment setup

From the project root:

```powershell
cd "D:\425 project\gnn-bert-music-context"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On Linux/macOS:

```bash
cd /path/to/gnn-bert-music-context
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 5. Install dependencies

Install the project package list from the repository root:

```powershell
python -m pip install -r requirements.txt
```

This includes:

- torch
- transformers
- librosa
- numpy
- scikit-learn
- pandas
- PyYAML
- torch-geometric
- networkx
- matplotlib
- jupyter
- seaborn
- soundfile
- tqdm

If PyG requires platform-specific backend packages for your OS/CUDA version, install the matching PyG extras as needed. The default CPU setup is usually sufficient for testing and prototyping.

---

## 6. Project configuration

The main global settings are stored in `config.yaml`.

Key settings include:

```yaml
global:
  sample_rate: 22050
  log_mel_bins: 128
  chroma_bins: 12
  segment_window_sec: 5.0
  segment_hop_sec: 2.5
  graph_similarity_threshold: 0.7
  bert_model_name: "distilbert-base-uncased"
```

This config drives:

- audio downsampling
- segment extraction
- FFT and mel spectrogram generation
- graph edge construction
- model architecture parameters

---

## 7. Data preparation

The repository expects a structured dataset layout under the `data/` directory.

### Recommended folder layout

```text
data/
├── audio/
│   ├── track_001.wav
│   ├── track_002.wav
│   └── ...
├── metadata.csv
├── captions.json
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### 7.1 FMA

- Download the Free Music Archive audio files or a subset.
- Place the audio files in `data/audio/`.
- Build metadata tables with track IDs, genres, and file paths.
- Use the metadata to map audio files to tags and labels.

### 7.2 MagnaTagATune

- Use the dataset as an optional future source for multi-label music tagging.
- A valid MagnaTagATune audio manifest is not yet established in this repository, so the current branch does not claim a real retrieval or audio-aligned classification benchmark from this source.
- Match audio files with tag labels only after a verified manifest is created.

### 7.3 DEAM

- Use DEAM for continuous valence and arousal regression targets.
- Align emotion labels with track IDs or segments.
- Normalize valence/arousal to a common scale before training.

### 7.4 MusicCaps

- Download the official MusicCaps CSV and timestamped YouTube clips with `src/download_musiccaps.py`.
- Run `src/preprocess_musiccaps.py` to create one graph and one caption record per verified local clip.
- The generated `data/splits/musiccaps/alignment_report.json` records the exact verified IDs.
- Run Task 4 with `--manifest-root data/splits/musiccaps`; do not use the FMA metadata manifests for the MusicCaps result.

### Example file schema for metadata

```csv
track_id,audio_path,genre,caption_id,valence,arousal
track_001,data/audio/track_001.wav,ambient,cap_001,0.34,0.61
```

### Example caption schema

```json
{
  "cap_001": "A warm, mellow instrumental track with soft synth pads and a calm emotional atmosphere."
}
```

---

## 8. Core workflow

### A. Audio preprocessing

The module `src/audio_features.py` handles:

- waveform loading
- normalization
- log-mel extraction
- chroma extraction
- MFCC features
- segment splitting

### B. Graph construction

The module `src/graph_builder.py` takes segment feature vectors and builds a PyG graph using:

- segment-to-node feature vectors
- cosine similarity edges
- temporal adjacency constraints
- similarity thresholding

### C. Text encoding

The module `src/bert_encoder.py` wraps a Hugging Face text encoder and outputs:

- text hidden states `H_text`
- pooled representation `t`

### D. GNN encoding

The module `src/gnn_model.py` builds graph embeddings using either GraphSAGE-style or GAT-style operations.

### E. Fusion modeling

The module `src/fusion_model.py` implements the task 3 fusion model:

- graph embedding `g`
- text hidden states `H_text`
- cross-attention fusion
- tag logits
- valence regression
- arousal regression

### F. Training engine

The script `src/train.py` coordinates the four task training loops.

### G. FMA preprocessing

The script `src/preprocess_fma.py` converts downloaded FMA MP3 files into PyG graph
samples, feature artifacts, and JSON split manifests:

```powershell
python src/preprocess_fma.py --limit 20
```

Use `--limit 0` to process every available track. If a label CSV is available,
pass it with `--metadata path/to/metadata.csv`. The generated files are written to
`data/processed/graphs`, `data/processed/audio_features`, and `data/splits`.

---

## 9. Running the project

### Synthesizing a quick demo

The notebook is the fastest way to test the full flow without real data.

```powershell
jupyter notebook notebooks/demo_context.ipynb
```

The notebook does the following:

- loads a real held-out FMA graph and metadata record
- creates a text prompt and tag list
- extracts segments and features
- builds a PyG graph
- runs inference through the fusion model
- prints predictions
- visualizes the graph adjacency structure

### Build real FMA samples

After extracting FMA-small, generate the assignment's minimum processed examples:

```powershell
python src/preprocess_fma.py --limit 20
```

Check the generated graph count:

```powershell
(Get-ChildItem data/processed/graphs -Filter *.pt).Count
Get-Content data/splits/train.json
```

### Task 1: BERT multi-label tag classification

```powershell
python src/train.py --task task1 --epochs 10 --batch-size 16 --lr 2e-5
```

### Task 2: GNN audio encoder

```powershell
python src/train.py --task task2 --epochs 15 --batch-size 16 --lr 1e-4
```

### Task 3: GNN-BERT fusion and emotion regression

```powershell
python src/train.py --task task3 --epochs 20 --batch-size 8 --lr 2e-5
```

### Task 4: Contrastive audio-text retrieval

```powershell
python src/train.py --task task4 --epochs 20 --batch-size 16 --lr 1e-4
```

### Evaluation

```powershell
python src/evaluate.py
```

The evaluator calculates:

- macro-F1 and micro-F1 for multi-label tags
- AUC-PR
- MAE for valence and arousal
- Recall@1 / Recall@5 / Recall@10 for retrieval

---

## 10. Validation and debugging

### Basic syntax validation

```powershell
python -m py_compile src/audio_features.py src/graph_builder.py
```

### Common issues

#### Python not found

If `python` is not recognized:

```powershell
where.exe python
```

Then install Python and add it to PATH, or use the full path to Python.

#### PyTorch/PyG installation issues

If PyG fails to install, install the correct wheel for your platform:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric
```

#### Missing backend extensions

For some platforms, PyG may require `torch-scatter`, `torch-sparse`, `torch-cluster`, or `torch-spline-conv`.

```powershell
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

---

## 11. Expected outputs

After training, the project typically writes artifacts into:

- `checkpoints/` for model weights
- `outputs/` for logs and evaluation summaries
- `data/` for processed metadata and splits

Typical outputs:

- per-epoch training losses
- validation metrics
- serialized model checkpoint files
- retrieval benchmarks
- tag predictions and emotion regression metrics

---

## 12. Recommended development workflow

1. Install Python and create the virtual environment.
2. Install repository dependencies.
3. Prepare dataset metadata and captions.
4. Run the notebook for a minimal end-to-end smoke test.
5. Move to Task 1 and Task 2 training for baseline models.
6. Run Task 3 for the multimodal fusion model.
7. Use Task 4 for retrieval evaluation.
8. Inspect outputs and tune hyperparameters in `config.yaml`.
9. Iterate on data cleaning and model architecture as needed.

---

## 13. Full project run summary

The intended workflow is:

```text
Python installed and on PATH
  -> create .venv
  -> install dependencies
  -> prepare datasets
  -> run notebook demo
  -> train Task 1
  -> train Task 2
  -> train Task 3
  -> train Task 4
  -> evaluate results
  -> review metrics and checkpoints
```

---

## 14. Notes

- This project is structured as a research/starter multimodal music understanding pipeline.
- It is designed for modular experimentation and extension.
- The real FMA notebook demo is the recommended first validation path; synthetic smoke data is available only through the explicit `--synthetic` training flag.
- The project is intentionally organized so that each major component can be swapped or improved independently.

---

## 15. Status of this environment

This shell currently does not have Python installed or on the PATH, so actual execution from this terminal is blocked by the local environment itself.

The commands above are the correct project execution steps for a properly configured Python environment.
