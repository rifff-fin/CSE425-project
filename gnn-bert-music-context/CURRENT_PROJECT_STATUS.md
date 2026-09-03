# Current Project Status

Updated: 2026-09-03

## Project goal

Build a multimodal music-context system for the CSE425 assignment by combining:

- Librosa audio features
- PyTorch Geometric music segment graphs
- DistilBERT text representations
- GNN-BERT cross-attention fusion
- Multi-label tagging, emotion regression, and audio-text retrieval

## Completed and verified

### Environment

- Python 3.12 installed
- Project dependencies installed from `requirements.txt`
- PyTorch and PyTorch Geometric import successfully

### Repository implementation

- `src/audio_features.py`: audio loading, resampling, normalization, mel/chroma/MFCC extraction, segmentation
- `src/graph_builder.py`: temporal and cosine-similarity graph construction
- `src/bert_encoder.py`: Hugging Face text encoder
- `src/gnn_model.py`: GraphSAGE/GAT graph encoder
- `src/fusion_model.py`: cross-attention tag and emotion model
- `src/contrastive.py`: contrastive audio-text model
- `src/train.py`: task-specific training loops and PyG-aware batching
- `src/evaluate.py`: tagging, regression, and retrieval metrics
- `src/preprocess_fma.py`: real FMA audio preprocessing command
- `src/fma_dataset.py`: manifest-backed FMA graph dataset for Task 2
- `src/enrich_fma_metadata.py`: joins official FMA genre labels to manifests
- `src/evaluate_fma.py`: evaluates real FMA Task 2 checkpoints on the test split

### Real data

- FMA-small archive downloaded to `data/raw/fma/fma_small.zip`
- FMA-small extracted to `data/raw/fma/extracted/fma_small/`
- 8,000 MP3 files verified after extraction
- Official FMA metadata archive downloaded to `data/raw/fma/metadata/`
- FMA metadata extracted, including `raw_tracks.csv`, `tracks.csv`, and `genres.csv`
- Official FMA `genre_top` labels joined to all 20 processed manifest records
- 20 real FMA tracks processed successfully
- 20 valid PyG graph files created in `data/processed/graphs/`
- 20 feature files created in `data/processed/audio_features/`
- Split manifests created: 16 train, 2 validation, 2 test records

### Runtime checks

- All Python source files compile successfully
- Demo notebook executes end to end with synthetic audio
- Task 2 one-epoch smoke test passes
- Task 3 one-epoch smoke test passes
- Task 4 one-epoch smoke test passes
- Real FMA Task 2 one-epoch training passes using processed graphs and genre labels
- Real FMA Task 2 test evaluation passes and writes `results/fma_task2_metrics.json`

## Current limitations

The FMA split records now have official top-level genre labels. Rich multi-label tags, captions, and emotion targets are still not connected.

Tasks 1, 3, and 4 still use `build_dummy_task_data()` by default. Task 2 can now load the generated JSON manifests and real graph files with `--real-data`.

No real caption or emotion source is connected yet:

- MagnaTagATune is not downloaded for multi-label tags
- DEAM is not downloaded for valence/arousal labels
- MusicCaps captions are not downloaded or aligned

The current metrics use a 2-sample test smoke split and are execution checks only, not research results.

## Required work remaining

### Priority 1: metadata and labels

- Add richer multi-label annotations from MagnaTagATune or FMA tag fields
- Match FMA track IDs to the extracted MP3 files
- Create a consistent tag vocabulary
- Rewrite manifests with real labels

### Priority 2: real training loader

- Load `test.json` during evaluation
- Add real text/caption loading
- Add target tensors for tags and emotion values
- Connect real datasets to Tasks 1-4

### Priority 3: evaluation and analysis

- Expand the real FMA split before reporting test metrics
- Add test-set evaluation after each task
- Save `results/metrics.json`
- Generate Macro-F1, Micro-F1, AUC-PR, and MAE results
- Add baseline comparison, ablation experiments, t-SNE, and retrieval examples

### Priority 4: submission package

- Prepare at least 20 processed graph samples
- Add plots and qualitative examples
- Write the 6-10 page final report
- Export `report/final_report.pdf`

## Next small step

Add test-set evaluation for the real FMA Task 2 path before integrating the larger MagnaTagATune, DEAM, or MusicCaps sources.

## Reproducible commands

From the project root:

```powershell
python src/preprocess_fma.py --limit 20
python -m py_compile src/*.py
python src/train.py --task task2 --epochs 1 --batch-size 4
python src/train.py --task task3 --epochs 1 --batch-size 4
python src/train.py --task task4 --epochs 1 --batch-size 4
python src/train.py --task task2 --real-data --epochs 1 --batch-size 4
```
