# Current Project Status

Updated: 2026-09-03 (Task 2 CNN baseline milestone)

## Project goal

Build a multimodal music-context system for the CSE425 assignment by combining:

- Librosa audio features
- PyTorch Geometric music segment graphs
- DistilBERT text representations
- GNN-BERT cross-attention fusion
- Multi-label tagging, emotion regression, and audio-text retrieval
## Requirement alignment
Compared with `project requirement.md`:

| Requirement area | Status | Evidence / gap |
|---|---|---|
| Primary audio dataset | Partial | FMA-small is downloaded, extracted, and 20 tracks are processed. Requirement recommends FMA-small/medium or another primary audio dataset. |
| Text/tag dataset           | Partial                              | FMA metadata now provides real artist, album, track, genre, and optional raw-tag text context. MagnaTagATune, MusicCaps, and lyrics are still not connected.                 |
| Audio preprocessing | Complete for current sample | 22,050 Hz audio, mel/chroma/MFCC features, normalization, and fixed-window segmentation are implemented. |
| Segment graph construction | Complete | Temporal adjacency plus cosine-similarity edges are implemented with PyTorch Geometric. |
| Task 1 BERT classifier | Partial | DistilBERT trains on real FMA metadata text context, and a held-out `test.json` evaluator now reports Macro-F1, Micro-F1, and AUC-PR; larger text/tag data and epoch curves are still required. |
| Task 2 GNN encoder | Partial | Real FMA graph classification works and a CNN mel-spectrogram baseline now runs; comparison is limited to the 20-track smoke dataset. |
| Task 3 GNN-BERT fusion | Code complete; experiment incomplete | Cross-attention and emotion heads exist, but real paired text, ablations, t-SNE, and case studies are missing. |
| Task 4 contrastive retrieval | Code complete; data incomplete | InfoNCE dual encoder exists, but MusicCaps alignment, held-out retrieval, and qualitative examples are missing. |
| Baselines | Partial | Leakage-safe majority and CNN mel-spectrogram baselines are implemented and evaluated; comparison remains limited to the 20-track smoke dataset. |
| Evaluation and analysis | Partial | F1, AUC-PR, MAE, and Recall@K utilities exist, but current real metrics use only a 2-sample test split. |
| Submission package | Partial | 20 graph samples and demo notebook exist; plots, retrieval examples, report PDF, and complete evaluation tables remain. |

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
- `src/fma_text_dataset.py`: manifest-backed FMA text-context dataset for Task 1
- `src/evaluate_fma_text.py`: evaluates the real FMA Task 1 checkpoint on the test split
- `src/evaluate_fma_baseline.py`: evaluates the train-frequency majority baseline on the FMA test split
- `src/cnn_baseline.py`: trains and evaluates the required CNN mel-spectrogram baseline for Task 2
- `.gitignore`: keeps source, manifests, processed fixtures, plots, and JSON metric summaries while ignoring local datasets and training artifacts
- `src/enrich_fma_metadata.py`: joins official FMA genre hierarchy and multi-label targets to manifests
- `src/evaluate_fma.py`: evaluates real FMA Task 2 checkpoints on the test split

### Real data

- FMA-small archive downloaded to `data/raw/fma/fma_small.zip`
- FMA-small extracted to `data/raw/fma/extracted/fma_small/`
- 8,000 MP3 files verified after extraction
- Official FMA metadata archive downloaded to `data/raw/fma/metadata/`
- FMA metadata extracted, including `raw_tracks.csv`, `tracks.csv`, and `genres.csv`
- Official FMA genre hierarchy joined to all 20 processed manifest records
- Each manifest now stores `genre_ids`, normalized `genre_tags`, multi-label `tags`, `text_tags`, and `text_context`
- Current FMA vocabulary contains 7 labels across the 20-track sample
- 20 real FMA tracks processed successfully
- 20 valid PyG graph files created in `data/processed/graphs/`
- 20 feature files created in `data/processed/audio_features/`
- Split manifests created: 16 train, 2 validation, 2 test records
- Manifest verification confirms 20 total records and 2 multi-label records

### Runtime checks

- All Python source files compile successfully
- Demo notebook executes end to end with synthetic audio
- Task 2 one-epoch smoke test passes
- Task 3 one-epoch smoke test passes
- Task 4 one-epoch smoke test passes
- Real FMA Task 2 one-epoch training passes using processed graphs and genre labels
- Real FMA Task 2 test evaluation passes and writes `results/fma_task2_metrics.json`
- FMA enrichment script passes syntax validation and updates all 20 records successfully
- FMA dataset loader returns valid PyG graphs and multi-hot targets using the expanded 7-label vocabulary
- FMA text dataset loader returns real metadata text and multi-hot targets
- Real FMA Task 1 one-epoch training passes and writes a Task 1 checkpoint
- Real FMA Task 1 test evaluation passes and writes `results/fma_task1_text_metrics.json`
- Task 1 test smoke metrics: Macro-F1 0.1429, Micro-F1 0.2857, AUC-PR 0.1769 on 2 held-out samples
- FMA majority baseline evaluation passes and writes `results/fma_majority_baseline_metrics.json`
- Majority baseline metrics: Macro-F1 0.0000, Micro-F1 0.0000, AUC-PR 0.1429 on 2 held-out samples
- CNN mel-spectrogram baseline one-epoch run passes and writes `results/fma_cnn_baseline_metrics.json`
- CNN baseline metrics: Macro-F1 0.0000, Micro-F1 0.0000, AUC-PR 0.1429 on 2 held-out samples
- CNN baseline uses the same 5-second, 22,050 Hz, 128-bin log-mel preprocessing family as the audio pipeline
- Text pipeline rerun completes successfully after metadata enrichment and writes updated manifests
- Latest Task 1 run: train loss 0.6943 and validation loss 0.6709 for one epoch
- Latest Task 1 test smoke metrics: Macro-F1 0.0952, Micro-F1 0.2222, AUC-PR 0.1833 on 2 held-out samples

## Current limitations

The FMA split records now have official hierarchical genre labels as multi-label targets and deterministic metadata text contexts. The current 20-track sample contains 7 labels, with 2 records receiving more than one label. Rich external user tags, captions, and emotion targets are still not connected.

Tasks 3 and 4 still use `build_dummy_task_data()` by default. Tasks 1 and 2 can now load the generated JSON manifests with `--real-data`; Task 1 uses metadata text and Task 2 uses real graph files.

No real caption or emotion source is connected yet:

- MagnaTagATune is not downloaded for richer user-generated multi-label tags; current FMA labels are genre-hierarchy based
- Current FMA text context is metadata-derived, not natural-language captions or lyrics
- DEAM is not downloaded for valence/arousal labels
- MusicCaps captions are not downloaded or aligned

The current metrics use a 2-sample test smoke split and are execution checks only, not research results. Task 1 metrics can vary between one-epoch runs because the training loop is not yet being used for a full controlled experiment.

## Required work remaining

### Priority 1: satisfy dataset requirements
- Connect a dedicated real text/tag source, preferably MagnaTagATune or MusicCaps
- Align richer real text/tags or captions with audio tracks/graphs
- Add richer user-generated multi-label annotations and freeze a consistent vocabulary
- Match FMA track IDs to the extracted MP3 files at larger scale
- Rewrite manifests with the larger real-data label set
### Priority 2: complete required baselines and real task pipelines
- Connect real data to Task 1, Task 3, and Task 4 instead of `build_dummy_task_data()`
- Compare the CNN mel-spectrogram baseline fairly against the GNN using a larger, controlled split and matched training settings
- Load `test.json` for every real-data evaluation path
- Add real text/caption loading and target tensors for tags and emotion values
- Add BERT-only, GNN-only, early-concat, and cross-attention ablation runs

### Priority 3: evaluation and analysis
- Expand the real FMA split before reporting test metrics
- Add test-set evaluation after Tasks 2, 3, and 4
- Save `results/metrics.json`
- Generate Macro-F1, Micro-F1, AUC-PR, and MAE results
- Add training curves, t-SNE visualizations, graph-path/caption case studies, and retrieval examples
- Compare all models against at least two baselines
- Add the Task 4 human evaluation with at least 5 listeners when retrieval data is available
### Priority 4: reproducible reporting
- Keep all reported metrics tied to leakage-safe held-out splits
- Save evaluation tables with dataset size, split definition, and model configuration
- Generate Macro-F1, Micro-F1, AUC-PR, MAE/R², and Recall@K results where applicable
- Document limitations caused by the current small FMA sample and missing text/emotion data
### Priority 5: submission package

- Prepare at least 20 processed graph samples
- Add plots and qualitative examples
- Write the 6-10 page final report
- Export `report/final_report.pdf`

## Next small step
Connect a dedicated paired text/tag source such as MagnaTagATune or MusicCaps, then connect those paired records to Tasks 3 and 4. Task 1 now has a complete small metadata-based train/validation/test smoke path, but its 2-sample test metrics are not research results.

## Reproducible commands

From the project root:

```powershell
python src/preprocess_fma.py --limit 20
$files = Get-ChildItem src -Filter *.py | ForEach-Object { $_.FullName }
python -m py_compile $files
python src/train.py --task task1 --real-data --epochs 1 --batch-size 4
python src/evaluate_fma_text.py --checkpoint checkpoints/task1_epoch_1.pt
python src/evaluate_fma_baseline.py
python src/cnn_baseline.py --epochs 1 --batch-size 4
python src/enrich_fma_metadata.py
python src/train.py --task task2 --epochs 1 --batch-size 4
python src/train.py --task task3 --epochs 1 --batch-size 4
python src/train.py --task task4 --epochs 1 --batch-size 4
python src/train.py --task task2 --real-data --epochs 1 --batch-size 4
```
