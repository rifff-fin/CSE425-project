# Current Project Status

Updated: 2026-09-08 (full FMA-small preprocessing and Task 1 epoch-F1 artifacts)

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
| Primary audio dataset | Complete for preprocessing; benchmark evidence mixed | FMA-small is downloaded/extracted and 7,994/8,000 tracks are processed into full graph/feature manifests. The controlled model demo remains 100 tracks; a one-epoch full-data GNN result is recorded. |
| Text/tag dataset | Partial | FMA metadata and DEAM targets are verified. MusicCaps now has 95 verified local audio-caption pairs after five unavailable source videos were skipped. MagnaTagATune still has metadata only. |
| Audio preprocessing | Complete for current sample | 22,050 Hz audio, mel/chroma/MFCC features, normalization, and fixed-window segmentation are implemented. |
| Segment graph construction | Complete | Temporal adjacency plus cosine-similarity edges are implemented with PyTorch Geometric. |
| Task 1 BERT classifier | Implemented with analysis artifacts | Three-epoch real FMA training history, loss curve, and five held-out prediction examples are in `results/`. |
| Task 2 GNN encoder | Implemented with controlled comparison | Three-epoch GNN and CNN runs use the same 80/10/10 FMA split; comparison is in `results/task2_gnn_cnn_comparison.json`. |
| Task 3 GNN-BERT fusion | Complete with documented dataset separation | FMA tag fusion/ablations/t-SNE, a real DEAM graph/text/emotion run, three FMA tag cases, and three MusicCaps caption/graph cases are recorded in `results/`. FMA and DEAM identifiers are not joined. |
| Task 4 contrastive retrieval | Real MusicCaps benchmark implemented on 95 verified pairs | Ten held-out MusicCaps clips, R@K evaluation, ten qualitative queries, zero-shot caption tags, and a five-listener rating sheet are generated. Human evaluation is complete with 150 supplied valid ratings. |
| Baselines | Implemented for required comparison | Leakage-safe majority and CNN mel-spectrogram baselines are implemented; the CNN/GNN controlled three-epoch comparison is in `results/task2_gnn_cnn_comparison.json`. |
| Evaluation and analysis | Implemented for current subsets | F1, AUC-PR, MAE, Recall@K, curves, ablations, t-SNE, case studies, and retrieval examples are generated. Results remain subset-scale evidence. |
| Submission package | Mostly complete | Code, plots, metrics, retrieval examples, evaluation sheet, and the LaTeX report source `report/final_report.tex` are present. A final PDF export and completed human ratings still require final presentation work. |

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
- `src/train.py`: task-specific training loops and PyG-aware batching; Task 1 records train/validation Macro-F1, Micro-F1, and AUC-PR at every epoch
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
- `src/evaluate_fma_multimodal.py`: evaluates real paired Task 3 tagging and Task 4 bidirectional retrieval on `test.json`
- `src/fma_paired_dataset.py`: loads aligned FMA graph/text/tag/emotion fields for multimodal tasks
- `src/aggregate_metrics.py`: combines Task 1/2/3/4 and baseline JSON outputs into `results/metrics.json`
- `src/align_musiccaps.py`: imports MusicCaps JSON/CSV captions and creates only ID-verified aligned split manifests
- `src/download_musiccaps.py`: downloads timestamped MusicCaps audio clips from the official IDs
- `src/preprocess_musiccaps.py`: creates graph/audio features and leakage-safe MusicCaps caption manifests
- `src/prepare_deam_targets.py`: reads `DEAM_Annotations.zip` or an extracted directory, normalizes DEAM song-level valence/arousal annotations, and reports cross-dataset overlap without fabricating matches
- `src/prepare_deam_dataset.py`: joins extracted DEAM audio, metadata, and normalized targets by exact numeric song ID and writes leakage-safe DEAM manifests

### Real data

- FMA-small archive downloaded to `data/raw/fma/fma_small.zip`
- FMA-small extracted to `data/raw/fma/extracted/fma_small/`
- 8,000 MP3 files verified after extraction
- Official FMA metadata archive downloaded to `data/raw/fma/metadata/`
- FMA metadata extracted, including `raw_tracks.csv`, `tracks.csv`, and `genres.csv`
- Official FMA genre hierarchy joined to the controlled 100-record manifests and the full 7,994-record manifests
- Each manifest now stores `genre_ids`, normalized `genre_tags`, multi-label `tags`, `text_tags`, and `text_context`
- The controlled FMA sample contains 26 labels; the full processed FMA-small manifest contains 114 labels
- 7,994 real FMA-small tracks processed successfully (6 source files were skipped because they could not be decoded or processed)
- 7,994 valid PyG graph files created in `data/processed_full/graphs/`
- 7,994 feature files created in `data/processed_full/audio_features/`
- Full split manifests created under `data/splits_full/`: 6,395 train, 799 validation, 800 test records
- Controlled manifest verification confirms 100 total records and 40 multi-label records; full manifests contain 7,994 records across 6,395/799/800 splits

### Runtime checks

- All Python source files compile successfully
- Demo notebook executes end to end with real FMA audio graph/text and the expanded Task 3 checkpoint
- Task 2 one-epoch smoke test passes
- Task 3 one-epoch smoke test passes
- Real paired Task 3 one-epoch training passes and writes `checkpoints/real_paired/task3_epoch_1.pt`
- Real FMA Task 3 held-out evaluator writes `results/expanded_fma_task3_metrics.json`; integrated DEAM Task 3 training/evaluation writes `results/task3_deam_integrated_metrics.json` with real valence/arousal MAE on verified DEAM graph/text/emotion pairs
- Task 3 caption/graph alignment cases are exported to `results/task3_caption_case_studies.json` from verified MusicCaps IDs
- Real FMA Task 2 one-epoch training passes using processed graphs and genre labels
- Real FMA Task 2 test evaluation passes and writes `results/fma_task2_metrics.json`
- FMA enrichment script passes syntax validation and updates the active manifests; the full run updated 7,994 records
- FMA dataset loader returns valid PyG graphs and multi-hot targets using the active manifest vocabulary
- FMA text dataset loader returns real metadata text and multi-hot targets
- Real FMA Task 1 one-epoch training passes and writes a Task 1 checkpoint
- Real FMA Task 1 test evaluation passes and writes `results/fma_task1_text_metrics.json`
- Task 1 test smoke metrics: Macro-F1 0.1429, Micro-F1 0.2857, AUC-PR 0.1769 on 2 held-out samples
- Task 1 epoch history now records train and validation Macro-F1/Micro-F1/AUC-PR for every epoch; the controlled three-epoch run is in `results/task1_f1_training_history.json`
- FMA majority baseline evaluation passes and writes `results/fma_majority_baseline_metrics.json`
- Majority baseline metrics: Macro-F1 0.0000, Micro-F1 0.0000, AUC-PR 0.1429 on 2 held-out samples
- CNN mel-spectrogram baseline one-epoch run passes and writes `results/fma_cnn_baseline_metrics.json`
- CNN baseline metrics: Macro-F1 0.0000, Micro-F1 0.0000, AUC-PR 0.1429 on 2 held-out samples
- CNN baseline uses the same 5-second, 22,050 Hz, 128-bin log-mel preprocessing family as the audio pipeline
- Text pipeline rerun completes successfully after metadata enrichment and writes updated manifests
- Latest Task 1 run: train loss 0.6943 and validation loss 0.6709 for one epoch
- Latest Task 1 test smoke metrics: Macro-F1 0.0952, Micro-F1 0.2222, AUC-PR 0.1833 on 2 held-out samples
- `prepare_deam_targets.py --annotations data/raw/deam/DEAM_Annotations.zip` passes directly from the downloaded ZIP and writes `data/processed/deam/deam_targets_from_zip.json`
- Direct ZIP processing verifies 1,802 unique DEAM song-level targets and reports 17 numeric ID coincidences with FMA; no coincidences are treated as valid audio alignment
- 95 real MusicCaps audio-caption pairs were downloaded/processed; 76 train, 9 validation, and 10 test records were created under `data/splits/musiccaps/`
- Final 20-epoch MusicCaps Task 4 metrics: caption-to-audio R@1 0.2000, R@5 0.6000, R@10 1.0000; audio-to-caption R@1 0.2000, R@5 0.8000, R@10 1.0000
- Task 4 training history and curve are in `results/musiccaps_task4_training_history.json` and `results/plots/musiccaps_task4_loss_curve.png`
- Ten qualitative retrieval examples are in `results/retrieval_examples/musiccaps_examples.json`
- A five-listener human evaluation sheet is in `results/retrieval_examples/human_evaluation.csv`; aggregation confirms 150 expected, 150 completed, and 0 missing or invalid ratings, so the supplied listener evidence is complete
- `notebooks/demo_context.ipynb` is now valid nbformat 4 JSON, executed successfully, and exported to `notebooks/demo_context.html`
- The proper report source is `report/final_report.tex`; PDF compilation remains pending because MiKTeX, TeX Live, and other local PDF toolchains are not installed

### DEAM audio acquisition (2026-09-05)

- Official `DEAM_audio.zip` downloaded successfully from the DEAM source at 1,343,203,527 bytes.
- ZIP integrity check passed with no bad entries; extraction produced 1,802 MP3 files.
- Exact normalized filename-stem matching joined all 1,802 audio files to all 1,802 DEAM emotion targets; no DEAM audio files were unmatched.
- Metadata matching also covered all 1,802 records.
- Leakage-safe DEAM manifests were created in `data/splits/deam/`: 1,441 train, 180 validation, and 181 test records.
- The DEAM pairing is valid within DEAM. The 17 coincidental numeric IDs between DEAM and FMA are still not treated as cross-dataset matches.

### Full FMA-small experiment (2026-09-08)

- All available decodable FMA-small audio was processed with `--limit 0 --skip-existing`; 7,994/8,000 tracks produced graph and feature artifacts.
- Metadata enrichment added official genre hierarchy labels and text contexts to all 7,994 records; the full vocabulary contains 114 labels.
- A one-epoch real-data Task 2 GNN run used the full manifests: 6,395 train, 799 validation, 800 test. Test Macro-F1 0.0277, Micro-F1 0.0395, AUC-PR 0.0188.
- The full-data CNN baseline was started but did not finish within the local execution window because it reloads audio on every sample; no incomplete CNN output is treated as a benchmark result.
- Task 1 three-epoch real-data F1 artifact is in `results/task1_f1_training_history.json`, with plot `results/plots/task1_f1_training_curve.png` and test metrics in `results/task1_f1_metrics.json` for the controlled 100-track split.

### Expanded FMA experiment (2026-09-05)

- Reprocessed 100 real FMA-small MP3 files successfully with no skipped tracks.
- Generated 100 graph files and 100 audio-feature files using the configured 22,050 Hz, 5-second window pipeline.
- Rebuilt leakage-safe manifests with 80 train, 10 validation, and 10 test records.
- Enriched all 100 records with official FMA genre hierarchy labels: 26 labels, including 40 multi-label records.
- Real Task 1 one-epoch run completed: train loss 0.6738, validation loss 0.6557. Test metrics are in `results/expanded_fma_task1_metrics.json`: Macro-F1 0.0641, Micro-F1 0.1200, AUC-PR 0.1094.
- Real Task 2 one-epoch run completed: train loss 1.0772, validation loss 1.0600. Test metrics are in `results/expanded_fma_task2_metrics.json`: Macro-F1 0.0429, Micro-F1 0.1111, AUC-PR 0.0983.
- Real FMA Task 3 one-epoch tag run completed; test metrics are in `results/expanded_fma_task3_metrics.json`. The integrated DEAM Task 3 one-epoch run reports Macro-F1 0.0165, Micro-F1 0.0357, AUC-PR 0.0803, valence MAE 4.2718, and arousal MAE 3.7425 in `results/task3_deam_integrated_metrics.json` on the 1-9 DEAM scale.
- Real Task 4 one-epoch run completed: train loss 2.1107, validation loss 1.4083. Test metrics are in `results/expanded_fma_task4_metrics.json`: caption-to-audio R@1 0.1000, R@5 0.5000, R@10 1.0000; audio-to-caption R@1 0.2000, R@5 0.5000, R@10 1.0000.

### Dataset audit (2026-09-05)

- All Python source files compile successfully after the audit.
- Official FMA repository and data-host links are reachable; the local FMA-small and metadata archives are present and the extracted metadata/audio fixtures remain available.
- Official MagnaTagATune endpoints are reachable and downloaded successfully: `data/raw/magnatagatune/clip_info_final.csv` (8,608,397 bytes) and `data/raw/magnatagatune/annotations_final.csv` (21,517,373 bytes).
- The downloaded MagnaTagATune tables parse as tab-separated files with 25,863 clips, 188 tag columns, and 89,395 positive tag assignments.
- MagnaTagATune audio is not downloaded. Its three-part audio archive is approximately 3 GB and is not required for the current FMA-based smoke pipeline until an audio source and matching manifest are selected.
- DEAM annotation, metadata, and official audio archives are present and parse successfully; the audio is paired to DEAM targets by exact song ID, but not to FMA.
- DEAM features are approximately 602 MB, Lakh clean MIDI is approximately 234 MB, and Million Song Dataset is approximately 1.8 GB for the subset or 280 GB for the full download. These are optional alternatives, not required to satisfy the minimum one-audio plus one-text/tag requirement already met by FMA plus metadata annotations.
- GTZAN, Million Song/tagtraum, Lakh MIDI, and EmoMusic files are not present locally. GTZAN's listed Kaggle source requires a separate download/account flow; EmoMusic currently redirects to a Google form.
- MusicCaps official metadata contains 5,521 records, but alignment against the 20 local FMA records remains 0 matched and 20 unmatched because MusicCaps IDs identify YouTube clips rather than FMA tracks.
- No dataset is treated as paired merely because numeric IDs happen to overlap; this prevents invalid DEAM/FMA leakage and fabricated multimodal examples.
- `results/metrics.json` was regenerated successfully; it reports the current leakage-safe 16/2/2 FMA smoke split with no missing metric inputs.
- The held-out FMA majority baseline and Task 1 text evaluator both pass after the audit. The only runtime messages are non-fatal PyTorch/Transformers future warnings.

## Current execution-ready state
The project is in a verified execution-ready state for the real-data path that is currently supported by the repository. Group 27 member information is recorded in `README.md` and `report/final_report.tex`.

- Real FMA graph/text pipeline runs end-to-end on the expanded 100-track sample and writes leakage-safe manifests under `data/splits/`
- The default `src/train.py` path uses real manifests unless `--synthetic` is explicitly passed
- The real notebook demo in `notebooks/demo_context.ipynb` loads the real FMA checkpoint and real graph/text data
- DEAM audio and emotion targets are downloaded and matched by exact numeric song ID within DEAM; they are intentionally not mixed with FMA because the source IDs are unrelated
- The aggregate summary under `results/metrics.json` is built only from the active real-data metric files, so it does not include stale 20-track outputs
- The repo remains honest about the fact that the MusicCaps result is a verified 95-pair subset and MagnaTagATune has metadata but no paired local audio manifest
- Submission packaging is blocked only by local PDF compilation; the supplied human evaluation is complete and the notebook conversion/execution blocker is resolved

## Remaining optional extensions

The following are still optional future research extensions and are not part of the current verified repository claim:

- Verified MusicCaps audio-caption pairing to a local audio source, if a matching playback source is later acquired
- MagnaTagATune audio download plus a valid audio manifest and retrieval benchmark for richer caption/tag alignment
- Larger multi-epoch experimental sweeps beyond the current one-epoch DEAM integration and controlled FMA analysis
- Report PDF generation and final submission packaging after five listeners provide real ratings and a LaTeX toolchain is installed

These remain future extensions until a valid real caption-to-audio source is acquired and matched with strict ID checks. The current repo should not be interpreted as having a completed MusicCaps or MagnaTagATune retrieval benchmark.

## Reproducible commands

From the project root:

```powershell
python src/preprocess_fma.py --limit 100 --metadata data/raw/fma/metadata/extracted/fma_metadata/tracks.csv
$files = Get-ChildItem src -Filter *.py | ForEach-Object { $_.FullName }
python -m py_compile $files
python src/train.py --task task1 --real-data --epochs 1 --batch-size 8 --checkpoint-dir checkpoints/expanded_fma
python src/evaluate_fma_text.py --checkpoint checkpoints/expanded_fma/task1_epoch_1.pt --output results/expanded_fma_task1_metrics.json
python src/evaluate_fma_baseline.py
python src/cnn_baseline.py --epochs 1 --batch-size 4
python src/enrich_fma_metadata.py
python src/train.py --task task2 --epochs 1 --batch-size 4
python src/train.py --task task3 --epochs 1 --batch-size 4
python src/train.py --task task4 --epochs 1 --batch-size 4
python src/train.py --task task2 --real-data --epochs 1 --batch-size 8 --checkpoint-dir checkpoints/expanded_fma
python src/train.py --task task3 --real-data --epochs 1 --batch-size 8 --checkpoint-dir checkpoints/expanded_fma
python src/train.py --task task4 --real-data --epochs 1 --batch-size 8 --checkpoint-dir checkpoints/expanded_fma
python src/evaluate_fma.py --checkpoint checkpoints/expanded_fma/task2_epoch_1.pt --output results/expanded_fma_task2_metrics.json
python src/evaluate_fma_multimodal.py --task task3 --checkpoint checkpoints/expanded_fma/task3_epoch_1.pt --output results/expanded_fma_task3_metrics.json
python src/evaluate_fma_multimodal.py --task task4 --checkpoint checkpoints/expanded_fma/task4_epoch_1.pt --output results/expanded_fma_task4_metrics.json
python src/aggregate_metrics.py --output results/metrics.json
python src/align_musiccaps.py --captions data/raw/musiccaps/musiccaps-public.csv --output-root data/splits/musiccaps_official
python src/prepare_deam_targets.py --annotations data/raw/deam/DEAM_Annotations.zip --output data/processed/deam/deam_targets_from_zip.json
python src/prepare_deam_dataset.py --audio-root data/raw/deam/audio --targets data/processed/deam/deam_targets_from_zip.json --metadata-root data/raw/deam/metadata --output-root data/splits/deam
python src/preprocess_deam.py --limit 100
python src/train.py --task task3 --manifest-root data/splits/deam_processed --epochs 1 --batch-size 8 --checkpoint-dir checkpoints/deam_real
python src/evaluate_fma_multimodal.py --task task3 --manifest-root data/splits/deam_processed --checkpoint checkpoints/deam_real/task3_epoch_1.pt --output results/deam_task3_metrics.json
```
