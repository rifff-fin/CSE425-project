# Tasks 1-3 Submission Checklist

This checklist limits the final review to the required computational work for Tasks 1, 2, and 3. Task 4 retrieval and human evaluation remain in the repository for the later submission pass.

## Task 1: BERT multi-label tagging

- [x] DistilBERT text encoder and multi-label classification head: `src/bert_encoder.py`, `src/train.py`
- [x] Real FMA metadata text/tag dataset: `src/fma_text_dataset.py`, `data/splits/train.json`, `data/splits/val.json`, `data/splits/test.json`
- [x] Training history: `results/task1_training_history.json`
- [x] Loss curve: `results/plots/task1_loss_curve.png`
- [x] Five held-out prediction examples: `results/task1_prediction_examples.json`
- [x] Held-out metrics: `results/expanded_fma_task1_metrics.json`

The assignment marks attention visualization as optional; no missing required Task 1 artifact remains.

## Task 2: GNN graph classification and baseline

- [x] Audio feature extraction and segment graphs: `src/audio_features.py`, `src/graph_builder.py`
- [x] GraphSAGE/GAT encoder: `src/gnn_model.py`
- [x] Real FMA graph dataset and processed graph fixtures: `src/fma_dataset.py`, `data/processed/graphs/`
- [x] GNN metrics: `results/task2_gnn_3epoch_metrics.json`
- [x] CNN mel-spectrogram baseline: `src/cnn_baseline.py`
- [x] CNN metrics and training history: `results/task2_cnn_3epoch_metrics.json`, `results/task2_cnn_history.json`
- [x] Controlled GNN/CNN comparison: `results/task2_gnn_cnn_comparison.json`
- [x] Expanded held-out metrics: `results/expanded_fma_task2_metrics.json`, `results/expanded_fma_cnn_baseline_metrics.json`

## Task 3: GNN-BERT fusion

- [x] Cross-attention fusion model: `src/fusion_model.py`
- [x] Real paired FMA dataset: `src/fma_paired_dataset.py`
- [x] End-to-end training and checkpoint: `src/train.py`, `checkpoints/real_paired/task3_epoch_1.pt`
- [x] Held-out fusion metrics: `results/expanded_fma_task3_metrics.json`
- [x] BERT-only, GNN-only, and early-concat ablations: `results/task3_analysis.json`
- [x] t-SNE figure: `results/plots/task3_tsne.png`
- [x] Three graph/text case studies: `results/task3_analysis.json`
- [x] DEAM within-dataset emotion evaluation: `results/deam_task3_metrics.json`

## Final validation

- [x] Python source compiles with `python -m py_compile src/*.py`
- [x] Task 3 case-study graph paths are repository-relative for portable submission
- [ ] Compile `report/final_report.tex` to PDF with MiKTeX or TeX Live
- [ ] Review the notebook and final report presentation

## Deliberately retained for the later Task 4 pass

Task 4 source, MusicCaps manifests, retrieval metrics, retrieval examples, checkpoints, and human-evaluation files are retained. They are not required to validate Tasks 1-3 and should be reviewed separately after listener ratings are collected.
