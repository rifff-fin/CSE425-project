# GNN-BERT Music Context: Experimental Report Draft

## Scope

This report records the reproducible real-data experiments for the CSE425 GNN-BERT music-context project. FMA-small provides the audio graphs and genre/tag targets. MusicCaps provides natural-language captions and timestamped audio clips for Task 4. DEAM is evaluated separately for within-dataset emotion regression.

## Requirement-by-requirement status

| Requirement | Status | Evidence | Remaining action |
|---|---|---|---|
| Task 1: BERT multi-label tagging | Complete with caption-proxy qualification | `results/task1_musiccaps_proxy_training_history.json`, `results/plots/task1_musiccaps_proxy_f1_curve.png`, `results/musiccaps_task1_proxy_metrics.json` | Five examples are present. Labels are deterministic lexical caption proxies, not independent human annotations. |
| Task 2: GNN graph classification | Complete for controlled and one-epoch full-data checks | `results/task2_gnn_3epoch_metrics.json`, `results/full_fma_task2_metrics.json` | Controlled result uses 100 tracks; full FMA result uses 7,994 processed tracks and one epoch. |
| Task 2: CNN comparison | Complete for controlled comparison | `results/task2_cnn_3epoch_metrics.json`, `results/task2_gnn_cnn_comparison.json`, `results/plots/task2_gnn_cnn_comparison.png` | Same 80/10/10 split and three epochs; full-data CNN was not completed. |
| Task 3: GNN-BERT fusion | Complete for controlled cross-dataset implementation | `src/train.py`, `results/task3_fma_submission_metrics.json`, `results/task3_deam_submission_metrics.json`, `checkpoints/task3_deam_submission/task3_epoch_1.pt` | FMA tag fusion and verified DEAM graph/text/emotion training are both evaluated; FMA and DEAM are not falsely joined by ID. |
| Task 3: ablations | Done computationally | `results/task3_analysis.json` | None for the listed BERT-only/GNN-only/early-concat probes |
| Task 3: t-SNE | Complete with dataset-qualified genre/mood coloring | `results/plots/task3_tsne.png`, `results/plots/task3_deam_tsne.png`, `results/task3_deam_tsne_metadata.json` | FMA plot uses metadata/lexical mood terms; DEAM plot uses validated valence/arousal mood quadrants. |
| Task 3: three case studies | Complete across tag and caption analysis | `results/task3_analysis.json`, `results/task3_caption_case_studies.json` | Three FMA graph/tag cases and three verified MusicCaps graph/caption cases are provided; datasets remain provenance-separated. |
| Task 4: real MusicCaps pairing | Done on verified subset | `data/splits/musiccaps/`, 95 pairs: 76 train, 9 validation, 10 test | Full 5,521-clip source benchmark is not claimed |
| Task 4: contrastive retrieval | Done computationally | `results/musiccaps_task4_metrics.json`, 20-epoch checkpoint and history | None for the local benchmark |
| Task 4: ten qualitative examples | Done computationally | `results/retrieval_examples/musiccaps_examples.json` | None |
| Task 4: zero-shot tags | Done as an honest baseline | `results/musiccaps_zero_shot_tags.json` | No MusicCaps ground-truth FMA tags exist, so accuracy is not claimed |
| Task 4: human evaluation | Prepared, not completed | `results/retrieval_examples/human_evaluation.csv`, `results/retrieval_examples/human_evaluation_summary.json`, `HUMAN_EVALUATION_INSTRUCTIONS.md` | Aggregation confirms 0/150 completed; five real listeners must enter ratings |
| Final report | Source complete, PDF pending | `report/final_report.tex` | Compile PDF with MiKTeX/TeX Live; notebook is now valid executed JSON |
| Demo notebook | Valid and executed | `notebooks/demo_context.ipynb`, `notebooks/demo_context.html` | Review exported HTML before submission |

## Implemented tasks
- Task 1: DistilBERT multi-label classification on verified MusicCaps captions using a fixed 50-label deterministic caption-to-tag proxy ontology (41 labels observed); the older FMA metadata-text experiment is retained separately.
- Task 2: GraphSAGE classification on FMA segment graphs, compared with a CNN mel-spectrogram baseline.
- Task 3: GNN-BERT cross-attention fusion, with BERT-only, GNN-only, and early-concat analysis probes. FMA tag fusion and a real DEAM graph/text/emotion extension are evaluated without invalid cross-dataset joining.
- Task 4: GNN-BERT contrastive retrieval on 95 verified MusicCaps audio-caption pairs.

## Evidence files

- Task 1 MusicCaps proxy loss/F1 curve: `results/plots/task1_musiccaps_proxy_f1_curve.png`
- Task 1 MusicCaps proxy epoch history: `results/task1_musiccaps_proxy_training_history.json`
- Task 1 MusicCaps proxy metrics and five examples: `results/musiccaps_task1_proxy_metrics.json`
- Task 1 original FMA metadata-text examples: `results/task1_prediction_examples.json`
- Task 2 comparison: `results/task2_gnn_cnn_comparison.json`
- Task 2 comparison plot: `results/plots/task2_gnn_cnn_comparison.png`
- Task 2 full-data GNN metrics: `results/full_fma_task2_metrics.json`
- Task 3 analysis: `results/task3_analysis.json`
- Task 3 caption/graph cases: `results/task3_caption_case_studies.json`
- Task 3 integrated DEAM metrics: `results/task3_deam_integrated_metrics.json`
- Task 3 FMA t-SNE: `results/plots/task3_tsne.png`
- Task 3 DEAM genre/mood t-SNE: `results/plots/task3_deam_tsne.png`
- Task 3 DEAM t-SNE metadata: `results/task3_deam_tsne_metadata.json`
- Task 4 metrics: `results/musiccaps_task4_metrics.json`
- Task 4 qualitative retrievals: `results/retrieval_examples/musiccaps_examples.json`
- Task 4 zero-shot tags: `results/musiccaps_zero_shot_tags.json`
- Human evaluation form: `results/retrieval_examples/human_evaluation.csv`
- Human evaluation aggregation: `results/retrieval_examples/human_evaluation_summary.json` (currently incomplete: 0/150 ratings)
- Executed notebook: `notebooks/demo_context.ipynb`

## Results

The expanded MusicCaps test split contains 10 clips. After 20 epochs, caption-to-audio retrieval is R@1 0.20, R@5 0.60, and R@10 1.00. Audio-to-caption retrieval is R@1 0.20, R@5 0.80, and R@10 1.00. These are results on the verified 95-pair local subset, not a claim over all 5,521 MusicCaps records.

The Task 1 MusicCaps caption-proxy history records epoch-level train/validation Macro-F1 and Micro-F1; the proxy evaluation reports Macro-F1 0.0662, Micro-F1 0.2167, and AUC-PR 0.1800 on 10 held-out caption records. Because labels are derived lexically from captions, these numbers are proxy-task execution evidence rather than independent semantic accuracy. The Task 2 comparison uses the same 80/10/10 split and three epochs for both models. The machine-readable comparison file contains the exact Macro-F1, Micro-F1, and AUC-PR values. Task 3 fusion/ablation artifacts are present for the FMA tag task, and the same fusion model has a real integrated DEAM graph/text/emotion run with valence/arousal MAE. The FMA and DEAM datasets remain separate because their identifiers are unrelated. Caption/graph alignment cases are exported from verified MusicCaps pairs.

## Limitations

Five of the first 100 YouTube sources were unavailable and were excluded without fabricating audio. MagnaTagATune audio is not included. The current human-evaluation aggregation reports 150 missing ratings; five listeners must complete the sheet before submission. The report PDF must be exported from this draft after the human scores and final figures are reviewed.

## Reproduction

Run from the project root:

```powershell
python src/download_musiccaps.py --limit 100
python src/preprocess_musiccaps.py --limit 100
python src/train.py --task task4 --manifest-root data/splits/musiccaps --epochs 20 --batch-size 16 --lr 1e-4 --checkpoint-dir checkpoints/musiccaps_task4_95_epoch20 --history-output results/musiccaps_task4_training_history.json
python src/evaluate_fma_multimodal.py --task task4 --manifest-root data/splits/musiccaps --checkpoint checkpoints/musiccaps_task4_95_epoch20/task4_epoch_20.pt --output results/musiccaps_task4_metrics.json
python src/plot_training_history.py results/musiccaps_task4_training_history.json --output results/plots/musiccaps_task4_loss_curve.png
python src/export_retrieval_examples.py --manifest-root data/splits/musiccaps --checkpoint checkpoints/musiccaps_task4_95_epoch20/task4_epoch_20.pt --output results/retrieval_examples/musiccaps_examples.json --limit 10
```