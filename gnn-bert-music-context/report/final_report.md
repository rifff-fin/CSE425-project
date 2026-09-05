# GNN-BERT Music Context: Experimental Report Draft

## Scope

This report records the reproducible real-data experiments for the CSE425 GNN-BERT music-context project. FMA-small provides the audio graphs and genre/tag targets. MusicCaps provides natural-language captions and timestamped audio clips for Task 4. DEAM is evaluated separately for within-dataset emotion regression.

## Requirement-by-requirement status

| Requirement | Status | Evidence | Remaining action |
|---|---|---|---|
| Task 1: BERT multi-label tagging | Done computationally | `results/task1_training_history.json`, `results/plots/task1_loss_curve.png`, `results/task1_prediction_examples.json` | Optional attention visualization; not required for the core deliverable |
| Task 2: GNN graph classification | Done computationally | `results/task2_gnn_3epoch_metrics.json` | Larger FMA benchmark is optional; current controlled result uses 100 tracks |
| Task 2: CNN comparison | Done computationally | `results/task2_cnn_3epoch_metrics.json`, `results/task2_gnn_cnn_comparison.json` | None for the required comparison |
| Task 3: GNN-BERT fusion | Done computationally | `src/train.py`, `checkpoints/real_paired/task3_epoch_1.pt` | Larger multi-epoch fusion sweep is optional |
| Task 3: ablations | Done computationally | `results/task3_analysis.json` | None for the listed BERT-only/GNN-only/early-concat probes |
| Task 3: t-SNE | Done computationally | `results/plots/task3_tsne.png` | None |
| Task 3: three case studies | Done computationally | `results/task3_analysis.json` | None; graph-edge previews are included |
| Task 4: real MusicCaps pairing | Done on verified subset | `data/splits/musiccaps/`, 95 pairs: 76 train, 9 validation, 10 test | Full 5,521-clip source benchmark is not claimed |
| Task 4: contrastive retrieval | Done computationally | `results/musiccaps_task4_metrics.json`, 20-epoch checkpoint and history | None for the local benchmark |
| Task 4: ten qualitative examples | Done computationally | `results/retrieval_examples/musiccaps_examples.json` | None |
| Task 4: zero-shot tags | Done as an honest baseline | `results/musiccaps_zero_shot_tags.json` | No MusicCaps ground-truth FMA tags exist, so accuracy is not claimed |
| Task 4: human evaluation | Prepared, not completed | `results/retrieval_examples/human_evaluation.csv`, `HUMAN_EVALUATION_INSTRUCTIONS.md` | Five real listeners must enter 150 ratings |
| Final report | Source complete | `report/final_report.tex` | Compile PDF with MiKTeX/TeX Live |
| Demo notebook | Existing real FMA demo | `notebooks/demo_context.ipynb` | Run and review before submission |

## Implemented tasks

- Task 1: DistilBERT multi-label classification on FMA metadata text.
- Task 2: GraphSAGE classification on FMA segment graphs, compared with a CNN mel-spectrogram baseline.
- Task 3: GNN-BERT cross-attention fusion, with BERT-only, GNN-only, and early-concat analysis probes.
- Task 4: GNN-BERT contrastive retrieval on 95 verified MusicCaps audio-caption pairs.

## Evidence files

- Task 1 curve: `results/plots/task1_loss_curve.png`
- Task 1 examples: `results/task1_prediction_examples.json`
- Task 2 comparison: `results/task2_gnn_cnn_comparison.json`
- Task 3 analysis: `results/task3_analysis.json`
- Task 3 t-SNE: `results/plots/task3_tsne.png`
- Task 4 metrics: `results/musiccaps_task4_metrics.json`
- Task 4 qualitative retrievals: `results/retrieval_examples/musiccaps_examples.json`
- Task 4 zero-shot tags: `results/musiccaps_zero_shot_tags.json`
- Human evaluation form: `results/retrieval_examples/human_evaluation.csv`

## Results

The expanded MusicCaps test split contains 10 clips. After 20 epochs, caption-to-audio retrieval is R@1 0.20, R@5 0.60, and R@10 1.00. Audio-to-caption retrieval is R@1 0.20, R@5 0.80, and R@10 1.00. These are results on the verified 95-pair local subset, not a claim over all 5,521 MusicCaps records.

The FMA Task 2 comparison uses the same 80/10/10 split and three epochs for both models. The machine-readable comparison file contains the exact Macro-F1, Micro-F1, and AUC-PR values.

## Limitations

Five of the first 100 YouTube sources were unavailable and were excluded without fabricating audio. MagnaTagATune audio is not included. Human evaluation requires five listeners to fill the rating sheet. The report PDF must be exported from this draft after the human scores and final figures are reviewed.

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