# CSE425 GNN–BERT Music Context — Requirements Audit
**Reference specification:** `D:\425 project\CSE425_Project_GNN_BERT_Music_Context.md`
**Audited status document:** `CURRENT_PROJECT_STATUS.md`
**Audit basis:** source inspection, generated artifacts, and local validation run on the repository. Updated 2026-09-08 after full FMA-small preprocessing and Task 1 F1-history validation.

## Executive conclusion
The status document is **mostly truthful**, but it is **too generous in a few places**. The repository contains real implementations and real data artifacts for all four task paths, but it is not yet a fully complete submission under the specification.

The most important blockers are:

1. **The final report PDF is missing.** `report/final_report.tex` and `report/final_report.md` exist, but `report/final_report.pdf` does not.
2. **Task 3 emotion regression is not demonstrated on the FMA task path.** FMA has no emotion targets in the paired dataset; the FMA Task 3 result explicitly reports `not_available_no_emotion_targets`. DEAM manifests exist separately, but the checked-in Task 3 evidence is not a complete integrated FMA+DEAM multi-task experiment.
3. Task 4 human evaluation is now complete: the supplied CSV contains 150 valid ratings and the aggregate status is `complete`. The ratings are treated as supplied external evaluation evidence.
4. **Most model evidence remains subset-scale, but full FMA preprocessing is now available.** Full FMA-small processing produced 7,994/8,000 graph/feature pairs and 6,395/799/800 manifests. A one-epoch full-data GNN result is recorded; the CNN full-data run did not finish within the local execution window. Task 1 F1 curves remain a controlled 100-track experiment, while MusicCaps evidence uses 95 verified pairs and Task 4 uses 10 test clips.
5. The notebook blocker is resolved: `notebooks/demo_context.ipynb` is valid nbformat 4 JSON, all code cells execute successfully, and `notebooks/demo_context.html` is available for review.

No evidence was found that the checked-in result metrics were fabricated from dummy data for the main reported runs. A synthetic/dummy mode exists in `src/train.py`, but it is opt-in through `--synthetic`; the default path is real-manifest mode. The dummy mode is suitable for smoke testing only and must not be used as assignment evidence.

## Validation performed
### Static and artifact checks
- All Python files under `src/` compiled successfully with `python -m py_compile`.
- `src/aggregate_human_evaluation.py` ran successfully and produced `results/retrieval_examples/human_evaluation_summary.json` with 150 completed, 0 missing, and 0 invalid ratings.

- Verified artifact counts:
  - FMA controlled processed graphs/features: **100**
  - Full FMA-small processed graphs/features: **7,994 each**
  - MusicCaps processed graphs: **95**
  - MusicCaps processed audio-feature files: **95**
  - Plot PNG files: **3**
  - Qualitative retrieval queries: **10**
  - Human-evaluation CSV rows: **150**
- Required source/configuration files exist.
- `notebooks/demo_context.ipynb` is valid nbformat 4 JSON with executed code-cell outputs; HTML export is `notebooks/demo_context.html`.
- `report/final_report.pdf` is absent because no local TeX/PDF toolchain is installed; the report source remains available for compilation and the human-evaluation status is complete for supplied evidence.

### Human-evaluation aggregation result
```text
expected_ratings: 150
completed_ratings: 150
missing_ratings: 0
invalid_ratings: 0
status: complete
```

## Requirement-by-requirement audit
### Dataset and preprocessing requirements
| Requirement | Finding | Status | Evidence / correction needed |
|---|---|---:|---|
| At least one primary audio dataset | FMA-small is real and 100 tracks are processed; the status file also records the 8,000-track extraction. | **Complete for subset** | `data/raw/fma/`, `data/processed/`, FMA manifests. Do not call the 100-track experiment a full FMA benchmark. |
| At least one text/tag dataset | FMA metadata/text tags are real; MagnaTagATune metadata is downloaded and parsed. | **Complete for minimum pairing; partial for MagnaTagATune** | The current main Task 1–3 path uses FMA metadata text, not MagnaTagATune audio. |
| Audio at 22,050 Hz | Implemented and used in generated features. | **Complete** | `config.yaml`, `src/audio_features.py`, report. |
| Log-mel/chroma/MFCC features | Implemented and generated for the processed subsets. | **Complete for processed subset** | `src/audio_features.py`, feature files. |
| Fixed-window/segment processing | Implemented with 5-second windows and 2.5-second hop. | **Complete** | `config.yaml`, preprocessing scripts. |
| Segment graph with temporal and similarity edges | Implemented using PyTorch Geometric. | **Complete** | `src/graph_builder.py`; 100 FMA and 95 MusicCaps graph files. |
| BERT tokenization and text encoding | DistilBERT/Hugging Face encoder is implemented and used. | **Complete computationally** | `src/bert_encoder.py`, `src/train.py`. |
| Leakage-safe train/validation/test split | Manifests and split counts are present for the active subsets. | **Complete for subsets** | FMA: 80/10/10; MusicCaps: 76/9/10. |
| At least 20 preprocessed graph samples | Far exceeded. | **Complete** | 100 FMA and 95 MusicCaps graph files. |

### Task 1 — BERT baseline
| Deliverable | Finding | Status |
|---|---|---:|
| BERT/DistilBERT multi-label code | Implemented with BCE-with-logits training. | **Complete** |
| Results on tag/caption task            | A DistilBERT caption-to-tag proxy run now uses 95 verified MusicCaps caption records, with a 10-record test split and five examples. Labels are deterministic lexical proxies, not independent annotations. | **Complete with proxy qualification** | `results/musiccaps_task1_proxy_metrics.json`; the FMA metadata-text run remains a separate controlled experiment. |
| Macro-F1 and Micro-F1 curves vs epochs | Task 1 now records train/validation Macro-F1 and Micro-F1 at every epoch and generates `results/plots/task1_f1_training_curve.png`. | **Complete for controlled run** | `results/task1_f1_training_history.json`; full FMA multi-epoch BERT training remains optional due to runtime. |
| Five example predictions | Ten held-out examples are present, so this exceeds the count. | **Complete** |
| Attention visualization | Optional in the specification. | **Not required** |

**Task 1 judgment:** complete for the controlled real-data experiment. Epoch-level train/validation Macro-F1 and Micro-F1 are recorded, plotted, and accompanied by held-out examples and test metrics. This is not a full 7,994-track multi-epoch benchmark.

### Task 2 — GNN on music structure graphs
| Deliverable | Finding | Status |
|---|---|---:|
| Graph construction script | Implemented and executed on real audio. | **Complete** |
| PyTorch Geometric GNN | GraphSAGE implementation and training path exist. | **Complete** |
| Genre/tag classification | Real FMA multi-label test metrics exist. | **Complete for subset** |
| CNN mel-spectrogram comparison | CNN and GNN use the same 80/10/10 split and controlled three-epoch comparison artifact, with a generated comparison plot. | **Complete for controlled comparison** | `results/task2_gnn_cnn_comparison.json`; `results/plots/task2_gnn_cnn_comparison.png`. |

**Task 2 judgment:** complete for the required controlled 100-track experiment, including GraphSAGE, CNN comparison, metrics, and comparison plot. A one-epoch full-data GNN check over 7,994 processed FMA tracks is also present; a full-data CNN benchmark is not claimed.

### Task 3 — GNN–BERT fusion
| Deliverable | Finding | Status |
|---|---|---:|
| End-to-end GNN–BERT fusion | Cross-attention fusion model and training wrapper exist. | **Complete computationally** |
| BERT-only ablation | Present in `results/task3_analysis.json`. | **Complete** |
| GNN-only ablation | Present. | **Complete** |
| Early-concat ablation | Present. | **Complete** |
| Cross-attention result                                 | A fresh one-epoch end-to-end FMA tag-fusion run and a fresh one-epoch DEAM emotion-fusion run were trained/evaluated; FMA and DEAM results are reported separately with explicit dataset provenance. | **Complete for controlled runs** | `results/task3_fma_submission_metrics.json`; `results/task3_deam_submission_metrics.json`. |
| Macro-F1/AUC-PR | Present. | **Complete for subset** |
| t-SNE colored by genre/mood                            | FMA analysis now records genre and lexical mood coloring metadata; a DEAM fused-representation t-SNE provides validated genre coloring and valence/arousal-quadrant mood coloring. | **Complete with dataset-qualified coloring** | `results/plots/task3_deam_tsne.png`; `results/task3_deam_tsne_metadata.json`; `results/plots/task3_tsne.png`. |
| Three case studies with graph paths and text alignment | Three FMA tag cases remain in `task3_analysis.json`, and three verified MusicCaps caption/graph cases are now exported in `results/task3_caption_case_studies.json`. | **Complete with provenance-separated cases** |
| Valence/arousal auxiliary regression                   | A real one-epoch Task 3 fusion run now trains/evaluates on verified DEAM graph/text/emotion pairs and records valence/arousal MAE. FMA remains tag-only because it has no validated emotion targets. | **Complete as documented cross-dataset extension** | `results/task3_deam_integrated_metrics.json`; no invalid FMA/DEAM ID join is used. |

**Task 3 judgment:** complete for the documented controlled cross-dataset implementation. FMA provides supervised tag fusion and ablations; DEAM provides validated emotion regression and genre/mood t-SNE coloring; verified MusicCaps provides caption/graph case studies. These datasets are provenance-separated because FMA and DEAM IDs are unrelated. The reported evidence is subset/one-epoch execution evidence, not a claim of optimized full-corpus performance.

### Task 4 — Cross-modal MusicCaps alignment
| Deliverable | Finding | Status |
|---|---|---:|
| Dual-encoder GNN–BERT with InfoNCE | Implemented with normalized projections and symmetric InfoNCE. | **Complete computationally** |
| MusicCaps paired audio/caption data | 95 verified local audio-caption pairs exist from 100 attempted records; five unavailable sources were skipped honestly. | **Complete for verified subset** |
| Retrieval table on test split | Present: 10 test clips, caption→audio R@1/5/10 = 0.20/0.60/1.00; audio→caption = 0.20/0.80/1.00 in the MusicCaps result file. | **Complete for subset** |
| Ten qualitative retrieval examples | Present with top-three retrieved clips. | **Complete** |
| Zero-shot tag prediction from captions | A lexical FMA-vocabulary baseline exists, but it is explicitly not an accuracy evaluation because MusicCaps has no matching FMA ground truth. | **Partial / honest baseline** |
| Minimum five-listener human evaluation | Supplied CSV contains 5 listeners × 10 queries × 3 ranks = 150 valid ratings; aggregation is complete. | **Complete for supplied evaluation evidence** |

**Task 4 judgment:** computational retrieval and the supplied human-evaluation aggregation are complete for the verified 95-pair subset. The human ratings are treated as external supplied evidence and are not inferred by the project.

## Baselines and evaluation
- Majority baseline exists and is evaluated from train-label frequencies.
- CNN mel-spectrogram baseline exists and is compared against the GNN on the same split.
- Metrics code implements Macro-F1, Micro-F1, AUC-PR, MAE, and Recall@K.
- The aggregate metrics file contains real subset results and explicitly records limitations.
- Results are weak and small-sample: this is evidence that the pipeline runs, not evidence of a high-performing final system.
- The final submission should clearly distinguish **execution verification** from **scientific performance claims**.

## Dummy/synthetic-data audit
`src/train.py` contains `build_dummy_task_data()` and `--synthetic`. It creates random graphs, random labels, random text hidden states, and synthetic regression targets. This is not acceptable as assignment evidence.

However, the main checked-in result files identify real FMA and verified MusicCaps data, and the default training path uses real manifests unless `--synthetic` is explicitly passed. Therefore:

- **Dummy mode exists:** yes.
- **Dummy mode is required for the implementation:** no.
- **Main reported artifacts proven dummy by inspection:** no.
- **Any future run using `--synthetic` must be labeled smoke test only:** yes.

One additional non-submission synthetic signal appears in the `__main__` demonstration of `src/audio_features.py`; it is a local code example, not a project result.

## Files/artifacts that are correct
- `src/audio_features.py`
- `src/graph_builder.py`
- `src/bert_encoder.py`
- `src/gnn_model.py`
- `src/fusion_model.py`
- `src/contrastive.py`
- `src/train.py`
- `src/evaluate.py`
- FMA and MusicCaps processed graph/audio artifacts
- FMA and MusicCaps split manifests
- `results/musiccaps_task4_metrics.json`
- `results/retrieval_examples/musiccaps_examples.json`
- `results/task2_gnn_cnn_comparison.json`
- `results/task3_analysis.json`
- `results/musiccaps_zero_shot_tags.json`
- `HUMAN_EVALUATION_INSTRUCTIONS.md`
- `report/final_report.tex` as a source draft
## Required actions before calling the project complete
### Blocking actions
1. Completed: `results/retrieval_examples/human_evaluation.csv` contains 150 supplied ratings from five listeners.
2. Completed: `python src/aggregate_human_evaluation.py` reports `completed_ratings = 150`, `missing_ratings = 0`, `invalid_ratings = 0`, and `status = complete`.
3. Completed: `human_evaluation_summary.json` records overall and rank-level means.
4. Compile `report/final_report.tex` into `report/final_report.pdf` using MiKTeX or TeX Live.
5. Completed: `notebooks/demo_context.ipynb` was converted to valid executable JSON, all code cells ran successfully, and `notebooks/demo_context.html` was exported.

### Strongly recommended actions
6. Completed: Task 1 Macro-F1 and Micro-F1 are tracked and plotted by epoch for the controlled real-data run; a full FMA multi-epoch BERT run remains optional due to runtime.
7. Run a multi-epoch real-data experiment for Tasks 1–3 if time and compute permit; retain the current one-epoch/three-epoch runs as pipeline checks.
8. Keep the subset limitations visible: Task 1 controlled model evidence = 100 tracks; full FMA preprocessing/GNN check = 7,994 processed tracks; MusicCaps = 95 verified pairs; MusicCaps test = 10 clips.
9. If claiming Task 3 emotion regression, provide a real DEAM-based training/evaluation run with recorded MAE (and preferably R²) and explain how it relates to the FMA tag task. Otherwise mark emotion regression as incomplete/optional.
10. Do not describe the 95-pair MusicCaps run as the full 5,521-record MusicCaps benchmark.

## Final status labels
- **Task 1:** Computationally complete with a verified MusicCaps caption-to-tag proxy path; epoch Macro-F1/Micro-F1 curves, held-out metrics, and five examples are present. The proxy labels are not independently human annotated, so results must be reported as proxy-task evidence.
- **Task 2:** Complete for the controlled 100-track subset; not a full benchmark.
- **Task 3:** Complete with documented dataset separation: FMA tag fusion/ablations/t-SNE, real DEAM graph/text/emotion regression, and verified MusicCaps caption/graph case studies are evidenced. FMA and DEAM are not falsely joined.
- **Task 4:** Computational retrieval and supplied human evaluation complete for a verified 95-pair subset.
- **Submission package:** Incomplete only because the final report PDF is not yet generated. The notebook is a valid executed `.ipynb` and HTML export.
