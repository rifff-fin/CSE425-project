# Video Demonstration and Submission Guide
## Project documentation and submission location
The repository-specific commands and current implementation status are in [`README.md`](README.md). The written-report checklist is in [`report/final_report.md`](report/final_report.md). The video should show the implemented source files, prepared evaluation outputs, final report, and demo notebook. Submit the project ZIP through the LMS upload field separately from the video; do not spend recording time creating or verifying the ZIP, and do not upload it to YouTube.

## What the video must demonstrate
Record the screen and microphone. Show the terminal commands, important project files, real outputs, and at least one audio result. Do not claim results that are not displayed or recorded.

### Suggested 8–12 minute recording
1. **Introduction (30 seconds)**
   - Show the project name, group name, and GitHub repository URL.
   - Say which tasks are demonstrated and identify any optional task.
2. **Project structure (45 seconds)**
   - Show `README.md`, `requirements.txt`, `config.yaml`, `src/`, `notebooks/`, `results/`, and `report/`.
3. **Environment and installation (1 minute)**
   - Show the commands below and confirm that dependencies are installed.
4. **Audio preprocessing and graph creation (1–2 minutes)**
   - Show an audio file being converted into features and a graph.
   - Display the graph or notebook visualization, including nodes and edges.
5. **Model demonstration (2–3 minutes)**
   - Run the demo notebook or a prepared inference command.
   - Explain the BERT text representation, GNN audio representation, fusion, and prediction.
6. **Results and evaluation (1–2 minutes)**
   - Open the generated JSON metrics, plots, retrieval examples, and report tables.
   - Explain Macro-F1, AUC-PR, MAE, and Recall@K where applicable.
7. **Report and submission link (30 seconds)**
   - Show the final report and confirm that the demonstrated outputs are included.
   - Display the YouTube URL and verify that its visibility is Public or Unlisted.

## PowerShell commands to use
```powershell
Set-Location 'D:\425 project\gnn-bert-music-context'
python --version
# Create the environment only once. If .venv already exists, skip this line.
if (-not (Test-Path '.venv\Scripts\python.exe')) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Preflight checks: run these before downloading or training.
Get-ChildItem
Get-ChildItem src
Get-ChildItem notebooks
Get-ChildItem results
Get-ChildItem data\splits\musiccaps
Get-ChildItem checkpoints\musiccaps_task4_95
Test-Path 'data\splits\musiccaps\test.json'
Test-Path 'checkpoints\musiccaps_task4_95\task4_epoch_1.pt'

# Recommended video demo: use the verified local 95-pair MusicCaps subset.
# This is reproducible and does not require a new YouTube download.
python src\evaluate_fma_multimodal.py --task task4 --manifest-root data\splits\musiccaps --checkpoint checkpoints\musiccaps_task4_95\task4_epoch_1.pt --output results\video_task4.json
# Optional: train one epoch and save it in a clearly named folder.
python src\train.py --task task4 --manifest-root data\splits\musiccaps --epochs 1 --batch-size 4 --lr 1e-4 --checkpoint-dir checkpoints\video_task4 --history-output results\video_task4_history.json
python src\evaluate_fma_multimodal.py --task task4 --manifest-root data\splits\musiccaps --checkpoint checkpoints\video_task4\task4_epoch_1.pt --output results\video_task4_trained_metrics.json
# Show an audio file during the video, if local MusicCaps audio exists.
$audio = Get-ChildItem 'data\raw\musiccaps\audio\*.wav' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($audio) { Start-Process $audio.FullName; Write-Output "Playing $($audio.FullName)" } else { Write-Output 'No local WAV file; show the graph/notebook instead.' }

# Open the executed demonstration notebook.
jupyter notebook notebooks\demo_context.ipynb
# The project ZIP is uploaded separately through the LMS submission field.
# Do not create or verify the ZIP as part of the recorded model demonstration.
```

### Optional download/preprocessing commands
Run these only when you intentionally want to create a fresh local subset. They require `data/raw/musiccaps/musiccaps-public.csv`, `yt-dlp`, internet access, and FFmpeg. The downloader may skip unavailable YouTube sources.

```powershell
python src\download_musiccaps.py --limit 10
python src\preprocess_musiccaps.py --limit 10
```

The preprocessing command requires at least three successfully downloaded audio-caption pairs. If fewer than three are available, use the verified local `data\splits\musiccaps` manifests instead. Do not claim that all requested clips downloaded successfully.

> **Important:** The current repository already has a verified 95-pair MusicCaps subset and a one-epoch checkpoint. Downloading is not required for the video demonstration.

If the project already contains verified checkpoints and results, show those instead of retraining during the recording. Training/download commands may take a long time and YouTube demonstrations should be reliable.

## Speaking script — four model tasks
Use Monitor 2 for this teleprompter while Monitor 1 is recorded. The course specification defines exactly four model tasks. Task 4 is the advanced optional rubric task and is included because this project implements it. The end-to-end notebook stages are distributed between Task 2 (record loading, audio features, graph construction, and graph visualization) and Task 3 (fusion checkpoint inference and prediction interpretation). ZIP upload is separate from the video and is not a fifth task. In every section, first run the commands, then show the exact files listed, and read the voiceover only while those files are visible. Do not claim a result that is not displayed.

### Opening and setup
**PowerShell Commands**
```powershell
Set-Location 'D:\425 project\gnn-bert-music-context'
.\.venv\Scripts\Activate.ps1
python --version
Get-ChildItem
Get-ChildItem src
Get-ChildItem data
Get-ChildItem notebooks
Get-ChildItem results
Get-ChildItem report
code README.md
code config.yaml
```
**On-Screen Visual Checklist — show these before speaking**
- `README.md`: project title, Group 27 table, and four supported model tasks.
- Explorer or PowerShell: `src`, `data`, `notebooks`, `results`, and `report`.
- `config.yaml`: 22,050 Hz, 128 mel bins, 12 chroma bins, five-second segments, and `distilbert-base-uncased`.
**Voiceover Teleprompter — say while the checklist is visible**
“Hello. We are Group 27 demonstrating GNN-BERT Music Context. The project combines BERT language representations with audio structure graphs. I will demonstrate four model tasks: Task 1, the BERT tag baseline; Task 2, the audio graph model; Task 3, GNN-BERT fusion; and Task 4, advanced MusicCaps retrieval. The end-to-end notebook stages are shown inside Tasks 2 and 3, where they belong technically. I am using the project environment and saved checkpoints, and I will show actual outputs rather than placeholder results.”

### Task 1 — BERT baseline for music tag understanding
**PowerShell Commands**
```powershell
code src\bert_encoder.py
code src\train.py
Get-ChildItem data\splits\musiccaps_task1_proxy
Get-Content data\splits\musiccaps_task1_proxy\proxy_report.json
Get-Content results\expanded_fma_task1_metrics.json
Get-Content results\task1_prediction_examples.json
Invoke-Item results\plots\task1_f1_training_curve.png
```
**On-Screen Visual Checklist — show in this order**
- The Task 1 description and the text-only baseline objective.
- `src\bert_encoder.py`: `BERTTextEncoder.__init__` and `forward`, including `distilbert-base-uncased`, the tokenizer settings, `input_ids`, `attention_mask`, `last_hidden_state`, and the pooled first-token vector.
- `src\train.py`: `run_task_1`, especially `model.classifier`, `BCEWithLogitsLoss`, the optimizer step, gradient clipping, validation mode, and checkpoint/history output.
- `src\train.py` `main`: the Task 1 dataset selection and the `nn.Linear(768, num_tags)` classification head; point out that Task 1 does not construct a graph.
- `data\splits\musiccaps_task1_proxy\proxy_report.json`, then `train.json`, `val.json`, and `test.json`.
- `results\expanded_fma_task1_metrics.json`: dataset/split/sample count, label list, and held-out metrics.
- `results\task1_prediction_examples.json`: one or two real inputs, target tags, thresholded predictions, and top-five scores.
- `results\plots\task1_f1_training_curve.png`: the saved training history visualisation.
**Voiceover Teleprompter — read beside the matching file on Monitor 1**

- **While the Task 1 description is visible, say:**
  “I am beginning Task 1, the BERT baseline for music tag understanding. This is deliberately a text-only experiment: the model receives a music description, such as a caption or metadata text, and predicts multiple tags for the same track. Because the task is multi-label, several tags may be correct at once. This gives us a clean language baseline before we introduce audio graphs in Task 2 or combine both modalities in Task 3.”

- **While `src\bert_encoder.py` is visible, say:**
  “This file defines `BERTTextEncoder`. During initialization it loads the `distilbert-base-uncased` tokenizer and transformer backbone, uses a maximum sequence length of 128, and applies dropout to the pooled vector. In `forward`, the tokenizer pads and truncates the text, automatically adds the model’s special tokens, and returns `input_ids` plus an `attention_mask`. DistilBERT uses that mask to produce contextual hidden states with shape `[batch, sequence length, 768]`. The code takes the first token representation, the model’s pooled `[CLS]`-style summary, and returns both the full sequence and this text vector. At this stage there is no waveform, graph, or GNN input—only text.”

- **While the classification-head and forward code in `src\train.py` is visible, say:**
  “The classifier is attached in `main` as `nn.Linear(768, num_tags)`, so it converts the pooled DistilBERT vector into one logit per tag. The encoder returns the pooled vector, and `run_task_1` applies this head before comparing the logits with the multi-hot target vector. The logits are independent because this is multi-label prediction: one track can have several positive tags. At inference, sigmoid maps each logit to a score from zero to one, and a threshold such as 0.5 produces the displayed tag decisions. Training uses `BCEWithLogitsLoss`, which combines the sigmoid operation and binary cross-entropy in a numerically stable way across all tag outputs.”

- **While `src\train.py` is visible, say:**
  “This is the Task 1 training path. The loader supplies a list of text strings and a multi-hot `tags` tensor. For each batch, the encoder tokenizes the text, returns the pooled representation, and the linear classifier produces tag logits. The loop computes `BCEWithLogitsLoss`, clears old gradients, backpropagates, clips the gradient norm to 1.0, steps AdamW, and advances the warm-up scheduler. Validation switches to evaluation mode and uses `torch.no_grad()`. At the end of each epoch, the code records loss, Macro-F1, Micro-F1, and AUC-PR, then saves a Task 1 checkpoint and optional history JSON. The Task 1 branch never creates a `Batch`, graph, or GNN representation, so it remains a genuine text-only baseline for comparison with Tasks 2 and 3.”

- **While `data\splits\musiccaps_task1_proxy\proxy_report.json` and the split files are visible, say:**
  “These files are the reproducible proxy benchmark used for the MusicCaps text example. `proxy_report.json` supplies the label vocabulary, and the dataset class reads each caption and creates a multi-hot target by matching the recorded tag names. The train, validation, and test manifests keep the evaluation split explicit. These labels come from deterministic lexical matches between captions and tag names, not from independent human annotation. Therefore I will describe this result accurately as a caption-to-tag proxy benchmark, not as a human-ground-truth MusicCaps tagging result.”

- **While `results\expanded_fma_task1_metrics.json` is visible, say:**
  “This is the saved held-out Task 1 test result. I first point out the metadata: it identifies Task 1, the test split, the sample count, and the label vocabulary. Macro-F1 calculates F1 separately for each tag and averages those scores, giving every tag equal weight. Micro-F1 pools true positives, false positives, and false negatives over all tag decisions, so common tags have more influence. AUC-PR summarizes the precision–recall curve from the continuous sigmoid scores and is informative when positive labels are sparse. The displayed file reports Macro-F1 of 0.0, Micro-F1 of 0.0, and AUC-PR of approximately 0.0334, so this small proxy result should be presented honestly rather than overstated.”

- **While `results\task1_prediction_examples.json` is visible, say:**
  “This file lets us inspect behaviour rather than relying only on aggregate metrics. Each example includes the track text, target tags, tags selected at the 0.5 threshold, and the top-five labels with their sigmoid scores. For example, the displayed predictions show scores close to 0.5 and can differ from the target tag, which is consistent with the reported zero F1 on this held-out sample. These are real outputs from the text-only classifier.”

- **While `results\plots\task1_f1_training_curve.png` is visible, say:**
  “This curve shows how the training and validation F1 measures changed across epochs; I will distinguish the plotted history from the final held-out test JSON. Taken together, the encoder implementation explains the representation, the training branch explains optimisation, the proxy manifests define the data, and the metrics and examples show the actual outcome. This completes Task 1 as a transparent BERT text baseline. We can now compare it with the audio-only relational representation in Task 2 and the multimodal GNN-BERT model in Task 3.”

### Task 2 — GNN on music structure graphs
**PowerShell Commands**
```powershell
code src\audio_features.py
code src\graph_builder.py
code src\gnn_model.py
code notebooks\demo_context.ipynb
Get-ChildItem data\processed\graphs
Get-Content results\expanded_fma_task2_metrics.json
Get-Content results\task2_gnn_3epoch_metrics.json
Get-Content results\task2_gnn_cnn_comparison.json
Invoke-Item results\plots\task2_gnn_cnn_comparison.png
```
**On-Screen Visual Checklist — show in this order**
- `src\audio_features.py`: audio resampling/feature extraction and segment preparation.
- `src\graph_builder.py`: audio segment features, temporal edges, and similarity edges.
- `src\gnn_model.py`: GraphSAGE/GAT layers, message passing, mean pooling, and prediction head.
- `notebooks\demo_context.ipynb`: the record-loading and preprocessing cells used for the real demonstration.
- Notebook output showing the selected record and its metadata text/context.
- Notebook output showing prepared audio features, segment count, and the constructed graph's node and edge counts.
- `data\processed\graphs`: the processed graph samples, showing node features and edges; show at least 20 `.pt` or `.json` samples when available.
- The notebook graph visualization, with nodes, edges, title, and readable layout.
- `results\expanded_fma_task2_metrics.json` and `results\task2_gnn_3epoch_metrics.json`: Task 2 test metrics.
- `results\task2_gnn_cnn_comparison.json`: numerical GNN/CNN comparison.
- `results\plots\task2_gnn_cnn_comparison.png`: visual GNN-versus-CNN comparison.
**Voiceover Teleprompter — read beside the matching file on Monitor 1**

- **While the Task 2 description is visible, say:**
  “Task 2 is the audio-only GNN model. Unlike Task 1, this experiment does not use captions or BERT. It builds a graph from audio features and predicts genres or music tags.”

- **While `src\audio_features.py` is visible, say:**
  “This is the audio-feature preparation stage. The pipeline resamples audio and prepares segment-level descriptors such as mel or chroma features. Those prepared features are the node inputs for the graph.”

- **While `src\graph_builder.py` is visible, say:**
  “This file receives the prepared segment descriptors and constructs the music graph. Each audio segment becomes a node. Temporal edges connect neighbouring segments, while similarity edges connect segments whose feature descriptors are sufficiently similar.”

- **While `src\gnn_model.py` is visible, say:**
  “This file implements the graph encoder. GraphSAGE updates each node by combining its current representation with the mean representation of its neighbours. After message passing, mean pooling converts all node representations into one graph-level vector. The prediction head maps that vector to tag or genre outputs.”

- **While the notebook setup and record-loading cells are visible, say:**
  “I am now using the real demonstration record. The notebook loads the held-out record and its metadata rather than inventing an example. This setup belongs with Task 2 because the audio record must first be converted into the graph consumed by the audio-only GNN.”

- **While the notebook record output and the feature/graph source files are visible, say:**
  “The notebook uses the verified processed graph for this reproducible inference example. The source files show how audio features are prepared and how segments and relationships become graph nodes and edges; the stored graph is the resulting GNN input used here.”

- **While `data\processed\graphs` is visible, say:**
  “These are the processed graph artifacts used by Task 2. Each file stores the node feature matrix and the edge index consumed by the GNN. The nodes represent audio segments, and the edges represent the temporal or feature-similarity relationships used during message passing.”

- **While the graph sample or notebook graph visualization is visible, say:**
  “The plotted points correspond to graph nodes and the connecting lines correspond to graph edges. The visualization confirms that the model is not receiving a flat audio vector only; it is receiving audio segments together with their relationships.”

- **While the Task 2 JSON metrics are visible, say:**
  “These JSON files contain the held-out Task 2 test metrics. I am showing the values from the saved evaluation files, not placeholder values.”

- **While `results\task2_gnn_cnn_comparison.json` and `task2_gnn_cnn_comparison.png` are visible, say:**
  “This comparison evaluates the required CNN mel-spectrogram baseline against the audio graph model. The numerical JSON gives the recorded comparison, and the PNG makes the difference visible. Task 2 therefore tests whether relational graph structure provides useful information beyond a conventional spectrogram CNN.”



### Task 3 — GNN-BERT fusion for multi-context understanding
**PowerShell Commands**
```powershell
code src\fusion_model.py; code src\train.py
code notebooks\demo_context.ipynb
Get-Content results\expanded_fma_task3_metrics.json
Get-Content results\task3_fma_submission_metrics.json
Get-Content results\task3_analysis.json
Get-Content results\task3_caption_case_studies.json
Invoke-Item results\plots\task3_tsne.png
Invoke-Item results\plots\task3_deam_tsne.png
```
**On-Screen Visual Checklist — show in this order**

- `src\fusion_model.py`: graph projection, text projection, cross-attention, tag head, valence head, and arousal head.
- `notebooks\demo_context.ipynb`: the real Task 3 checkpoint inference code and printed tensor shapes.
- Notebook output: text context, reference tags, graph node count, and graph edge count.
- The checkpoint-loading and inference cell, including evaluation mode and `torch.no_grad()`.
- Notebook output: text representation shape `(1, 28, 768)`, graph representation shape `(1, 128)`, and tag-logit shape `(1, 26)` when those are the displayed values.
- Notebook output after sigmoid: ranked top predicted tags, probabilities, and comparison with reference tags.
- `results\expanded_fma_task3_metrics.json` and `results\task3_fma_submission_metrics.json`: dataset, split, Macro-F1, Micro-F1, and AUC-PR.
- `results\task3_analysis.json`: ablation/analysis evidence.
- `results\plots\task3_tsne.png` and `results\plots\task3_deam_tsne.png`: Task 3 representation visualizations.
- `results\task3_caption_case_studies.json`: three graph–caption alignment case studies.
**Voiceover Teleprompter — read beside the matching file or notebook output on Monitor 1**

- **While the Task 3 description is visible, say:**
  “Task 3 is the hard GNN-BERT fusion task. It combines structural information from the audio graph with semantic information from captions or metadata to predict multi-context music labels.”

- **While `src\fusion_model.py` is visible, say:**
  “The GNN produces a graph representation, and BERT produces contextual text hidden states. The fusion model projects both into a compatible dimension and applies cross-attention before prediction. The tag head predicts multiple tags, while the valence and arousal heads provide auxiliary emotion outputs when emotion targets are available.”

- **While the Task 3 checkpoint-loading and inference cell in `notebooks\demo_context.ipynb` is visible, say:**
  “This cell loads the real trained Task 3 checkpoint into `FusionTrainingModel`, switches the model to evaluation mode, and disables gradients with `torch.no_grad()`. The `encode` call produces the text hidden states and graph embedding. The fusion head then returns tag logits, valence prediction, and arousal prediction.”

- **While the notebook input and printed context are visible, say:**
  “The notebook now shows the actual text context and reference tags for one track. It also prints the graph size. In this example, 11 graph nodes represent audio segments and 110 edges represent the stored relationships between those nodes.”

- **While the notebook prints tensor shapes, say:**
  “The text representation has shape `(1, 28, 768)`: one example, 28 text positions, and 768 BERT features. The graph representation has shape `(1, 128)`: one graph-level audio vector with 128 features. The tag logits have shape `(1, 26)`, meaning the classifier produces one score for each of 26 tags.”

- **While the notebook decodes predictions, say:**
  “The notebook applies sigmoid to the tag logits, sorts the probabilities in descending order, and selects the highest-scoring tags. This converts raw model scores into interpretable ranked tag probabilities that can be compared with the reference tags.”

- **While the Task 3 JSON metrics are visible, say:**
  “These JSON files contain the held-out Task 3 evaluation results. Macro-F1 averages the F1 score of each tag, Micro-F1 pools all tag decisions, and AUC-PR summarizes precision and recall. I will only discuss MAE or R-squared if the displayed result contains emotion targets.”

- **While `results\task3_analysis.json`, the t-SNE plots, and case studies are visible, say:**
  “These files extend the numerical result with ablation evidence, representation visualizations, and three graph-caption case studies. They show how the BERT branch, GNN branch, and fusion representation are analyzed together.”

### Task 4 — advanced MusicCaps cross-modal alignment
**PowerShell Commands**
```powershell
code src\evaluate_fma_multimodal.py; code src\contrastive.py
Get-ChildItem data\splits\musiccaps; Get-ChildItem checkpoints\musiccaps_task4_95
Test-Path 'checkpoints\musiccaps_task4_95\task4_epoch_1.pt'
python src\evaluate_fma_multimodal.py --task task4 --manifest-root data\splits\musiccaps --checkpoint checkpoints\musiccaps_task4_95\task4_epoch_1.pt --output results\video_task4.json
Get-Content results\video_task4.json
Get-Content results\retrieval_examples\musiccaps_examples.json
Invoke-Item results\plots\musiccaps_task4_loss_curve.png
```
**On-Screen Visual Checklist — show in this order**

- `src\contrastive.py`: normalized graph/text embeddings and contrastive objective.
- `src\evaluate_fma_multimodal.py`: test-manifest loading, checkpoint loading, similarity matrix, and retrieval evaluation.
- `data\splits\musiccaps\train.json`, `val.json`, `test.json`, and `alignment_report.json`.
- `checkpoints\musiccaps_task4_95\task4_epoch_1.pt` and the successful `Test-Path` output.
- `results\video_task4.json`: MusicCaps test sample count and both retrieval directions.
- `results\retrieval_examples\musiccaps_examples.json`: caption, query ID, top-three retrieved IDs, and similarity values.
- `results\plots\musiccaps_task4_loss_curve.png`, the Task 4 contrastive training curve, if available.
**Voiceover Teleprompter — read beside the matching file on Monitor 1**

- **While the Task 4 description is visible, say:**
  “Task 4 is the advanced MusicCaps cross-modal alignment task. It learns a shared embedding space so that a matching audio graph and caption are close, while nonmatching pairs are separated.”

- **While `src\contrastive.py` is visible, say:**
  “The contrastive module normalizes the graph and text embeddings and applies the InfoNCE objective. Within a batch, the matching graph-caption pair is the positive pair and the other pairs provide negatives.”

- **While `src\evaluate_fma_multimodal.py` is visible, say:**
  “This evaluation script loads the held-out manifest and saved checkpoint, encodes the graph and captions, builds the similarity matrix, and calculates retrieval in both directions.”

- **While the MusicCaps manifests and checkpoint are visible, say:**
  “These are the reproducible MusicCaps train, validation, and test manifests. This is the saved checkpoint used for the prepared evaluation, so I am not starting a new long training run.”

- **While `results\video_task4.json` is visible, say:**
  “This JSON reports caption-to-audio and audio-to-caption Recall@1, Recall@5, and Recall@10 on the held-out test split. Recall@K asks whether the correct matching item appears within the top K results.”

- **While the retrieval examples and loss curve are visible, say:**
  “These examples show the query caption, query audio ID, retrieved audio IDs, ranks, and similarity values. The loss curve shows how the contrastive objective changed during training. The similarity values are used to rank candidates; the correct paired item is then checked at ranks one, five, and ten. Task 4 is optional in the rubric, but it is included in this project demonstration.”

## Closing report and video-link checks
These are closing activities after Tasks 1–4, not an additional model task. Do not create or verify the ZIP during the recording; upload it separately through the LMS field.

**On-Screen Visual Checklist**

- `report\final_report.pdf`, including the Task 1–4 results and discussion.
- The separate YouTube URL field and its Public/Unlisted visibility, if this is shown in the recording.

**Voiceover Teleprompter**

- “The four model tasks are complete. The report consolidates the Task 1, Task 2, Task 3, and optional Task 4 evidence already shown; these closing checks are not a fifth task.”
- “The project ZIP is submitted separately through the LMS upload field; it is not part of this video demonstration.”
- “The YouTube URL is submitted separately in the video-link field. Its visibility is set to Public or Unlisted, and the link is tested before submission. Thank you.”

## Final checklist
- [ ] YouTube video has clear audio and readable terminal output.
- [ ] Video URL works in an incognito/private browser window.
- [ ] Video visibility is Public or Unlisted, as required by the instructor.
- [ ] Final report PDF is generated from `report/final_report.tex` and included if required.
- [ ] Demo notebook and results are included or linked as required.
- [ ] The local evaluation command completed successfully and `results/video_task4.json` was created.
- [ ] Any displayed metric is labelled with its dataset, split, and checkpoint.
- [ ] The YouTube link is pasted into the form's video field.
- [ ] The project ZIP is uploaded separately through the “Zip files of codes” upload field, if required.
