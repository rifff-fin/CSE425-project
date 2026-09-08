# Supervised Neural Network Project: GNN-Based BERT for Understanding Context from Music

**Course:** Neural Networks (CSE425 / EEE474 / CSE715)
**Prepared By:** Moin Mostakim
**Submission Deadline:** 2nd October, 2026

---

> **Document status:** Project specification. Update this document only when the course requirements or approved project scope changes.

## 1 Project Motivation

Music is a multi-layered signal where "context" spans melody, harmony, rhythm, lyrics, metadata tags, and listener-described semantics. A single clip may simultaneously express:

- Genre and era (e.g., Jazz, 1960s)
- Mood and emotion (e.g., melancholic, high arousal)
- Harmonic structure (chord progressions, key changes)
- Lyrical semantics (themes, sentiment in vocals)
- Timbral texture (instrumentation, production style)

Pure sequence models (CNN/RNN on spectrograms) capture local patterns but often miss relational structure: how chord C → G → Am relates to a lyrical theme, or how repeated segments form a song graph.

**Goal:** Build a hybrid BERT + Graph Neural Network (GNN) system that understands musical context by combining:

1. **BERT** — contextual language representations from lyrics, tags, and natural-language music descriptions (MusicCaps).
2. **GNN** — message passing on music structure graphs (chord transitions, segment similarity, co-occurrence of audio events).

Unlike generative music projects, this assignment focuses on understanding and prediction: multi-label tagging, emotion regression, and cross-modal alignment between audio structure and text.

---

## 2 Problem Definition

Let a music track be represented as a tuple:

$$T = (X_{audio}, X_{text}, G, y),$$

where:

- $X_{audio}$ = log-mel spectrogram or chroma features over $T$ frames,
- $X_{text}$ = tokenized lyrics, user tags, or caption tokens,
- $G = (V, E)$ = music structure graph (nodes = segments/chords/events; edges = transitions or similarity),
- $y$ = context labels (genre, mood tags, valence/arousal, etc.).

**BERT text encoder** (frozen or fine-tuned) maps text to contextual embeddings:

$$H_{text} = \text{BERT}(X_{text}) \in \mathbb{R}^{L \times d}.$$

**GNN encoder** with $L$ layers updates node features via message passing:

$$h_i^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{AGG}\left(h_i^{(l)}, \{h_j^{(l)} : j \in \mathcal{N}(i)\}\right)\right),$$

where $h_i^{(0)}$ is initialized from audio segment embeddings or chord features.

**Fusion readout** combines graph-level representation $g$ and text CLS vector $t$:

$$z = \text{Fusion}(g, t), \qquad \hat{y} = \sigma(Wz + b).$$

**Objective** (multi-label context understanding):

$$\mathcal{L} = -\sum_{k=1}^{K}\left[y_k \log \hat{y}_k + (1 - y_k)\log(1 - \hat{y}_k)\right] + \lambda \mathcal{L}_{aux}.$$

---

## 3 Dataset Requirements

Students must use **at least one primary audio dataset** and **one text/tag dataset** from Table 1. Advanced tasks should combine FMA or MagnaTagATune with MusicCaps or DEAM.

**Recommended primary pairing**

- **Medium/Hard:** FMA-medium (audio + genre/tags) + chord/segment graphs built from chroma features.
- **Advanced:** MusicCaps captions (BERT) aligned with audio graphs from corresponding 10 s clips.
- **Emotion extension:** DEAM for valence/arousal regression as auxiliary $\mathcal{L}_{aux}$.

**Preprocessing pipeline**

1. **Audio:** Resample to 22,050 Hz; extract log-mel spectrogram (128 bins) or chroma (12 bins); normalize per track.
2. **Segmentation:** Split each track into fixed windows (e.g., 5–10 s) or beat-synchronous segments using `librosa`.
3. **Graph construction:**
   - Chord-transition graph: nodes = unique chords; edges = observed transitions weighted by count.
   - Segment graph: nodes = time segments; edges = temporal adjacency + cosine similarity of MFCC/chroma > $\tau$.
4. **Text:** Tokenize lyrics/tags/captions with BERT tokenizer (max length 128–256); pad/truncate.
5. **Splits:** Use the recommended FMA split when applicable. For MagnaTagATune, create reproducible train/validation/test splits grouped by track (and preferably by artist when artist metadata is available); for DEAM, use the dataset's predefined partition if provided. Prevent artist or source-track leakage across train/validation/test.

### Table 1: Recommended datasets for GNN–BERT music context understanding

| Dataset Name | Context / Label Coverage | Task | Official Link |
|---|---|---|---|
| FMA (small / medium) | Genre (8–16 classes), top tags, artist & album metadata | 1–3 | FMA Download |
| MagnaTagATune | 188 multi-label tags (genre, mood, instrument); 25,877 clips | 1, 3 | MagnaTagATune |
| GTZAN | 10 genres; 1,000 tracks (30 s clips); standard easy baseline | 1–2 | GTZAN Download |
| DEAM | Continuous valence & arousal (1–9) every 0.5 s | 3–4 | DEAM Download |
| MusicCaps | 5,521 clips with expert natural-language captions (Google) | 1, 4 | MusicCaps |
| Million Song Dataset + tagtraum | Last.fm-style tags for 188,916 tracks (Echo Nest audio features) | 1, 3 | MSD Download |
| Lakh MIDI Clean | Symbolic MIDI for chord-transition & melody graph construction | 2–3 | LMD Download |
| EmoMusic | 1,000 clips with quadrant emotion labels (happy/sad × calm/energetic) | 3 | EmoMusic |

*Task key: 1 = BERT tags — 2 = GNN structure — 3 = GNN–BERT fusion — 4 = contrastive retrieval. Use ≥ 1 audio dataset and ≥ 1 text/tag source.*

---

## 4 Model Tasks and Mathematical Formulation

### 4.1 Task 1 (Easy): BERT Baseline for Music Tag Understanding

**Goal.** Implement a BERT-based multi-label classifier on textual music context (tags, captions, or lyrics) without graph structure.

**Model.**

$$t = \text{BERT}_{CLS}(X_{text}), \qquad \hat{y}_k = \sigma(w_k^{\top} t + b_k).$$

**Loss** (binary cross-entropy per tag $k$):

$$\mathcal{L}_{BERT} = -\frac{1}{K}\sum_{k=1}^{K}\left[y_k \log \hat{y}_k + (1 - y_k)\log(1 - \hat{y}_k)\right].$$

**Deliverables.**

- BERT fine-tuning code (HuggingFace `bert-base-uncased` or `distilbert-base-uncased`)
- Results on MagnaTagATune tag subset (top-50 tags) or MusicCaps caption → tag proxy task
- Macro-F1 / Micro-F1 curves vs. training epochs
- 5 example predictions with attention visualization (optional)

### 4.2 Task 2 (Medium): GNN on Music Structure Graphs

**Goal.** Build a GraphSAGE or GAT encoder on segment/chord graphs using audio-only node features; predict genre or top tags.

**GraphSAGE update:**

$$h_i^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{CONCAT}\left(h_i^{(l)}, \text{MEAN}_{j \in \mathcal{N}(i)} h_j^{(l)}\right)\right).$$

**Graph readout** (mean pooling):

$$g = \frac{1}{|V|}\sum_{i \in V} h_i^{(L)}, \qquad \hat{y} = \sigma(Wg + b).$$

**Deliverables.**

- Graph construction scripts (chroma/MFCC segment graphs)
- GNN implementation (PyTorch Geometric)
- Genre classification on GTZAN or FMA-small
- Comparison vs. CNN baseline on mel-spectrogram

### 4.3 Task 3 (Hard): GNN–BERT Fusion for Multi-Context Understanding

**Goal.** Fuse structural (GNN) and semantic text (BERT) representations to predict multi-label context: genre + mood tags + (optional) valence/arousal.

**Cross-attention fusion** (recommended):

$$A = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d}}\right), \qquad Q = gW_Q, \quad K = H_{text}W_K,$$

$$z = \text{CONCAT}(g, A H_{text}), \qquad \hat{y} = \sigma(Wz).$$

**Multi-task loss:**

$$\mathcal{L} = \mathcal{L}_{tags} + \alpha \lVert v - \hat{v} \rVert_2^2 + \beta \lVert a - \hat{a} \rVert_2^2,$$

where $(v, a)$ are DEAM valence/arousal targets when available.

**Deliverables.**

- End-to-end GNN–BERT fusion model
- Ablation: BERT-only, GNN-only, early concat, cross-attention
- Results on FMA-medium or MagnaTagATune (Macro-F1, AUC-PR)
- t-SNE of $z$ coloured by genre and mood
- 3 case studies showing graph paths + caption/lyric alignment

### 4.4 Task 4 (Advanced): Cross-Modal MusicCaps Alignment

**Goal.** Learn a shared embedding space between audio graphs and natural-language descriptions (MusicCaps) using contrastive learning.

**InfoNCE contrastive loss** for paired (graph, caption) $(g_i, t_i)$:

$$\mathcal{L}_{NCE} = -\log \frac{\exp(\text{sim}(g_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(g_i, t_j)/\tau)},$$

where $\text{sim}(u, v) = u^{\top}v / (\lVert u \rVert \lVert v \rVert)$ and $\tau$ is temperature.

**Retrieval metrics:** Caption → Audio R@1, R@5, R@10; Audio → Caption R@K.

**Deliverables.**

- Dual-encoder GNN–BERT with contrastive training
- Retrieval evaluation table on MusicCaps test split
- 10 qualitative retrieval examples (query caption → top-3 matched clips)
- Zero-shot tag prediction from captions vs. Task 3 supervised model

---

## 5 Task Roadmap Summary

### Table 2: Four-task roadmap with objectives and deliverables

| Task | Model | Key equations | Deliverables |
|---|---|---|---|
| 1 (Easy) | BERT tag classifier | $\hat{y}_k = \sigma(w_k^{\top}t)$; $\mathcal{L}_{BERT}$ | BERT code; tag F1; 5 predictions |
| 2 (Medium) | GNN on segment/chord graph | GraphSAGE update; mean readout $g$ | Graph builder; GNN code; vs. CNN |
| 3 (Hard) | GNN–BERT fusion | Cross-attention; multi-task $\mathcal{L}$ | Ablations; t-SNE; case studies |
| 4 (Advanced, optional) | Contrastive dual-encoder | InfoNCE $\mathcal{L}_{NCE}$; R@K | Retrieval table; 10 qualitative examples |

---

## 6 Evaluation Metrics

**Tag / genre classification.**

$$\text{Prec}_k = \frac{TP_k}{TP_k + FP_k}, \qquad \text{Rec}_k = \frac{TP_k}{TP_k + FN_k}, \qquad F1_k = \frac{2\,\text{Prec}_k\,\text{Rec}_k}{\text{Prec}_k + \text{Rec}_k}.$$

Macro-F1 $= \frac{1}{K}\sum_k F1_k$; Micro-F1 pools globally.

**AUC-PR.** Area under precision–recall curve per tag; report mean AUC-PR over tags.

**Emotion regression (DEAM).**

$$\text{MAE}_v = \frac{1}{N}\sum_{i=1}^{N} |v_i - \hat{v}_i|, \qquad R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}.$$

**Graph coherence score (optional analysis).** Measure whether high-attention edges align with repeated chord patterns:

$$S_{graph} = \frac{1}{|E|}\sum_{(i,j) \in E} \mathbb{1}[\cos(h_i, h_j) > \tau].$$

**Human evaluation (Task 4, optional).** If Task 4 is completed, at least 5 listeners should rate whether each retrieved clip matches its caption on a 1–5 scale; report the number of raters and the mean score.

---

## 7 Algorithms for the Tasks

**Algorithm 1: Task 1 — BERT Multi-Label Tag Classifier**

```
Require: Tag corpus D = {(X_text^(i), y^(i))}
Require: Pretrained BERT, learning rate η, epochs E
1: Add classification head W, b on top of CLS token
2: for epoch = 1 to E do
3:     for each mini-batch in D do
4:         t ← BERT_CLS(X_text)
5:         ŷ ← σ(Wt + b)
6:         L ← BCE(y, ŷ)
7:         Update BERT (optional) and head via η∇L
8:     end for
9: end for
10: return fine-tuned BERT tag model
```

**Algorithm 2: Task 2 — GNN Encoder on Music Segment Graph**

```
Require: Audio tracks {T_i}, label set y
1: for each track T do
2:     Extract segment features h_i^(0) from mel/chroma windows
3:     Build graph G = (V, E) (temporal + similarity edges)
4: end for
5: for epoch = 1 to E do
6:     for each graph G in batch do
7:         for layer l = 0 to L_g-1 do
8:             h_i^(l+1) ← GraphSAGE(h_i^(l), N(i))
9:         end for
10:        g ← MEANPOOL({h_i^(L_g)})
11:        𝓛 ← BCE(y, σ(Wg))
12:        Update GNN parameters
13:    end for
14: end for
```

**Algorithm 3: Task 3 — GNN–BERT Fusion for Context Understanding**

```
Require: Paired data (G, X_text, y)
1: H_text ← BERT(X_text), t ← H_text[CLS]
2: g ← GNN_Readout(G)
3: c ← CrossAttention(g, H_text)   ▷ or use t for simple concatenation
4: z ← CONCAT(g, c)                 ▷ or CONCAT(g, t)
5: ŷ ← σ(Wz + b)
6: L ← L_tags + α L_emotion
7: Backprop through fusion, GNN, and (partial) BERT
```

**Algorithm 4: Task 4 — Contrastive GNN–BERT (MusicCaps)**

```
Require: Paired (G_i, caption_i), temperature τ, batch size N
1: g_i ← Normalize(GNN(G_i))
2: t_i ← Normalize(BERT_CLS(caption_i))
3: for each training step do
4: Compute similarity matrices S^gt_ij = g_i^T t_j / τ and S^tg_ij = t_i^T g_j / τ
5: L_NCE ← 1/2 [CE(S^gt, diag) + CE(S^tg, diag)]
6: Update encoders to minimize L_NCE
7: end for
8: Evaluate retrieval R@1, R@5, R@10 on held-out pairs
```

---

## 8 Baseline Models for Comparison

Students must compare against at least two baselines:

- **B1:** Majority-class / random tag predictor
- **B2:** CNN on mel-spectrogram (no graph, no text)
- **B3:** BERT-only (Task 1)
- **B4 (optional):** PCA + MLP on hand-crafted audio features

### Table 3: Illustrative performance comparison
> **Important:** The values below are placeholders for demonstrating the reporting format. Replace every value with results from the same held-out split, and report the random seed and number of runs.

| Model | Macro-F1 | AUC-PR | MAE (emotion) | R@5 (retrieval) |
|---|---|---|---|---|
| Random tags | 0.05 | 0.12 | – | 0.02 |
| CNN mel-spec | 0.41 | 0.38 | 1.25 | – |
| Task 1: BERT-only | 0.48 | 0.44 | – | – |
| Task 2: GNN-only | 0.52 | 0.47 | 1.10 | – |
| Task 3: GNN–BERT | 0.61 | 0.55 | 0.92 | – |
| Task 4: Contrastive | 0.55 | 0.50 | – | 0.38 |

---

## 9 Judging Rubric

### Marks distribution (100 total)

| Component | Marks | Notes |
|---|---|---|
| Task 1 (Easy) | 18 | BERT tags + F1 report |
| Task 2 (Medium) | 22 | GNN graph + CNN baseline |
| Task 3 (Hard) | 22 | Fusion + ablations |
| Task 4 (Advanced, optional) | 18 | Contrastive retrieval; completes the advanced-track marks |
| Report + GitHub | 20 | Reproducibility + paper quality |
| **Total** | **100** | |

### Table 4: Final grading rubric (maps to 15% course grade; total 100 marks)

| Category | Criteria | Weight |
|---|---|---|
| Dataset & preprocessing | Correct splits, no leakage, graph construction documented | 15% |
| Model implementation | BERT + GNN correct, stable training, reproducible code | 25% |
| Context understanding quality | Strong tag/emotion/retrieval results vs. baselines | 20% |
| Baseline comparison | ≥ 2 baselines with fair experimental setup | 15% |
| Metrics & analysis | Macro-F1, AUC-PR, ablations, visualizations | 15% |
| Report & presentation | Clear NeurIPS/IEEE-style report, diagrams, discussion | 10% |

---

## 10 Final Submission Requirements

Each group must submit:

1. GitHub repository or ZIP with full source code
2. Preprocessed graph samples (at least 20 example `.pt` / `.json` graphs)
3. Evaluation tables + plots (F1, AUC-PR, t-SNE, retrieval examples)
4. Final report PDF (6–10 pages; NeurIPS / IEEE / ICML Overleaf template)
5. Demo notebook (`notebooks/demo_context.ipynb`) with one end-to-end inference example

### GitHub project structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/            # FMA, MagnaTagATune, MusicCaps downloads
│   ├── processed/      # graphs, mel-spec, BERT caches
│   └── splits/         # train/val/test JSON
├── notebooks/
│   ├── eda.ipynb
│   └── demo_context.ipynb
├── src/
│   ├── audio_features.py   # mel, chroma, segmentation
│   ├── graph_builder.py    # chord + segment graphs
│   ├── bert_encoder.py
│   ├── gnn_model.py        # GraphSAGE / GAT
│   ├── fusion_model.py     # cross-attention GNN-BERT
│   ├── contrastive.py      # Task 4 InfoNCE
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── metrics.json
│   ├── plots/
│   └── retrieval_examples/
└── report/
    └── final_report.pdf
```

### Report template links (Overleaf)

- NeurIPS 2024: https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh
- IEEE Conference: https://www.overleaf.com/latex/templates/ieee-conference-template
- ICML 2025: https://www.overleaf.com/latex/templates/icml2025-template