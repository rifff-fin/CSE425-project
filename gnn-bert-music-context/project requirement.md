ETA ASOLE PDF tar md 


# Supervised Neural Network Project: GNN-Based BERT for Understanding Context from Music

**Course:** Neural Networks (CSE425 / EEE474 / CSE715)
**Prepared By:** Moin Mostakim
**Submission Deadline:** 2nd October, 2026

---

## 1. Project Motivation

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

Unlike generative music projects, this assignment focuses on **understanding and prediction**: multi-label tagging, emotion regression, and cross-modal alignment between audio structure and text.

---

## 2. Problem Definition

Let a music track be represented as a tuple:

```
T = (X_audio, X_text, G, y)
```

where:

- `X_audio` = log-mel spectrogram or chroma features over T frames
- `X_text` = tokenized lyrics, user tags, or caption tokens
- `G = (V, E)` = music structure graph (nodes = segments/chords/events; edges = transitions or similarity)
- `y` = context labels (genre, mood tags, valence/arousal, etc.)

**BERT text encoder** (frozen or fine-tuned) maps text to contextual embeddings:

```
H_text = BERT(X_text) ∈ R^(L×d)
```

**GNN encoder** with L layers updates node features via message passing:

```
h_i^(l+1) = σ( W^(l) · AGG( h_i^(l), {h_j^(l) : j ∈ N(i)} ) )
```

where `h_i^(0)` is initialized from audio segment embeddings or chord features.

**Fusion readout** combines graph-level representation `g` and text CLS vector `t`:

```
z = Fusion(g, t),   ŷ = σ(Wz + b)
```

**Objective** (multi-label context understanding):

```
L = − Σ_{k=1}^{K} [ y_k log ŷ_k + (1 − y_k) log(1 − ŷ_k) ] + λ L_aux
```

---

## 3. Dataset Requirements

Students must use **at least one primary audio dataset** and **one text/tag dataset** from Table 1. Advanced tasks should combine FMA or MagnaTagATune with MusicCaps or DEAM.

### Recommended primary pairing

- **Medium/Hard:** FMA-medium (audio + genre/tags) + chord/segment graphs built from chroma features.
- **Advanced:** MusicCaps captions (BERT) aligned with audio graphs from corresponding 10s clips.
- **Emotion extension:** DEAM for valence/arousal regression as auxiliary `L_aux`.

### Preprocessing pipeline

1. **Audio:** Resample to 22,050 Hz; extract log-mel spectrogram (128 bins) or chroma (12 bins); normalize per track.
2. **Segmentation:** Split each track into fixed windows (e.g., 5–10s) or beat-synchronous segments using `librosa`.
3. **Graph construction:**
   - *Chord-transition graph:* nodes = unique chords; edges = observed transitions weighted by count.
   - *Segment graph:* nodes = time segments; edges = temporal adjacency + cosine similarity of MFCC/chroma > τ.
4. **Text:** Tokenize lyrics/tags/captions with BERT tokenizer (max length 128–256); pad/truncate.
5. **Splits:** Use official FMA / MagnaTagATune splits; for DEAM use standard train/val partition; no artist leakage across train/test when possible.

### Table 1: Recommended datasets for GNN–BERT music context understanding

| Dataset Name | Context / Label Coverage | Task | Official Link |
|---|---|---|---|
| FMA (small / medium) | Genre (8–16 classes), top tags, artist & album metadata | 1–3 | FMA Download |
| MagnaTagATune | 188 multi-label tags (genre, mood, instrument); 25,877 clips | 1, 3 | MagnaTagATune |
| GTZAN | 10 genres; 1,000 tracks (30s clips); standard easy baseline | 1–2 | GTZAN Download |
| DEAM | Continuous valence & arousal (1–9) every 0.5s | 3–4 | DEAM Download |
| MusicCaps | 5,521 clips with expert natural-language captions (Google) | 1, 4 | MusicCaps |
| Million Song Dataset + tagtraum | Last.fm-style tags for 188,916 tracks (Echo Nest audio features) | 1, 3 | MSD Download |
| Lakh MIDI Clean | Symbolic MIDI for chord-transition & melody graph construction | 2–3 | LMD Download |
| EmoMusic | 1,000 clips with quadrant emotion labels (happy/sad × calm/energetic) | 3 | EmoMusic |

**Task key:** 1 = BERT tags — 2 = GNN structure — 3 = GNN–BERT fusion — 4 = contrastive retrieval. Use ≥ 1 audio dataset and ≥ 1 text/tag source.

---

## 4. Model Tasks and Mathematical Formulation

### 4.1 Task 1 (Easy): BERT Baseline for Music Tag Understanding

**Goal.** Implement a BERT-based multi-label classifier on textual music context (tags, captions, or lyrics) without graph structure.

**Model.**

```
t = BERT_CLS(X_text),   ŷ_k = σ(w_k^T t + b_k)
```

**Loss** (binary cross-entropy per tag k):

```
L_BERT = − (1/K) Σ_{k=1}^{K} [ y_k log ŷ_k + (1 − y_k) log(1 − ŷ_k) ]
```

**Deliverables:**

- BERT fine-tuning code (HuggingFace `bert-base-uncased` or `distilbert-base-uncased`)
- Results on MagnaTagATune tag subset (top-50 tags) or MusicCaps caption → tag proxy task
- Macro-F1 / Micro-F1 curves vs. training epochs
- 5 example predictions with attention visualization (optional)

### 4.2 Task 2 (Medium): GNN on Music Structure Graphs

**Goal.** Build a GraphSAGE or GAT encoder on segment/chord graphs using audio-only node features; predict genre or top tags.

**GraphSAGE update:**

```
h_i^(l+1) = σ( W^(l) · CONCAT( h_i^(l), MEAN_{j∈N(i)} h_j^(l) ) )
```

**Graph readout** (mean pooling):

```
g = (1/|V|) Σ_{i∈V} h_i^(L),   ŷ = σ(Wg + b)
```

**Deliverables:**

- Graph construction scripts (chroma/MFCC segment graphs)
- GNN implementation (PyTorch Geometric)
- Genre classification on GTZAN or FMA-small
- Comparison vs. CNN baseline on mel-spectrogram

### 4.3 Task 3 (Hard): GNN–BERT Fusion for Multi-Context Understanding

**Goal.** Fuse structural (GNN) and semantic text (BERT) representations to predict multi-label context: genre + mood tags + (optional) valence/arousal.

**Cross-attention fusion** (recommended):

```
A = softmax( QK^T / √d ),   Q = g W_Q,   K = H_text W_K
z = CONCAT(g, A H_text),   ŷ = σ(Wz)
```

**Multi-task loss:**

```
L = L_tags + α ‖v − v̂‖² + β ‖a − â‖²
```

where `(v, a)` are DEAM valence/arousal targets when available.

**Deliverables:**

- End-to-end GNN–BERT fusion model
- Ablation: BERT-only, GNN-only, early concat, cross-attention
- Results on FMA-medium or MagnaTagATune (Macro-F1, AUC-PR)
- t-SNE of z coloured by genre and mood
- 3 case studies showing graph paths + caption/lyric alignment

### 4.4 Task 4 (Advanced): Cross-Modal MusicCaps Alignment

**Goal.** Learn a shared embedding space between audio graphs and natural-language descriptions (MusicCaps) using contrastive learning.

**InfoNCE contrastive loss** for paired (graph, caption) `(g_i, t_i)`:

```
L_NCE = − log [ exp(sim(g_i, t_i)/τ) / Σ_{j=1}^{N} exp(sim(g_i, t_j)/τ) ]
```

where `sim(u, v) = u^T v / (‖u‖ ‖v‖)` and τ is temperature.

**Retrieval metrics:** Caption → Audio R@1, R@5, R@10; Audio → Caption R@K.

**Deliverables:**

- Dual-encoder GNN–BERT with contrastive training
- Retrieval evaluation table on MusicCaps test split
- 10 qualitative retrieval examples (query caption → top-3 matched clips)
- Zero-shot tag prediction from captions vs. Task 3 supervised model

---

## 5. Task Roadmap Summary

### Table 2: Four-task roadmap with objectives and deliverables

| Task | Model | Key equations | Deliverables |
|---|---|---|---|
| 1 (Easy) | BERT tag classifier | ŷ_k = σ(w_k^T t); L_BERT | BERT code; tag F1; 5 predictions |
| 2 (Medium) | GNN on segment/chord graph | GraphSAGE update; mean readout g | Graph builder; GNN code; vs. CNN |
| 3 (Hard) | GNN–BERT fusion | Cross-attention; multi-task L | Ablations; t-SNE; case studies |
| 4 (Advanced) | Contrastive dual-encoder | InfoNCE L_NCE; R@K | Retrieval table; 10 qual. examples |

---

## 6. Evaluation Metrics

### Tag / genre classification

```
Prec_k = TP_k / (TP_k + FP_k)
Rec_k  = TP_k / (TP_k + FN_k)
F1_k   = 2 · Prec_k · Rec_k / (Prec_k + Rec_k)
```

Macro-F1 = (1/K) Σ_k F1_k; Micro-F1 pools globally.

### AUC-PR

Area under precision–recall curve per tag; report mean AUC-PR over tags.

### Emotion regression (DEAM)

```
MAE_v = (1/N) Σ_{i=1}^{N} |v_i − v̂_i|
R² = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²
```

### Graph coherence score (optional analysis)

Measure whether high-attention edges align with repeated chord patterns:

```
S_graph = (1/|E|) Σ_{(i,j)∈E} 1[cos(h_i, h_j) > τ]
```

### Human evaluation (Task 4)

Minimum 5 listeners rate whether retrieved clip matches caption on scale [1, 5].

---

## 7. Algorithms for the Tasks

### Algorithm 1: Task 1 — BERT Multi-Label Tag Classifier

```
Require: Tag corpus D = {(X_text^(i), y^(i))}
Require: Pretrained BERT, learning rate η, epochs E

Add classification head W, b on top of CLS token
for epoch = 1 to E do
    for each mini-batch in D do
        t ← BERT_CLS(X_text)
        ŷ ← σ(Wt + b)
        L ← BCE(y, ŷ)
        Update BERT (optional) and head via η∇L
    end for
end for
return fine-tuned BERT tag model
```

### Algorithm 2: Task 2 — GNN Encoder on Music Segment Graph

```
Require: Audio tracks {T_i}, label set y

for each track T do
    Extract segment features h_i^(0) from mel/chroma windows
    Build graph G = (V, E) (temporal + similarity edges)
end for

for epoch = 1 to E do
    for each graph G in batch do
        for layer l = 0 to L−1 do
            h_i^(l+1) ← GraphSAGE(h_i^(l), N(i))
        end for
        g ← MEANPOOL({h_i^(L)})
        L ← BCE(y, σ(Wg))
        Update GNN parameters
    end for
end for
```

### Algorithm 3: Task 3 — GNN–BERT Fusion for Context Understanding

```
Require: Paired data (G, X_text, y)

H_text ← BERT(X_text), t ← H_text[CLS]
g ← GNN_Readout(G)
z ← CrossAttention(g, H_text)   # or CONCAT(g, t)
ŷ ← σ(Wz + b)
L ← L_tags + α L_emotion
Backprop through fusion, GNN, and (partial) BERT
```

### Algorithm 4: Task 4 — Contrastive GNN–BERT (MusicCaps)

```
Require: Paired (G_i, caption_i), temperature τ, batch size N

g_i ← Normalize(GNN(G_i))
t_i ← Normalize(BERT_CLS(caption_i))
for each training step do
    Compute similarity matrix S_ij = g_i^T t_j / τ
    L_NCE ← −(1/N) Σ_i log( exp(S_ii) / Σ_j exp(S_ij) )
    Update encoders to minimize L_NCE
end for
Evaluate retrieval R@1, R@5, R@10 on held-out pairs
```

---

## 8. Baseline Models for Comparison

Students must compare against **at least two baselines**:

- **B1:** Majority-class / random tag predictor
- **B2:** CNN on mel-spectrogram (no graph, no text)
- **B3:** BERT-only (Task 1)
- **B4 (optional):** PCA + MLP on hand-crafted audio features

### Table 3: Illustrative performance comparison (replace with your experimental results)

| Model | Macro-F1 | AUC-PR | MAE (emotion) | R@5 (retrieval) |
|---|---|---|---|---|
| Random tags | 0.05 | 0.12 | – | 0.02 |
| CNN mel-spec | 0.41 | 0.38 | 1.25 | – |
| Task 1: BERT-only | 0.48 | 0.44 | – | – |
| Task 2: GNN-only | 0.52 | 0.47 | 1.10 | – |
| Task 3: GNN–BERT | 0.61 | 0.55 | 0.92 | – |
| Task 4: Contrastive | 0.55 | 0.50 | – | 0.38 |

---

## 9. Judging Rubric

### Marks distribution (100 total)

| Component | Marks | Notes |
|---|---|---|
| Task 1 (Easy) | 18 | BERT tags + F1 report |
| Task 2 (Medium) | 22 | GNN graph + CNN baseline |
| Task 3 (Hard) | 22 | Fusion + ablations |
| Task 4 (Advanced) | 18 | Contrastive retrieval (optional → bonus) |
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

## 10. Final Submission Requirements

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