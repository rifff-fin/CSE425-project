# ভিডিও Demonstration এবং Submission Guide — বাংলা Dub Script
## Project documentation এবং submission location
Repository-এর commands এবং বর্তমান implementation status আছে [`README.md`](README.md)-এ। Written report-এর checklist আছে [`report/final_report.md`](report/final_report.md)-এ। ভিডিওতে implemented source files, prepared evaluation outputs, final report এবং demo notebook দেখাবেন। Project ZIP LMS-এর upload field-এ আলাদাভাবে submit করবেন; video record করার সময় ZIP তৈরি বা verify করার দরকার নেই এবং ZIP YouTube-এ upload করবেন না।

## ভিডিওতে যা অবশ্যই demonstrate করতে হবে
Screen এবং microphone record করুন। Terminal commands, গুরুত্বপূর্ণ project files, বাস্তব outputs এবং অন্তত একটি audio result দেখান। যে result screen-এ দেখানো বা record করা হয়নি, সেটি claim করবেন না।

### প্রস্তাবিত ৮–১২ মিনিটের recording flow
1. **Introduction — ৩০ সেকেন্ড**
   - Project name, group name এবং GitHub repository URL দেখান।
   - কোন task demonstrate করছেন এবং Task 4 optional কি না তা বলুন।
2. **Project structure — ৪৫ সেকেন্ড**
   - `README.md`, `requirements.txt`, `config.yaml`, `src/`, `notebooks/`, `results/` এবং `report/` দেখান।
3. **Environment এবং installation — ১ মিনিট**
   - নিচের commands চালিয়ে dependencies installed আছে কি না দেখান।
4. **Audio preprocessing এবং graph creation — ১–২ মিনিট**
   - Audio থেকে feature এবং graph তৈরি হওয়ার flow দেখান।
   - Nodes এবং edges-সহ graph বা notebook visualization দেখান।
5. **Model demonstration — ২–৩ মিনিট**
   - Demo notebook অথবা prepared inference command চালান।
   - BERT text representation, GNN audio representation, fusion এবং prediction বুঝিয়ে বলুন।
6. **Results এবং evaluation — ১–২ মিনিট**
   - JSON metrics, plots, retrieval examples এবং report tables খুলে দেখান।
   - প্রযোজ্য ক্ষেত্রে Macro-F1, AUC-PR, MAE এবং Recall@K ব্যাখ্যা করুন।
7. **Report এবং submission link — ৩০ সেকেন্ড**
   - Final report দেখান এবং demonstrated outputs অন্তর্ভুক্ত আছে কি না নিশ্চিত করুন।
   - YouTube URL দেখান এবং visibility Public বা Unlisted কি না verify করুন।

## ব্যবহার করার PowerShell commands
```powershell
Set-Location 'D:\425 project\gnn-bert-music-context'
python --version
# Environment একবারই তৈরি করবেন। .venv থাকলে এই line skip করুন।
if (-not (Test-Path '.venv\Scripts\python.exe')) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Download বা training-এর আগে preflight checks চালান।
Get-ChildItem
Get-ChildItem src
Get-ChildItem notebooks
Get-ChildItem results
Get-ChildItem data\splits\musiccaps
Get-ChildItem checkpoints\musiccaps_task4_95
Test-Path 'data\splits\musiccaps\test.json'
Test-Path 'checkpoints\musiccaps_task4_95\task4_epoch_1.pt'

# Verified local 95-pair MusicCaps subset ব্যবহার করুন।
# এটি reproducible এবং নতুন YouTube download প্রয়োজন হয় না।
python src\evaluate_fma_multimodal.py --task task4 --manifest-root data\splits\musiccaps --checkpoint checkpoints\musiccaps_task4_95\task4_epoch_1.pt --output results\video_task4.json
# Optional: পরিষ্কার নামের folder-এ এক epoch train করতে পারেন।
python src\train.py --task task4 --manifest-root data\splits\musiccaps --epochs 1 --batch-size 4 --lr 1e-4 --checkpoint-dir checkpoints\video_task4 --history-output results\video_task4_history.json
python src\evaluate_fma_multimodal.py --task task4 --manifest-root data\splits\musiccaps --checkpoint checkpoints\video_task4\task4_epoch_1.pt --output results\video_task4_trained_metrics.json
# Local MusicCaps audio থাকলে video-তে একটি audio file চালান।
$audio = Get-ChildItem 'data\raw\musiccaps\audio\*.wav' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($audio) { Start-Process $audio.FullName; Write-Output "Playing $($audio.FullName)" } else { Write-Output 'No local WAV file; show the graph/notebook instead.' }

# Executed demonstration notebook খুলুন।
jupyter notebook notebooks\demo_context.ipynb
# Project ZIP LMS-এ আলাদাভাবে upload হবে। Video demonstration-এর সময় ZIP তৈরি বা verify করবেন না।
```

### Optional download এবং preprocessing commands
শুধুমাত্র নতুন local subset তৈরি করতে চাইলে এগুলো চালাবেন। এগুলোর জন্য `data/raw/musiccaps/musiccaps-public.csv`, `yt-dlp`, internet access এবং FFmpeg দরকার। Downloader কিছু unavailable YouTube source skip করতে পারে।

```powershell
python src\download_musiccaps.py --limit 10
python src\preprocess_musiccaps.py --limit 10
```

Preprocessing-এর জন্য কমপক্ষে তিনটি successfully downloaded audio-caption pair দরকার। তিনটির কম পাওয়া গেলে verified local `data\splits\musiccaps` manifests ব্যবহার করুন। সব requested clip download হয়েছে—এমন claim করবেন না।

> **গুরুত্বপূর্ণ:** Repository-তে verified 95-pair MusicCaps subset এবং one-epoch checkpoint আছে। Video demonstration-এর জন্য নতুন download প্রয়োজন নেই।

Verified checkpoint এবং results আগে থেকেই থাকলে recording-এর সময় নতুন করে training না করে সেগুলো দেখান। Training বা download দীর্ঘ সময় নিতে পারে, তাই YouTube demonstration reliable রাখুন।
## Speaking script — চারটি model task
Monitor 2-এ এই teleprompter রাখুন এবং Monitor 1 record করুন। Course specification-এ ঠিক চারটি model task আছে। Task 4 advanced optional rubric task, এবং এই project-এ এটি implement করা হয়েছে। Notebook-এর end-to-end stages technical flow অনুযায়ী Task 2 এবং Task 3-এ ভাগ করা হয়েছে: Task 2-এ record load, audio feature, graph construction এবং graph visualization; Task 3-এ fusion checkpoint inference এবং prediction interpretation। ZIP upload video থেকে আলাদা; এটি কোনো পঞ্চম task নয়। প্রতিটি section-এ আগে commands চালান, তারপর যে files দেখাতে বলা হয়েছে সেগুলো দেখান এবং screen-এ visible থাকা অবস্থায় voiceover পড়ুন। Display বা record না করা কোনো result claim করবেন না।

### Opening এবং setup
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

**Screen-এ দেখানোর checklist**

- `README.md`: project title, Group 27 table এবং চারটি supported model task।
- Explorer অথবা PowerShell: `src`, `data`, `notebooks`, `results` এবং `report`।
- `config.yaml`: 22,050 Hz, 128 mel bins, 12 chroma bins, five-second segments এবং `distilbert-base-uncased`।

**Voiceover — checklist visible থাকা অবস্থায় বলুন**

“Hello। আমরা Group 27, GNN-BERT Music Context project demonstrate করছি। এই project BERT language representation এবং audio structure graph একসঙ্গে ব্যবহার করে। আমরা চারটি model task demonstrate করব: Task 1, BERT tag baseline; Task 2, audio graph model; Task 3, GNN-BERT fusion; এবং Task 4, advanced MusicCaps retrieval। Notebook-এর end-to-end stages technical ভাবে Task 2 এবং Task 3-এর মধ্যে দেখানো হবে। আমরা project environment এবং saved checkpoint ব্যবহার করছি এবং placeholder নয়, বাস্তব output দেখাব।”

### Task 1 — music tag understanding-এর জন্য BERT baseline
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

**Screen-এ এই order-এ দেখান**

- `src\bert_encoder.py`: tokenizer, DistilBERT encoder এবং pooled text output।
- `src\train.py`: Task 1 dataset, model, loss এবং training branch।
- `data\splits\musiccaps_task1_proxy\proxy_report.json`, তারপর `train.json`, `val.json` এবং `test.json`।
- `results\expanded_fma_task1_metrics.json`: held-out Task 1 test metrics।
- `results\task1_prediction_examples.json`: example predictions।
- `results\plots\task1_f1_training_curve.png`: Task 1 F1 training curve।

**Voiceover Teleprompter**

- **Task 1 description visible থাকলে বলুন:**
  “এখন আমি Task 1 শুরু করছি, যা music tag understanding-এর BERT baseline। এই task-এ graph structure ব্যবহার করা হয় না; শুধু textual music context ব্যবহার করা হয়। Model caption, tag অথবা lyric পেয়ে একই track-এর একাধিক সম্ভাব্য music tag predict করে।”

- **`src\bert_encoder.py` visible থাকলে বলুন:**
  “এই file-এ text encoder implement করা হয়েছে। প্রথমে tokenizer sentence-কে token ID-তে রূপান্তর করে, special token যোগ করে এবং attention mask তৈরি করে। এরপর DistilBERT token sequence পড়ে contextual hidden state তৈরি করে। Pooled text representation input context-এর summary হিসেবে classifier-এ যায়। এই stage-এ Task 1 শুধু text ব্যবহার করে; audio graph ব্যবহার করে না।”

- **Classification head এবং forward code visible থাকলে বলুন:**
  “Classification head pooled DistilBERT representation থেকে প্রতিটি tag-এর জন্য একটি logit তৈরি করে। এটি multi-label problem, তাই logits independent; একটি track-এর একাধিক tag থাকতে পারে। Inference-এর সময় sigmoid প্রতিটি logit-কে zero থেকে one-এর মধ্যে probability-তে রূপান্তর করে। Training objective হলো সব tag output-এর উপর binary cross-entropy।”

- **`src\train.py` visible থাকলে বলুন:**
  “এটি Task 1-এর training branch। এটি text example এবং label load করে, token ID ও attention mask BERT encoder-এ পাঠায়, binary cross-entropy loss গণনা করে এবং classifier ও নির্বাচিত BERT parameter update করে। এই branch-এ graph builder বা GNN call হয় না; তাই এটি Task 2 এবং Task 3-এর জন্য text-only baseline।”

- **Proxy split files visible থাকলে বলুন:**
  “এগুলো reproducible Task 1 proxy split। MusicCaps caption এবং tag name-এর deterministic lexical match থেকে proxy label তৈরি হয়েছে। এগুলো pipeline demonstration-এর জন্য useful, কিন্তু independent human annotation নয়। তাই result-টিকে caption-to-tag proxy benchmark হিসেবে সৎভাবে বর্ণনা করছি।”

- **Task 1 metrics visible থাকলে বলুন:**
  “এই JSON-এ held-out Task 1 test metrics আছে। Macro-F1 প্রতিটি tag-এর F1 হিসাব করে average করে, তাই প্রতিটি tag সমান গুরুত্ব পায়। Micro-F1 সব tag decision-এর true positive, false positive এবং false negative একসঙ্গে pool করে। AUC-PR precision-recall curve-এর summary, যা multi-label এবং imbalanced data-তে useful। আমি displayed file-এর বাস্তব value পড়ছি; placeholder value নয়।”

- **Prediction examples visible থাকলে বলুন:**
  “এই file-এ example input, target tag, predicted tag এবং prediction score আছে। Text-only classifier বাস্তবে কী predict করছে, এটি তার qualitative check।”

- **F1 curve visible থাকলে বলুন:**
  “এই plot training চলাকালে F1 metric কীভাবে পরিবর্তিত হয়েছে তা দেখায়। Implementation, proxy split, test JSON, prediction examples এবং training curve একসঙ্গে Task 1-এর সম্পূর্ণ evidence দেয়। এই BERT-only result Task 2-এর audio-only GNN এবং Task 3-এর multimodal fusion-এর text baseline।”

### Task 2 — music structure graph-এর উপর GNN
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

**Screen-এ এই order-এ দেখান**

- `src\audio_features.py`: audio resampling, feature extraction এবং segment preparation।
- `src\graph_builder.py`: audio segment feature, temporal edge এবং similarity edge।
- `src\gnn_model.py`: GraphSAGE/GAT layer, message passing, mean pooling এবং prediction head।
- `notebooks\demo_context.ipynb`: real demonstration-এর record-loading এবং preprocessing cell।
- Notebook output: selected record এবং metadata text/context।
- Notebook output: prepared audio feature, segment count এবং graph node/edge count।
- `data\processed\graphs`: processed graph sample; সম্ভব হলে অন্তত ২০টি `.pt` বা `.json` sample দেখান।
- Notebook graph visualization: node, edge, title এবং readable layout।
- `results\expanded_fma_task2_metrics.json` এবং `results\task2_gnn_3epoch_metrics.json`: Task 2 test metrics।
- `results\task2_gnn_cnn_comparison.json`: numerical GNN/CNN comparison।
- `results\plots\task2_gnn_cnn_comparison.png`: visual comparison।

**Voiceover Teleprompter**

- **Task 2 description visible থাকলে বলুন:**
  “Task 2 হলো audio-only GNN model। Task 1-এর মতো এখানে caption বা BERT ব্যবহার করছি না। Audio feature থেকে একটি graph তৈরি করে genre অথবা music tag predict করা হচ্ছে।”

- **`src\audio_features.py` visible থাকলে বলুন:**
  “এটি audio-feature preparation stage। Pipeline audio resample করে এবং mel অথবা chroma-এর মতো segment-level descriptor তৈরি করে। এই prepared feature-গুলোই graph node-এর input।”

- **`src\graph_builder.py` visible থাকলে বলুন:**
  “এই file prepared segment descriptor নিয়ে music graph তৈরি করে। প্রতিটি audio segment একটি node। পাশের segment-এর মধ্যে temporal edge এবং feature similarity যথেষ্ট হলে similarity edge তৈরি হয়।”

- **`src\gnn_model.py` visible থাকলে বলুন:**
  “এই file graph encoder implement করে। GraphSAGE প্রতিটি node-এর বর্তমান representation এবং neighbour node-এর mean representation একসঙ্গে ব্যবহার করে update করে। Message passing-এর পরে mean pooling সব node representation-কে একটি graph-level vector-এ রূপান্তর করে। Prediction head সেই vector থেকে tag বা genre output দেয়।”

- **Notebook setup এবং record-loading cell visible থাকলে বলুন:**
  “এখন আমি real demonstration record ব্যবহার করছি। Notebook held-out record এবং তার metadata load করছে; example বানিয়ে দেখানো হচ্ছে না। এই setup Task 2-এর অংশ, কারণ audio record-কে আগে audio-only GNN-এর graph input-এ রূপান্তর করতে হয়।”

- **Notebook record output এবং feature/graph source files visible থাকলে বলুন:**
  “এই reproducible inference example-এ notebook verified processed graph ব্যবহার করছে। Source file-গুলো দেখায় কীভাবে audio feature প্রস্তুত হয় এবং segment ও relationship কীভাবে graph node ও edge হয়। Stored graph-টিই এখানে GNN input হিসেবে ব্যবহৃত হচ্ছে।”

- **`data\processed\graphs` visible থাকলে বলুন:**
  “এগুলো Task 2-এর processed graph artifact। প্রতিটি file-এ GNN-এ পাঠানো node feature matrix এবং edge index থাকে। Node হলো audio segment এবং edge হলো temporal অথবা feature-similarity relationship।”

- **Graph sample অথবা notebook visualization visible থাকলে বলুন:**
  “Plot-এর point-গুলো graph node এবং connecting line-গুলো graph edge। Visualization নিশ্চিত করে যে model শুধু flat audio vector পাচ্ছে না; audio segment-এর সঙ্গে তাদের relationship-ও পাচ্ছে।”

- **Task 2 JSON metrics visible থাকলে বলুন:**
  “এই JSON file-গুলো held-out Task 2 test metrics ধারণ করে। আমি saved evaluation file-এর value দেখাচ্ছি, কোনো placeholder নয়।”

- **GNN/CNN comparison visible থাকলে বলুন:**
  “এই comparison required CNN mel-spectrogram baseline এবং audio graph model-এর performance তুলনা করে। JSON-এ numerical comparison এবং PNG-তে visual comparison আছে। অর্থাৎ Task 2 পরীক্ষা করছে relational graph structure conventional spectrogram CNN-এর অতিরিক্ত useful information দেয় কি না।”

### Task 3 — multi-context understanding-এর জন্য GNN-BERT fusion
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

**Screen-এ এই order-এ দেখান**

- `src\fusion_model.py`: graph projection, text projection, cross-attention, tag head, valence head এবং arousal head।
- `notebooks\demo_context.ipynb`: real Task 3 checkpoint inference code এবং printed tensor shape।
- Notebook output: text context, reference tag, graph node count এবং graph edge count।
- Checkpoint-loading এবং inference cell, যেখানে evaluation mode ও `torch.no_grad()` দেখা যায়।
- Notebook output: text representation shape `(1, 28, 768)`, graph representation shape `(1, 128)` এবং tag-logit shape `(1, 26)`, যদি এই value display হয়।
- Sigmoid-এর পরে ranked top predicted tag, probability এবং reference tag-এর comparison।
- `results\expanded_fma_task3_metrics.json` এবং `results\task3_fma_submission_metrics.json`: dataset, split, Macro-F1, Micro-F1 এবং AUC-PR।
- `results\task3_analysis.json`: ablation এবং analysis evidence।
- `results\plots\task3_tsne.png` এবং `results\plots\task3_deam_tsne.png`: Task 3 representation visualization।
- `results\task3_caption_case_studies.json`: graph-caption alignment-এর তিনটি case study।

**Voiceover Teleprompter**

- **Task 3 description visible থাকলে বলুন:**
  “Task 3 হলো hard GNN-BERT fusion task। এটি audio graph-এর structural information এবং caption বা metadata-এর semantic information একসঙ্গে নিয়ে multi-context music label predict করে।”

- **`src\fusion_model.py` visible থাকলে বলুন:**
  “GNN graph representation তৈরি করে এবং BERT contextual text hidden state তৈরি করে। Fusion model দুটিকে compatible dimension-এ project করে এবং prediction-এর আগে cross-attention প্রয়োগ করে। Tag head একাধিক tag predict করে; emotion target available থাকলে valence এবং arousal head auxiliary output দেয়।”

- **Checkpoint-loading ও inference cell visible থাকলে বলুন:**
  “এই cell real trained Task 3 checkpoint-কে `FusionTrainingModel`-এ load করে, model-কে evaluation mode-এ রাখে এবং `torch.no_grad()` দিয়ে gradient বন্ধ করে। `encode` call text hidden state এবং graph embedding তৈরি করে। এরপর fusion head tag logits, valence prediction এবং arousal prediction return করে।”

- **Notebook input এবং printed context visible থাকলে বলুন:**
  “Notebook এখন একটি track-এর actual text context এবং reference tag দেখাচ্ছে। Graph size-ও print করা হয়েছে। এই example-এ ১১টি graph node audio segment এবং ১১০টি edge stored relationship represent করছে।”

- **Tensor shape visible থাকলে বলুন:**
  “Text representation-এর shape `(1, 28, 768)`—একটি example, ২৮টি text position এবং প্রতিটিতে ৭৬৮টি BERT feature। Graph representation-এর shape `(1, 128)`—একটি graph-level audio vector-এ ১২৮টি feature। Tag logits-এর shape `(1, 26)`—অর্থাৎ ২৬টি tag-এর জন্য একটি করে score।”

- **Prediction decode visible থাকলে বলুন:**
  “Notebook tag logits-এর উপর sigmoid প্রয়োগ করে, probability-গুলো descending order-এ sort করে এবং highest-scoring tag বেছে নেয়। এতে raw model score interpretable ranked tag probability-তে পরিণত হয়, যা reference tag-এর সঙ্গে তুলনা করা যায়।”

- **Task 3 JSON metrics visible থাকলে বলুন:**
  “এই JSON file-গুলো held-out Task 3 evaluation result ধারণ করে। Macro-F1 প্রতিটি tag-এর F1 average করে, Micro-F1 সব tag decision pool করে এবং AUC-PR precision-recall summary দেয়। Displayed result-এ emotion target থাকলেই শুধু MAE বা R-squared নিয়ে আলোচনা করব।”

- **Analysis, t-SNE এবং case study visible থাকলে বলুন:**
  “এই file-গুলো numerical result-এর সঙ্গে ablation evidence, representation visualization এবং তিনটি graph-caption case study যোগ করে। এগুলো BERT branch, GNN branch এবং fusion representation একসঙ্গে কীভাবে বিশ্লেষণ করা হয়েছে তা দেখায়।”

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

**Screen-এ এই order-এ দেখান**

- `src\contrastive.py`: normalized graph/text embedding এবং contrastive objective।
- `src\evaluate_fma_multimodal.py`: test manifest load, checkpoint load, similarity matrix এবং retrieval evaluation।
- `data\splits\musiccaps\train.json`, `val.json`, `test.json` এবং `alignment_report.json`।
- `checkpoints\musiccaps_task4_95\task4_epoch_1.pt` এবং successful `Test-Path` output।
- `results\video_task4.json`: MusicCaps test sample count এবং দুই দিকের retrieval result।
- `results\retrieval_examples\musiccaps_examples.json`: caption, query ID, top-three retrieved ID এবং similarity value।
- `results\plots\musiccaps_task4_loss_curve.png`: Task 4 contrastive training curve, available থাকলে।

**Voiceover Teleprompter**

- **Task 4 description visible থাকলে বলুন:**
  “Task 4 হলো advanced MusicCaps cross-modal alignment task। এটি এমন একটি shared embedding space শেখে যেখানে matching audio graph এবং caption কাছাকাছি থাকে, আর non-matching pair দূরে থাকে।”

- **`src\contrastive.py` visible থাকলে বলুন:**
  “Contrastive module graph এবং text embedding normalize করে এবং InfoNCE objective প্রয়োগ করে। একটি batch-এ matching graph-caption pair positive pair এবং অন্য pair-গুলো negative।”

- **`src\evaluate_fma_multimodal.py` visible থাকলে বলুন:**
  “এই evaluation script held-out manifest এবং saved checkpoint load করে, graph ও caption encode করে, similarity matrix তৈরি করে এবং দুই দিকের retrieval calculate করে।”

- **MusicCaps manifest এবং checkpoint visible থাকলে বলুন:**
  “এগুলো reproducible MusicCaps train, validation এবং test manifest। এটি prepared evaluation-এর saved checkpoint, তাই recording-এর সময় নতুন করে দীর্ঘ training শুরু করছি না।”

- **`results\video_task4.json` visible থাকলে বলুন:**
  “এই JSON held-out test split-এ caption-to-audio এবং audio-to-caption Recall@1, Recall@5 এবং Recall@10 report করে। Recall@K পরীক্ষা করে correct matching item top K result-এর মধ্যে আছে কি না।”

- **Retrieval example এবং loss curve visible থাকলে বলুন:**
  “এই example-এ query caption, query audio ID, retrieved audio ID, rank এবং similarity value দেখা যায়। Loss curve training-এর সময় contrastive objective কীভাবে পরিবর্তিত হয়েছে তা দেখায়। Similarity value দিয়ে candidate rank হয়, তারপর correct paired item rank one, five এবং ten-এর মধ্যে আছে কি না পরীক্ষা হয়। Task 4 rubric-এ optional, কিন্তু এই project demonstration-এ এটি অন্তর্ভুক্ত।”

## Closing report এবং video-link checks
এগুলো Task 1–4 শেষ হওয়ার পরের closing activity; কোনো অতিরিক্ত model task নয়। Recording-এর সময় ZIP তৈরি বা verify করবেন না; এটি LMS field-এ আলাদাভাবে upload করবেন।

**Screen-এ দেখানোর checklist**

- `report\final_report.pdf`, যেখানে Task 1–4-এর result এবং discussion আছে।
- আলাদা YouTube URL field এবং তার Public/Unlisted visibility, যদি recording-এ দেখানো হয়।

**Voiceover Teleprompter**

- “চারটি model task সম্পূর্ণ হয়েছে। Report-এ Task 1, Task 2, Task 3 এবং optional Task 4-এর evidence একত্র করা হয়েছে; এই closing check কোনো পঞ্চম task নয়।”
- “Project ZIP LMS upload field-এ আলাদাভাবে submit করা হবে; এটি এই video demonstration-এর অংশ নয়।”
- “YouTube URL video-link field-এ আলাদাভাবে submit করা হবে। Visibility Public অথবা Unlisted রাখা হয়েছে এবং submission-এর আগে link test হয়েছে। ধন্যবাদ।”

## Final checklist
- [ ] YouTube video-তে clear audio এবং readable terminal output আছে।
- [ ] Incognito/private browser-এ video URL কাজ করে।
- [ ] Video visibility instructor-এর requirement অনুযায়ী Public অথবা Unlisted।
- [ ] `report/final_report.tex` থেকে final report PDF generate করা হয়েছে এবং প্রয়োজন হলে included আছে।
- [ ] Demo notebook এবং results included অথবা linked আছে।
- [ ] Local evaluation command সফলভাবে complete হয়েছে এবং `results/video_task4.json` তৈরি হয়েছে।
- [ ] Display করা প্রতিটি metric-এর dataset, split এবং checkpoint label করা আছে।
- [ ] YouTube link form-এর video field-এ paste করা হয়েছে।
- [ ] Project ZIP প্রয়োজন হলে “Zip files of codes” upload field-এ আলাদাভাবে upload করা হয়েছে।
