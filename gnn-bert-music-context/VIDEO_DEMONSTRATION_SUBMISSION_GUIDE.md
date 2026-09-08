# Video Demonstration and Submission Guide
## Submission guideline location
The main project requirements are in [`GUIDELINE.MD`](GUIDELINE.MD). The repository-specific commands and current implementation status are in [`README.md`](README.md). The written-report checklist is in [`report/final_report.md`](report/final_report.md).

`GUIDELINE.MD` requires a source-code repository/ZIP, graph samples, evaluation tables and plots, a final report PDF, and a demo notebook. It does **not** define the LMS/Google Form upload fields for a YouTube link. If the submission form says **“Zip files of codes – Upload 1 supported file – Max 1 GB”**, upload one ZIP of the project code and paste the public/unlisted YouTube URL in the separate video-link field. Do not upload the ZIP to YouTube.

## What the video must demonstrate
Record the screen and microphone. Show the terminal commands, important project files, real outputs, and at least one audio result. Do not claim results that are not displayed or recorded.

### Suggested 8–12 minute recording
1. **Introduction (30 seconds)**
   - Show the project name, group name, and GitHub repository URL.
   - Say which tasks are demonstrated and identify any optional task.
2. **Project structure (45 seconds)**
   - Show `README.md`, `GUIDELINE.MD`, `requirements.txt`, `config.yaml`, `src/`, `notebooks/`, `results/`, and `report/`.
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
7. **Reproducibility and submission package (1 minute)**
   - Show the final ZIP contents and confirm that raw audio, secrets, and virtual environments are excluded.
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
# Create the code ZIP for the LMS upload; exclude raw audio, checkpoints, results, and .venv.
$zip = 'gnn-bert-music-context-code.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path README.md,GUIDELINE.MD,requirements.txt,config.yaml,src,notebooks,report,.gitignore,VIDEO_DEMONSTRATION_SUBMISSION_GUIDE.md -DestinationPath $zip -CompressionLevel Optimal
Get-Item $zip
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

## Speaking script
“Hello, this is Group 27 demonstrating our GNN-BERT Music Context project. The project combines audio graph representations from music segments with BERT representations from captions or metadata. The official requirements are in `GUIDELINE.MD`, and the implementation commands are documented in `README.md`.

First, I am showing the repository structure and the configuration. The audio settings include a 22,050 Hz sample rate, 128 mel bins, 12 chroma bins, and a five-second segment window. Next, the audio preprocessing extracts features and constructs a graph. The graph nodes represent audio segments and the edges represent temporal or feature similarity relationships.

For the text branch, BERT encodes the caption or metadata into contextual embeddings. The GNN processes the audio graph. The fusion model combines both representations to produce music-context predictions. For the advanced retrieval task, the graph and caption embeddings are trained so that a matching audio-caption pair is closer than nonmatching pairs.

Here I am running the demonstration and showing the actual output. These metrics are calculated from the displayed evaluation split; they are not invented example values. I will now show the plots, JSON metrics, retrieval examples, and the report evidence.

Finally, I am creating the code ZIP for the LMS. The YouTube video link will be submitted in the video-link field, and this ZIP will be uploaded in the field labeled ‘Zip files of codes’. The video is set to [Public/Unlisted], and the link has been tested before submission. Thank you.”

## Final checklist
- [ ] YouTube video has clear audio and readable terminal output.
- [ ] Video URL works in an incognito/private browser window.
- [ ] Video visibility is Public or Unlisted, as required by the instructor.
- [ ] ZIP contains source code and documentation and is below 1 GB.
- [ ] ZIP does not contain `.venv`, passwords, private tokens, or unnecessary raw audio.
- [ ] Final report PDF is generated from `report/final_report.tex` and included if required.
- [ ] Demo notebook and results are included or linked as required.
- [ ] The local evaluation command completed successfully and `results/video_task4.json` was created.
- [ ] Any displayed metric is labelled with its dataset, split, and checkpoint.
- [ ] The YouTube link is pasted into the form's video field.
- [ ] The ZIP is uploaded only to the “Zip files of codes” upload field.
