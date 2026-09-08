from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch
from sklearn.manifold import TSNE
from fma_dataset import load_manifest_label_names
from fma_paired_dataset import FMAPairedDataset
from train import FusionTrainingModel

def mood_label(valence: float, arousal: float) -> str:
    if valence >= 5 and arousal >= 5: return "happy-energetic"
    if valence >= 5 and arousal < 5: return "happy-calm"
    if valence < 5 and arousal >= 5: return "sad-energetic"
    return "sad-calm"

def main() -> None:
    parser = argparse.ArgumentParser(description="Create DEAM Task 3 fused-representation t-SNE colored by genre and validated mood.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits/deam_processed"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/plots/task3_deam_tsne.png"))
    parser.add_argument("--metadata-output", type=Path, default=Path("results/task3_deam_tsne_metadata.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output = args.output if args.output.is_absolute() else root / args.output
    metadata_output = args.metadata_output if args.metadata_output.is_absolute() else root / args.metadata_output
    labels = load_manifest_label_names(manifest_root)
    dataset = FMAPairedDataset(manifest_root / "test.json", labels)
    model = FusionTrainingModel("task3", num_tags=len(labels))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")); model.eval()
    fused = []
    genres = []
    moods = []
    ids = []
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            rows = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
            graph = Batch.from_data_list([row["graph"] for row in rows])
            graph_embedding, hidden_text = model.encode(graph, [row["texts"][0] for row in rows])
            fused.append(model.head.fused_representation(graph_embedding, hidden_text).cpu())
            for row in rows:
                record = dataset.records[start + len(genres) - start]
                genres.append(record.get("genre", "unknown") or "unknown")
                moods.append(mood_label(float(record["valence"]), float(record["arousal"])))
                ids.append(record["track_id"])
    points = torch.cat(fused).numpy()
    embedding = TSNE(n_components=2, perplexity=min(5, len(points) - 1), random_state=42, init="random").fit_transform(points)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, values, title in ((axes[0], genres, "DEAM genre"), (axes[1], moods, "DEAM mood from valence/arousal")):
        categories = sorted(set(values)); lookup = {value: index for index, value in enumerate(categories)}
        colors = [lookup[value] for value in values]
        axis.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="tab20", s=60)
        for i, track_id in enumerate(ids): axis.annotate(str(track_id), (embedding[i, 0], embedding[i, 1]), fontsize=7)
        axis.set_title(title); axis.set_xlabel("t-SNE 1"); axis.set_ylabel("t-SNE 2")
    figure.suptitle("Task 3 DEAM fused representation t-SNE")
    figure.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=160); plt.close(figure)
    metadata = {"task": "task3", "dataset": "DEAM verified audio-text-emotion pairs", "samples": len(ids), "plot": str(output.relative_to(root)).replace("\\", "/"), "genre_coloring": "DEAM metadata genre", "mood_coloring": "validated valence/arousal quadrants on the 1-9 scale", "track_ids": ids, "moods": moods}
    metadata_output.parent.mkdir(parents=True, exist_ok=True); metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__": main()
