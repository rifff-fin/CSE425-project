from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from evaluate import evaluate_tagging, evaluate_retrieval, evaluate_regression
from fma_dataset import load_fma_label_names, load_manifest_label_names
from fma_paired_dataset import FMAPairedDataset
from train import FusionTrainingModel

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a real paired Task 3 or Task 4 manifest on test.json.")
    parser.add_argument("--task", choices=["task3", "task4"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()

def collate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "graph": Batch.from_data_list([row["graph"] for row in rows]),
        "texts": [row["texts"][0] for row in rows],
        "tag_labels": torch.stack([row["tag_labels"] for row in rows]),
        "valence": torch.stack([row["valence"] for row in rows]),
        "arousal": torch.stack([row["arousal"] for row in rows]),
        "has_emotion": torch.stack([row["has_emotion"] for row in rows]),
    }

def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output = args.output if args.output.is_absolute() else root / args.output
    labels = load_fma_label_names(manifest_root) if args.task == "task3" else load_manifest_label_names(manifest_root)
    dataset = FMAPairedDataset(manifest_root / "test.json", labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = FusionTrainingModel(args.task, num_tags=len(labels))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    all_tags = []
    all_targets = []
    graph_embeddings = []
    text_embeddings = []
    valence_pred = []
    valence_true = []
    arousal_pred = []
    arousal_true = []
    with torch.no_grad():
        for batch in loader:
            graph_embedding, hidden_text = model.encode(batch["graph"], batch["texts"])
            graph_embeddings.append(F.normalize(model.head.g_proj(graph_embedding), dim=-1))
            text_embeddings.append(F.normalize(model.head.text_proj(hidden_text.mean(dim=1)), dim=-1))
            all_targets.append(batch["tag_labels"])
            if args.task == "task3":
                logits, vp, ap = model.head(graph_embedding, hidden_text)
                all_tags.append(logits)
                mask = batch["has_emotion"]
                if mask.any():
                    valence_pred.append(vp[mask]); valence_true.append(batch["valence"][mask])
                    arousal_pred.append(ap[mask]); arousal_true.append(batch["arousal"][mask])
    split_counts = {
        split: len(json.loads((manifest_root / f"{split}.json").read_text(encoding="utf-8")))
        for split in ("train", "val", "test")
    }
    result = {
        "task": args.task,
        "dataset": "MusicCaps verified audio-caption pairs" if args.task == "task4" else "FMA paired real sample",
        "split": "test",
        "train_samples": split_counts["train"],
        "validation_samples": split_counts["val"],
        "samples": len(dataset),
        "labels": labels,
    }
    if args.task == "task3":
        result.update(evaluate_tagging(torch.cat(all_tags), torch.cat(all_targets)))
        if valence_true:
            result["valence"] = evaluate_regression(torch.cat(valence_pred), torch.cat(valence_true))
            result["arousal"] = evaluate_regression(torch.cat(arousal_pred), torch.cat(arousal_true))
        else:
            result["emotion_metrics"] = "not_available_no_emotion_targets"
    else:
        graph_matrix = torch.cat(graph_embeddings) @ torch.cat(text_embeddings).T
        text_matrix = graph_matrix.T
        result["caption_to_audio"] = evaluate_retrieval(text_matrix, k_values=(1, 5, 10))
        result["audio_to_caption"] = evaluate_retrieval(graph_matrix, k_values=(1, 5, 10))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
