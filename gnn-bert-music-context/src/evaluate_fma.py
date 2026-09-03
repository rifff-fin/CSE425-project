from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from evaluate import evaluate_tagging
from fma_dataset import FMAGraphDataset, load_fma_label_names
from gnn_model import MusicGNNEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN on the FMA test split.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/fma_task2_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else project_root / args.manifest_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    output = args.output if args.output.is_absolute() else project_root / args.output

    label_names = load_fma_label_names(manifest_root)
    dataset = FMAGraphDataset(manifest_root / "test.json", label_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MusicGNNEncoder(in_channels=32, hidden_dim=128, num_layers=2, model_type="sage")
    model.classifier = nn.Linear(128, len(label_names))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    logits = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            embedding = model(batch["graph"])
            logits.append(model.classifier(embedding))
            targets.append(batch["tags"])

    scores = evaluate_tagging(torch.cat(logits), torch.cat(targets))
    result = {"task": "task2", "split": "test", "samples": len(dataset), "labels": label_names, **scores}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
