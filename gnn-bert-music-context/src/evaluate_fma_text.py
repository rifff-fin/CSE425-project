from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from evaluate import evaluate_tagging
from fma_dataset import load_fma_label_names
from fma_text_dataset import FMATextDataset
from bert_encoder import BERTTextEncoder

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the real FMA Task 1 text classifier.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/fma_task1_text_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else project_root / args.manifest_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    output = args.output if args.output.is_absolute() else project_root / args.output
    labels = load_fma_label_names(manifest_root)
    dataset = FMATextDataset(manifest_root / "test.json", labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda rows: {"texts": [r["texts"][0] for r in rows], "tags": torch.stack([r["tags"] for r in rows])})
    model = BERTTextEncoder(model_name="distilbert-base-uncased")
    model.classifier = nn.Linear(768, len(labels))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    logits, targets, track_ids, texts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            _, pooled = model(batch["texts"])
            logits.append(model.classifier(pooled))
            targets.append(batch["tags"])
            texts.extend(batch["texts"])
            track_ids.extend([row["track_id"] for row in dataset.records[len(track_ids):len(track_ids) + len(batch["texts"])]] )
    all_logits = torch.cat(logits)
    all_targets = torch.cat(targets)
    scores = evaluate_tagging(all_logits, all_targets)
    probabilities = torch.sigmoid(all_logits)
    examples = []
    for index in range(min(args.examples, len(dataset))):
        predicted = [labels[i] for i in torch.where(probabilities[index] >= 0.5)[0].tolist()]
        ranked = torch.argsort(probabilities[index], descending=True)[:5].tolist()
        examples.append({
            "track_id": track_ids[index],
            "text": texts[index],
            "true_tags": [labels[i] for i in torch.where(all_targets[index] > 0)[0].tolist()],
            "predicted_tags_threshold_0.5": predicted,
            "top_5_predictions": [{"label": labels[i], "score": float(probabilities[index, i])} for i in ranked],
        })
    result = {"task": "task1", "split": "test", "samples": len(dataset), "labels": labels, "examples": examples, **scores}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
