from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from evaluate import evaluate_tagging
from bert_encoder import BERTTextEncoder
from musiccaps_proxy_dataset import MusicCapsProxyTextDataset, load_proxy_label_names

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 1 DistilBERT on MusicCaps caption-to-tag proxy labels.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits/musiccaps_task1_proxy"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/musiccaps_task1_proxy_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output = args.output if args.output.is_absolute() else root / args.output
    labels = load_proxy_label_names(manifest_root)
    dataset = MusicCapsProxyTextDataset(manifest_root / "test.json", labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda rows: {"texts": [r["texts"][0] for r in rows], "tags": torch.stack([r["tags"] for r in rows])})
    model = BERTTextEncoder(model_name="distilbert-base-uncased")
    model.classifier = nn.Linear(768, len(labels))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    logits = []
    targets = []
    texts = []
    track_ids = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            _, pooled = model(batch["texts"]); logits.append(model.classifier(pooled)); targets.append(batch["tags"]); texts.extend(batch["texts"])
            track_ids.extend([row["track_id"] for row in dataset.records[offset:offset + len(batch["texts"])]])
            offset += len(batch["texts"])
    all_logits, all_targets = torch.cat(logits), torch.cat(targets)
    scores = evaluate_tagging(all_logits, all_targets); probabilities = torch.sigmoid(all_logits)
    examples = []
    for i in range(min(args.examples, len(dataset))):
        ranked = torch.argsort(probabilities[i], descending=True)[:5].tolist()
        examples.append({"track_id": track_ids[i], "caption": texts[i], "proxy_tags": [labels[j] for j in torch.where(all_targets[i] > 0)[0].tolist()], "predicted_tags_threshold_0.5": [labels[j] for j in torch.where(probabilities[i] >= 0.5)[0].tolist()], "top_5_predictions": [{"label": labels[j], "score": float(probabilities[i, j])} for j in ranked]})
    result = {"task": "task1_caption_to_tag_proxy", "dataset": "MusicCaps verified caption subset", "label_count": len(labels), "split": "test", "samples": len(dataset), "labels": labels, "proxy_labeling": "deterministic lexical caption matches; not independent human ground truth", "examples": examples, **scores}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
