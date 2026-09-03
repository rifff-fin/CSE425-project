from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from evaluate import evaluate_tagging
from fma_dataset import load_fma_label_names

def load_records(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def targets_from_records(records, labels):
    index = {label.lower(): i for i, label in enumerate(labels)}
    targets = torch.zeros((len(records), len(labels)), dtype=torch.float32)
    for row, record in enumerate(records):
        for tag in record.get("tags", []):
            if str(tag).lower() in index:
                targets[row, index[str(tag).lower()]] = 1.0
    return targets

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a train-frequency majority tag baseline on FMA.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--output", type=Path, default=Path("results/fma_majority_baseline_metrics.json"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = args.manifest_root if args.manifest_root.is_absolute() else project_root / args.manifest_root
    output = args.output if args.output.is_absolute() else project_root / args.output
    labels = load_fma_label_names(root)
    train_targets = targets_from_records(load_records(root / "train.json"), labels)
    test_targets = targets_from_records(load_records(root / "test.json"), labels)
    prevalence = train_targets.mean(dim=0)
    predictions = prevalence.unsqueeze(0).expand_as(test_targets)
    logits = torch.logit(predictions.clamp(1e-6, 1 - 1e-6))
    scores = evaluate_tagging(logits, test_targets)
    result = {"task": "baseline", "baseline": "train_frequency_majority", "split": "test", "train_samples": len(train_targets), "test_samples": len(test_targets), "labels": labels, "train_label_prevalence": {label: float(value) for label, value in zip(labels, prevalence)}, **scores}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
