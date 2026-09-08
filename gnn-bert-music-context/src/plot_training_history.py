from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss and epoch-level classification metrics from a training history.")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.history.read_text(encoding="utf-8"))
    rows = payload["history"]
    epochs = [row["epoch"] for row in rows]
    has_f1 = all("val_macro_f1" in row and "val_micro_f1" in row for row in rows)
    figure, axes = plt.subplots(2 if has_f1 else 1, 1, figsize=(7, 7 if has_f1 else 4), squeeze=False)
    loss_axis = axes[0, 0]
    loss_axis.plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train loss")
    loss_axis.plot(epochs, [row["val_loss"] for row in rows], marker="o", label="validation loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("BCE loss")
    loss_axis.set_title(f"{payload.get('task', 'training').upper()} loss")
    loss_axis.legend()
    if has_f1:
        metric_axis = axes[1, 0]
        metric_axis.plot(epochs, [row["val_macro_f1"] for row in rows], marker="o", label="validation Macro-F1")
        metric_axis.plot(epochs, [row["val_micro_f1"] for row in rows], marker="o", label="validation Micro-F1")
        if all("train_macro_f1" in row for row in rows):
            metric_axis.plot(epochs, [row["train_macro_f1"] for row in rows], marker="x", linestyle="--", label="train Macro-F1")
            metric_axis.plot(epochs, [row["train_micro_f1"] for row in rows], marker="x", linestyle="--", label="train Micro-F1")
        metric_axis.plot(epochs, [row.get("val_auc_pr", 0.0) for row in rows], marker="o", label="validation AUC-PR")
        metric_axis.set_xlabel("Epoch")
        metric_axis.set_ylabel("Score")
        metric_axis.set_ylim(0.0, 1.0)
        metric_axis.set_title(f"{payload.get('task', 'training').upper()} validation metrics")
        metric_axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()