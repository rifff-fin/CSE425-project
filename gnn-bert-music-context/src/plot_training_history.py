from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/validation loss history.")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.history.read_text(encoding="utf-8"))
    rows = payload["history"]
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in rows], marker="o", label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title(f"{payload.get('task', 'training').upper()} training curve")
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()