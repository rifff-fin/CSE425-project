from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot controlled Task 2 GNN/CNN test metrics.")
    parser.add_argument("--comparison", type=Path, default=Path("results/task2_gnn_cnn_comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("results/plots/task2_gnn_cnn_comparison.png"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    comparison = args.comparison if args.comparison.is_absolute() else root / args.comparison
    output = args.output if args.output.is_absolute() else root / args.output
    rows = json.loads(comparison.read_text(encoding="utf-8"))["comparison"]
    names = ["GraphSAGE" if row["model"] == "task2" else "Mel CNN" for row in rows]
    metrics = ["macro_f1", "micro_f1", "auc_pr"]
    labels = ["Macro-F1", "Micro-F1", "AUC-PR"]
    values = [[row[metric] for row in rows] for metric in metrics]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    width = 0.24
    x = list(range(len(names)))
    for offset, (label, metric_values) in enumerate(zip(labels, values)):
        axis.bar([item + (offset - 1) * width for item in x], metric_values, width=width, label=label)
    axis.set_xticks(x, names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test score")
    axis.set_title("Task 2 controlled GNN/CNN comparison")
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

if __name__ == "__main__":
    main()
