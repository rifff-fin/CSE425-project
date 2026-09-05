from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict
ACTIVE_FILES = {
    "task1": "expanded_fma_task1_metrics.json",
    "task2": "expanded_fma_task2_metrics.json",
    "task3": "expanded_fma_task3_metrics.json",
    "task4": "expanded_fma_task4_metrics.json",
    "majority_baseline": "expanded_fma_majority_baseline_metrics.json",
    "cnn_baseline": "expanded_fma_cnn_baseline_metrics.json",
}
LEGACY_FILES = {
    "fma_task1_text_metrics.json",
    "fma_task2_metrics.json",
    "task3_test_metrics.json",
    "task4_test_metrics.json",
    "fma_majority_baseline_metrics.json",
    "fma_cnn_baseline_metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-run metrics into results/metrics.json.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    return parser.parse_args()


def load_if_available(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    results_dir = args.results_dir if args.results_dir.is_absolute() else root / args.results_dir
    output = args.output if args.output.is_absolute() else root / args.output
    runs: Dict[str, Any] = {}
    missing = []
    ignored_legacy_files = []
    for candidate in sorted(results_dir.glob("*.json")):
        if candidate.name in LEGACY_FILES:
            ignored_legacy_files.append(candidate.name)
            continue
    for name, filename in ACTIVE_FILES.items():
        metrics = load_if_available(results_dir / filename)
        if metrics is None:
            missing.append(filename)
        else:
            runs[name] = metrics
    sample_counts = {
        name: value.get("samples", value.get("test_samples"))
        for name, value in runs.items()
        if value.get("samples", value.get("test_samples")) is not None
    }
    split_root = root / "data" / "splits"
    split_counts = {
        name: len(json.loads((split_root / f"{name}.json").read_text(encoding="utf-8")))
        for name in ("train", "val", "test")
    }
    report = {
        "split": "test",
        "dataset": "FMA-small expanded real sample",
        "total_processed_tracks": sum(split_counts.values()),
        "train_samples": split_counts["train"],
        "validation_samples": split_counts["val"],
        "test_samples": split_counts["test"],
        "sample_counts_by_run": sample_counts,
        "runs": runs,
        "limitations": [
            "Results are one-epoch execution checks on the expanded held-out split.",
            "FMA metadata text is not MusicCaps natural-language caption text.",
            "DEAM emotion metrics are reported separately because DEAM and FMA audio IDs are unrelated.",
        ],
    }
    if missing:
        report["missing_files"] = missing
    if ignored_legacy_files:
        report["ignored_legacy_files"] = sorted(ignored_legacy_files)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
