from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed MusicCaps listener ratings.")
    parser.add_argument("--input", type=Path, default=Path("results/retrieval_examples/human_evaluation.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/retrieval_examples/human_evaluation_summary.json"))
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    valid_scores: list[dict[str, Any]] = []
    missing = 0
    invalid = 0
    for row in rows:
        value = (row.get("rating_1_to_5") or "").strip()
        if not value:
            missing += 1
            continue
        try:
            score = int(value)
        except ValueError:
            invalid += 1
            continue
        if score < 1 or score > 5:
            invalid += 1
            continue
        valid_scores.append({**row, "score": score})

    by_rank: dict[str, list[int]] = {}
    for row in valid_scores:
        by_rank.setdefault(row["retrieved_rank"], []).append(row["score"])

    summary = {
        "expected_ratings": len(rows),
        "completed_ratings": len(valid_scores),
        "missing_ratings": missing,
        "invalid_ratings": invalid,
        "listeners": sorted({row["listener_id"] for row in rows}),
        "queries": len({row["query_audio_id"] for row in rows}),
        "mean_score": mean(row["score"] for row in valid_scores) if valid_scores else None,
        "score_distribution": dict(sorted(Counter(row["score"] for row in valid_scores).items())),
        "mean_score_by_rank": {
            rank: mean(scores) for rank, scores in sorted(by_rank.items())
        },
        "status": "complete" if valid_scores and not missing and not invalid else "incomplete",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
