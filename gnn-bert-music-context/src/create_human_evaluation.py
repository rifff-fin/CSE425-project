from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a five-listener MusicCaps human evaluation sheet.")
    parser.add_argument("--examples", type=Path, default=Path("results/retrieval_examples/musiccaps_examples.json"))
    parser.add_argument("--output", type=Path, default=Path("results/retrieval_examples/human_evaluation.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    examples_path = args.examples if args.examples.is_absolute() else root / args.examples
    output = args.output if args.output.is_absolute() else root / args.output
    examples = json.loads(examples_path.read_text(encoding="utf-8"))["examples"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["listener_id", "query_audio_id", "query_caption", "retrieved_rank", "retrieved_audio_id", "rating_1_to_5", "notes"])
        for listener in range(1, 6):
            for example in examples:
                for item in example["top_3_retrieved_audio"]:
                    writer.writerow([listener, example["query_audio_id"], example["query_caption"], item["rank"], item["track_id"], "", ""])
    print(f"Created human evaluation sheet: {output}")


if __name__ == "__main__":
    main()