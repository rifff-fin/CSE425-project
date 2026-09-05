from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a controlled model comparison table from metric JSON files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": payload.get("model", payload.get("task")),
            "epochs": payload.get("epochs", "see training command"),
            "macro_f1": payload.get("macro_f1"),
            "micro_f1": payload.get("micro_f1"),
            "auc_pr": payload.get("auc_pr"),
            "samples": payload.get("samples", payload.get("test_samples")),
            "source": str(path).replace("\\", "/"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"comparison": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": rows}, indent=2))


if __name__ == "__main__":
    main()