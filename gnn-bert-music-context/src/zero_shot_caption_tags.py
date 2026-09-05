from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import Counter

from fma_dataset import load_fma_label_names

STOPWORDS = {
    "about", "after", "also", "and", "are", "as", "at", "be", "because", "been", "being", "by", "can",
    "could", "does", "for", "from", "has", "have", "into", "its", "just", "like", "more", "not", "of",
    "on", "one", "only", "or", "over", "same", "that", "the", "their", "there", "these", "this", "to",
    "two", "used", "using", "was", "were", "what", "when", "which", "with",
}


def tokens(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z][a-z-]+", text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate caption-to-tag zero-shot overlap on MusicCaps captions.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits/musiccaps"))
    parser.add_argument("--label-manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--supervised-metrics", type=Path, default=Path("results/expanded_fma_task3_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("results/musiccaps_zero_shot_tags.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    label_root = args.label_manifest_root if args.label_manifest_root.is_absolute() else root / args.label_manifest_root
    supervised_metrics_path = args.supervised_metrics if args.supervised_metrics.is_absolute() else root / args.supervised_metrics
    rows = json.loads((manifest_root / "test.json").read_text(encoding="utf-8"))
    labels = load_fma_label_names(label_root)
    vocab = Counter(word for row in rows for word in tokens(row["caption"]))
    common = {word for word, count in vocab.items() if count >= 2}
    predictions = []
    for row in rows:
        caption_tokens = tokens(row["caption"])
        predicted = [label for label in labels if set(tokens(label)).issubset(caption_tokens)]
        predictions.append({"track_id": row["track_id"], "caption": row["caption"], "zero_shot_fma_tags": predicted, "caption_content_words": sorted(caption_tokens & common)})
    supervised = json.loads(supervised_metrics_path.read_text(encoding="utf-8")) if supervised_metrics_path.exists() else {}
    payload = {
        "dataset": "MusicCaps",
        "method": "zero-shot exact lexical match against the supervised FMA tag vocabulary",
        "samples": len(rows),
        "label_vocabulary": labels,
        "supervised_task3_reference": {key: supervised.get(key) for key in ("macro_f1", "micro_f1", "auc_pr")},
        "note": "MusicCaps has no ground-truth FMA tags, so zero-shot tag accuracy is not claimed.",
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()