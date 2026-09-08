from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List
# Fixed, auditable caption-proxy vocabulary. Labels are lexical concepts found in
# MusicCaps descriptions; they are not claimed to be human ground-truth tags.
PROXY_LABELS = [
    "acoustic", "arabic", "bass", "brass", "calm", "choir", "clarinet",
    "classical", "concert", "country", "dance", "dark", "drums", "electric",
    "electronic", "emotional", "energetic", "female vocal", "folk", "funk",
    "guitar", "happy", "hip hop", "horror", "instrumental", "jazz", "live",
    "male vocal", "melody", "metal", "orchestral", "piano", "pop", "rap",
    "reggae", "rock", "romantic", "sad", "saxophone", "singer", "soul",
    "strings", "synth", "tempo", "violin", "vocal", "voice", "woodwind",
    "world", "worship",
]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).replace("  ", " ").strip()


def matches(label: str, text: str) -> bool:
    normalized = normalize(text)
    if label == "female vocal":
        return bool(re.search(r"\b(female|woman|women|girl)\b", normalized) and re.search(r"\b(vocal|voice|sing|singer)\b", normalized))
    if label == "male vocal":
        return bool(re.search(r"\b(male|man|men|boy)\b", normalized) and re.search(r"\b(vocal|voice|sing|singer)\b", normalized))
    if label == "hip hop":
        return "hip hop" in normalized or "hip-hop" in text.lower()
    return bool(re.search(rf"\b{re.escape(label)}s?\b", normalized))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MusicCaps caption-to-tag proxy manifest for Task 1.")
    parser.add_argument("--input-root", type=Path, default=Path("data/splits/musiccaps"))
    parser.add_argument("--output-root", type=Path, default=Path("data/splits/musiccaps_task1_proxy"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    input_root = args.input_root if args.input_root.is_absolute() else root / args.input_root
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {label: 0 for label in PROXY_LABELS}
    split_counts: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        source = json.loads((input_root / f"{split}.json").read_text(encoding="utf-8"))
        records: List[Dict[str, Any]] = []
        for source_record in source:
            caption = str(source_record.get("caption", source_record.get("text_context", "")))
            tags = [label for label in PROXY_LABELS if matches(label, caption)]
            if not tags:
                continue
            record = dict(source_record)
            record["text_context"] = caption
            record["tags"] = tags
            record["proxy_label_source"] = "deterministic lexical matches against fixed 50-label MusicCaps caption vocabulary"
            records.append(record)
            for tag in tags:
                counts[tag] += 1
        (output_root / f"{split}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        split_counts[split] = len(records)
    report = {
        "dataset": "MusicCaps verified audio-caption subset",
        "task": "task1_caption_to_tag_proxy",
        "labels": PROXY_LABELS,
        "label_count": len(PROXY_LABELS),
        "splits": split_counts,
        "label_counts": {key: value for key, value in counts.items() if value},
        "excluded_caption_records_without_proxy_label": "records with no lexical ontology match are excluded from supervised proxy training",
        "warning": "These are deterministic caption-derived proxy labels, not independently annotated ground truth.",
    }
    (output_root / "proxy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
