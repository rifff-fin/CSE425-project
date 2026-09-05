from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align local MusicCaps-style captions with FMA manifests.")
    parser.add_argument("--captions", type=Path, required=True, help="JSON or CSV caption records.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("data/splits/musiccaps"))
    return parser.parse_args()

def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def caption_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                yield {"id": key, "caption": value}
            elif isinstance(value, dict):
                record = dict(value)
                record.setdefault("id", key)
                yield record
    elif isinstance(payload, list):
        for value in payload:
            if isinstance(value, dict):
                yield value

def record_keys(record: Dict[str, Any]) -> set[str]:
    keys = set()
    for field in ("id", "track_id", "ytid", "youtube_id", "audio_id", "filename", "audio_path"):
        value = record.get(field)
        if value is not None:
            keys.add(norm(value))
            keys.add(norm(Path(str(value)).stem))
    return {key for key in keys if key}

def load_caption_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    return list(caption_records(json.loads(path.read_text(encoding="utf-8"))))

def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    captions_path = args.captions if args.captions.is_absolute() else root / args.captions
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    captions = load_caption_records(captions_path)
    caption_index: Dict[str, Dict[str, Any]] = {}
    for record in captions:
        text = record.get("caption", record.get("text", record.get("description")))
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        if not text:
            continue
        for key in record_keys(record):
            caption_index.setdefault(key, {"caption": str(text), "source_id": record.get("id")})

    summary: Dict[str, Any] = {"caption_source": str(captions_path), "caption_records": len(captions), "matched": 0, "unmatched": 0}
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        manifest_path = manifest_root / f"{split}.json"
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        aligned: List[Dict[str, Any]] = []
        for record in records:
            keys = record_keys({"track_id": record.get("track_id"), "audio_path": record.get("audio_path")})
            match = next((caption_index[key] for key in keys if key in caption_index), None)
            if match is None:
                summary["unmatched"] += 1
                continue
            merged = dict(record)
            merged["caption"] = match["caption"]
            merged["caption_source_id"] = match["source_id"]
            merged["caption_start_s"] = match.get("start_s")
            merged["caption_end_s"] = match.get("end_s")
            merged["caption_aspects"] = match.get("aspect_list")
            merged["text_context"] = match["caption"]
            merged["text_source"] = "musiccaps"
            aligned.append(merged)
            summary["matched"] += 1
        (output_root / f"{split}.json").write_text(json.dumps(aligned, indent=2), encoding="utf-8")
        summary[f"{split}_records"] = len(aligned)
    (output_root / "alignment_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
