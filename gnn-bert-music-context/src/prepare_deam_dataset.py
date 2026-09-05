from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verified DEAM audio/target manifests.")
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/deam/audio"))
    parser.add_argument("--targets", type=Path, default=Path("data/processed/deam/deam_targets_from_zip.json"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/raw/deam/metadata"))
    parser.add_argument("--output-root", type=Path, default=Path("data/splits/deam"))
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_metadata(metadata_root: Path) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    for path in sorted(metadata_root.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                song_id = row.get("song_id") or row.get("id") or row.get("Id")
                if song_id and str(song_id).strip().isdigit():
                    metadata[str(int(song_id))] = row
    return metadata


def audio_files(audio_root: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(audio_root.rglob("*.mp3")):
        match = re.fullmatch(r"0*(\d+)", path.stem)
        if match:
            yield str(int(match.group(1))), path


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    audio_root = resolve(project_root, args.audio_root)
    targets_path = resolve(project_root, args.targets)
    metadata_root = resolve(project_root, args.metadata_root)
    output_root = resolve(project_root, args.output_root)

    target_payload: Dict[str, Any] = json.loads(targets_path.read_text(encoding="utf-8"))
    targets = {str(int(item["deam_song_id"])): item for item in target_payload["targets"]}
    metadata = load_metadata(metadata_root)
    records = []
    unmatched_audio = 0
    for song_id, audio_path in audio_files(audio_root):
        target = targets.get(song_id)
        if target is None:
            unmatched_audio += 1
            continue
        row = metadata.get(song_id, {})
        artist = str(row.get("Artist", row.get("artist", ""))).strip()
        title = str(row.get("Song title", row.get("Track", row.get("title", "")))).strip()
        genre = str(row.get("Genre", row.get("genre", ""))).strip()
        text_parts = [f"artist: {artist}", f"track: {title}", f"genre: {genre}"]
        record = {
            "deam_song_id": song_id,
            "audio_path": audio_path.relative_to(project_root).as_posix(),
            "text_context": ". ".join(part for part in text_parts if part.split(": ", 1)[-1]),
            "genre": genre,
            "tags": [genre.lower()] if genre else [],
            "valence": target["valence"],
            "arousal": target["arousal"],
            "valence_std": target["valence_std"],
            "arousal_std": target["arousal_std"],
            "audio_source": "DEAM official audio archive",
            "target_source": target["source"],
        }
        records.append(record)

    records.sort(key=lambda item: int(item["deam_song_id"]))
    train_end = int(len(records) * 0.8)
    val_end = int(len(records) * 0.9)
    splits = {"train.json": records[:train_end], "val.json": records[train_end:val_end], "test.json": records[val_end:]}
    output_root.mkdir(parents=True, exist_ok=True)
    for filename, split in splits.items():
        (output_root / filename).write_text(json.dumps(split, indent=2), encoding="utf-8")
    report = {
        "audio_root": str(audio_root),
        "target_records": len(targets),
        "matched_audio_records": len(records),
        "unmatched_audio_files": unmatched_audio,
        "metadata_records": len(metadata),
        "splits": {name.removesuffix(".json"): len(rows) for name, rows in splits.items()},
        "pairing_rule": "exact normalized numeric DEAM song_id to audio filename stem",
    }
    (output_root / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
