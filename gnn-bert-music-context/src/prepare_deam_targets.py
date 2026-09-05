from __future__ import annotations
import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize DEAM song-level valence/arousal annotations and report FMA overlap.")
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/deam/DEAM_Annotations.zip"), help="DEAM annotation ZIP or extracted directory.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/deam/deam_targets.json"))
    return parser.parse_args()

def find_static_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("static_annotations_averaged_songs_*.csv"))
    if not files:
        raise FileNotFoundError("DEAM static annotation CSVs were not found")
    return files

def read_static_rows(source: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = sorted(name for name in archive.namelist() if Path(name).name.startswith("static_annotations_averaged_songs_") and name.endswith(".csv"))
            if not names:
                raise FileNotFoundError(f"No DEAM static annotation CSVs found in {source}")
            for name in names:
                with archive.open(name) as raw:
                    text = (line.decode("utf-8-sig") for line in raw)
                    rows.extend(csv.DictReader(text, skipinitialspace=True))
        return rows
    for static_path in find_static_files(source):
        with static_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle, skipinitialspace=True))
    return rows

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    annotation_root = args.annotations if args.annotations.is_absolute() else project_root / args.annotations
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else project_root / args.manifest_root
    output = args.output if args.output.is_absolute() else project_root / args.output
    rows = read_static_rows(annotation_root)
    targets: List[Dict[str, Any]] = []
    seen_ids = set()
    for row in rows:
        try:
            song_id = str(row["song_id"]).strip()
            if song_id in seen_ids:
                continue
            targets.append({
                "deam_song_id": song_id,
                "valence": float(row["valence_mean"]),
                "valence_std": float(row["valence_std"]),
                "arousal": float(row["arousal_mean"]),
                "arousal_std": float(row["arousal_std"]),
                "source": "DEAM song-level averaged annotations",
            })
            seen_ids.add(song_id)
        except (KeyError, TypeError, ValueError):
            continue
    fma_ids = set()
    for split in ("train", "val", "test"):
        records = json.loads((manifest_root / f"{split}.json").read_text(encoding="utf-8"))
        fma_ids.update(str(record.get("track_id", "")).strip().lstrip("0") or "0" for record in records)
    overlap = [item for item in targets if item["deam_song_id"].lstrip("0") in fma_ids]
    result = {
        "source": str(annotation_root),
        "total_deam_targets": len(targets),
        "fma_track_id_overlap": len(overlap),
        "targets": targets,
        "limitations": ["DEAM IDs are not assumed to identify FMA tracks; overlap is reported only by exact normalized numeric ID."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "targets"}, indent=2))

if __name__ == "__main__":
    main()
