from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, List
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data/raw/fma/metadata/extracted/fma_metadata/tracks.csv"
SPLIT_ROOT = PROJECT_ROOT / "data/splits"
LABEL_SOURCE = "data/raw/fma/metadata/extracted/fma_metadata/tracks.csv"
RAW_METADATA_PATH = METADATA_PATH.parent / "raw_tracks.csv"


def parse_genre_ids(value: Any) -> List[int]:
    # Parse FMA's serialized genre-id list while tolerating empty values.
    if value is None or pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed if str(item).isdigit()]


def normalize_labels(labels: List[str]) -> List[str]:
    # Normalize labels and preserve deterministic ordering without duplicates.
    return sorted({label.strip().lower() for label in labels if label.strip()})


def parse_text_tags(value: Any) -> List[str]:
    # Parse raw FMA tag lists when they are stored as Python literals.
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def build_text_context(row: Any, tags: List[str]) -> str:
    # Create deterministic text context for BERT from FMA metadata.
    parts = [
        f"artist: {row.get('artist_name', '')}",
        f"album: {row.get('album_title', '')}",
        f"track: {row.get('track_title', '')}",
        f"genres: {', '.join(tags)}",
    ]
    return ". ".join(part for part in parts if part.split(": ", 1)[-1].strip())


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH, header=[0, 1], index_col=0)
    raw_metadata = pd.read_csv(RAW_METADATA_PATH).set_index("track_id")
    genres_path = METADATA_PATH.parent / "genres.csv"
    genres = pd.read_csv(genres_path)
    genre_titles = {
        int(row.genre_id): str(row.title)
        for row in genres.itertuples()
        if not pd.isna(row.genre_id) and not pd.isna(row.title)
    }
    genre_top_column = ("track", "genre_top")
    genres_all_column = ("track", "genres_all")
    updated = 0
    labeled = 0
    label_counts = {}

    for split_path in sorted(SPLIT_ROOT.glob("*.json")):
        if split_path.name == "README.json":
            continue
        records = json.loads(split_path.read_text(encoding="utf-8"))
        for record in records:
            track_id = int(record["track_id"])
            if track_id not in metadata.index:
                continue
            genre_top = metadata.loc[track_id, genre_top_column]
            genre_ids = parse_genre_ids(metadata.loc[track_id, genres_all_column])
            genre_labels = [genre_titles[genre_id] for genre_id in genre_ids if genre_id in genre_titles]
            if not pd.isna(genre_top):
                genre_labels.append(str(genre_top))
            tags = normalize_labels(genre_labels)
            if not tags:
                continue
            record["genre_top"] = None if pd.isna(genre_top) else str(genre_top)
            record["genre_ids"] = genre_ids
            record["genre_tags"] = tags
            record["tags"] = tags
            raw_row = raw_metadata.loc[track_id] if track_id in raw_metadata.index else None
            raw_tags = parse_text_tags(raw_row.get("tags")) if raw_row is not None else []
            record["text_tags"] = normalize_labels(raw_tags)
            record["text_context"] = build_text_context(raw_row, tags) if raw_row is not None else f"genres: {', '.join(tags)}"
            record["metadata_available"] = True
            record["label_source"] = LABEL_SOURCE
            updated += 1
            labeled += int(bool(tags))
            for tag in tags:
                label_counts[tag] = label_counts.get(tag, 0) + 1
        split_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"Updated records: {updated}")
    print(f"Records with FMA genre labels: {labeled}")
    print(f"Unique genre labels: {len(label_counts)}")
    print(f"Label counts: {dict(sorted(label_counts.items()))}")
    print(f"Text contexts added: {updated}")


if __name__ == "__main__":
    main()
