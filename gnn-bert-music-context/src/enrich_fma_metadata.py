from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data/raw/fma/metadata/extracted/fma_metadata/tracks.csv"
SPLIT_ROOT = PROJECT_ROOT / "data/splits"
LABEL_SOURCE = "data/raw/fma/metadata/extracted/fma_metadata/tracks.csv"


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH, header=[0, 1], index_col=0)
    genre_column = ("track", "genre_top")
    updated = 0
    labeled = 0

    for split_path in sorted(SPLIT_ROOT.glob("*.json")):
        if split_path.name == "README.json":
            continue
        records = json.loads(split_path.read_text(encoding="utf-8"))
        for record in records:
            track_id = int(record["track_id"])
            if track_id not in metadata.index:
                continue
            genre = metadata.loc[track_id, genre_column]
            if pd.isna(genre):
                continue
            record["genre_top"] = str(genre)
            record["tags"] = [str(genre).lower()]
            record["label_source"] = LABEL_SOURCE
            updated += 1
            labeled += 1
        split_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"Updated records: {updated}")
    print(f"Records with FMA genre labels: {labeled}")


if __name__ == "__main__":
    main()
