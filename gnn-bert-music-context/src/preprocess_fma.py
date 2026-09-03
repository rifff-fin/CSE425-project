from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from audio_features import AudioProcessor
from graph_builder import MusicGraphBuilder


def load_metadata(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {str(row["track_id"]).zfill(6): dict(row) for row in rows if row.get("track_id")}


def track_id_from_path(path: Path) -> str:
    return path.stem.zfill(6)


def iter_audio_files(audio_root: Path, limit: int) -> Iterable[Path]:
    files = sorted(audio_root.rglob("*.mp3"))
    if not files:
        raise FileNotFoundError(f"No MP3 files found under {audio_root}")
    return files[:limit] if limit > 0 else files


def parse_tags(row: Dict[str, Any]) -> List[str]:
    raw_tags = row.get("tags") or row.get("genre") or ""
    return [tag.strip() for tag in raw_tags.replace("|", ";").split(";") if tag.strip()]


def process_track(
    audio_path: Path,
    processor: AudioProcessor,
    graph_builder: MusicGraphBuilder,
    output_root: Path,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    track_id = track_id_from_path(audio_path)
    segments = processor.segment_file(audio_path)
    vectors = [processor.summarize_segment(segment) for segment in segments]
    graph = graph_builder.build_graph(vectors)

    graph_path = output_root / "graphs" / f"{track_id}.pt"
    feature_path = output_root / "audio_features" / f"{track_id}.pt"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, graph_path)
    torch.save(
        {
            "track_id": track_id,
            "segments": [
                {
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "node_feature": vector,
                }
                for segment, vector in zip(segments, vectors)
            ],
        },
        feature_path,
    )

    row = metadata.get(track_id, {})
    return {
        "track_id": track_id,
        "audio_path": str(audio_path).replace("\\", "/"),
        "graph_path": str(graph_path).replace("\\", "/"),
        "feature_path": str(feature_path).replace("\\", "/"),
        "tags": parse_tags(row),
        "metadata_available": bool(row),
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.edge_index.size(1)),
    }


def write_splits(records: List[Dict[str, Any]], split_root: Path) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    total = len(records)
    train_end = max(1, int(total * 0.8))
    val_end = max(train_end, int(total * 0.9))
    splits = {
        "train.json": records[:train_end],
        "val.json": records[train_end:val_end],
        "test.json": records[val_end:],
    }
    for filename, rows in splits.items():
        (split_root / filename).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph samples and manifests from FMA audio.")
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/fma/extracted/fma_small"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--limit", type=int, default=20, help="Number of MP3 files; use 0 for all files.")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=2.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    audio_root = args.audio_root if args.audio_root.is_absolute() else project_root / args.audio_root
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    split_root = args.split_root if args.split_root.is_absolute() else project_root / args.split_root
    metadata_path = None if args.metadata is None else (args.metadata if args.metadata.is_absolute() else project_root / args.metadata)

    processor = AudioProcessor(
        sample_rate=args.sample_rate,
        segment_window_sec=args.window_sec,
        segment_hop_sec=args.hop_sec,
    )
    graph_builder = MusicGraphBuilder(similarity_threshold=args.similarity_threshold)
    metadata = load_metadata(metadata_path)
    records: List[Dict[str, Any]] = []

    for index, audio_path in enumerate(iter_audio_files(audio_root, args.limit), start=1):
        try:
            record = process_track(audio_path, processor, graph_builder, output_root, metadata)
            records.append(record)
            print(f"[{index}] processed {record['track_id']} ({record['num_nodes']} nodes)")
        except Exception as error:
            print(f"[{index}] skipped {audio_path.name}: {error}", file=sys.stderr)

    if not records:
        raise RuntimeError("No tracks were processed successfully.")
    write_splits(records, split_root)
    print(f"Processed tracks: {len(records)}")
    print(f"Graph output: {output_root / 'graphs'}")
    print(f"Split manifests: {split_root}")


if __name__ == "__main__":
    main()
