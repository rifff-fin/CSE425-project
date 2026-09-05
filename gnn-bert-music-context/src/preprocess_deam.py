from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from audio_features import AudioProcessor
from graph_builder import MusicGraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PyG graphs for verified DEAM manifests.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits/deam"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/deam"))
    parser.add_argument("--split-root", type=Path, default=Path("data/splits/deam_processed"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=2.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest_root = resolve(project_root, args.manifest_root)
    output_root = resolve(project_root, args.output_root)
    split_root = resolve(project_root, args.split_root)
    processor = AudioProcessor(
        sample_rate=args.sample_rate,
        segment_window_sec=args.window_sec,
        segment_hop_sec=args.hop_sec,
    )
    graph_builder = MusicGraphBuilder(similarity_threshold=args.similarity_threshold)
    if args.limit > 0:
        split_limits = {
            "train": max(1, int(args.limit * 0.8)),
            "val": max(1, int(args.limit * 0.1)),
            "test": max(1, args.limit - int(args.limit * 0.8) - int(args.limit * 0.1)),
        }
    else:
        split_limits = {"train": 0, "val": 0, "test": 0}
    output_splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for split_name in ("train", "val", "test"):
        source_path = manifest_root / f"{split_name}.json"
        records = json.loads(source_path.read_text(encoding="utf-8"))
        selected_records = records[:split_limits[split_name]] if args.limit > 0 else records
        for record in selected_records:
            if args.limit > 0 and len(output_splits[split_name]) >= split_limits[split_name]:
                break
            audio_path = project_root / record["audio_path"]
            segments = processor.segment_file(audio_path)
            vectors = [processor.summarize_segment(segment) for segment in segments]
            graph = graph_builder.build_graph(vectors)
            graph_path = output_root / "graphs" / f"{record['deam_song_id']}.pt"
            feature_path = output_root / "audio_features" / f"{record['deam_song_id']}.pt"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph, graph_path)
            torch.save(
                {
                    "track_id": record["deam_song_id"],
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
            enriched = dict(record)
            enriched["track_id"] = record["deam_song_id"]
            enriched["graph_path"] = graph_path.relative_to(project_root).as_posix()
            enriched["feature_path"] = feature_path.relative_to(project_root).as_posix()
            enriched["num_nodes"] = int(graph.num_nodes)
            enriched["num_edges"] = int(graph.edge_index.size(1))
            output_splits[split_name].append(enriched)
            print(f"[{sum(len(rows) for rows in output_splits.values())}] processed DEAM {record['deam_song_id']}")

    split_root.mkdir(parents=True, exist_ok=True)
    for split_name, rows in output_splits.items():
        (split_root / f"{split_name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = {
        "source_manifest_root": str(manifest_root),
        "processed_records": sum(len(rows) for rows in output_splits.values()),
        "splits": {name: len(rows) for name, rows in output_splits.items()},
        "graph_root": str(output_root / "graphs"),
    }
    (split_root / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
