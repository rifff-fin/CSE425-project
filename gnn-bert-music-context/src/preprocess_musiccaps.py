from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from audio_features import AudioProcessor
from graph_builder import MusicGraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph manifests for verified MusicCaps audio-caption pairs.")
    parser.add_argument("--captions", type=Path, default=Path("data/raw/musiccaps/musiccaps-public.csv"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/musiccaps/audio"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/musiccaps"))
    parser.add_argument("--split-root", type=Path, default=Path("data/splits/musiccaps"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    captions = args.captions if args.captions.is_absolute() else root / args.captions
    audio_root = args.audio_root if args.audio_root.is_absolute() else root / args.audio_root
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    split_root = args.split_root if args.split_root.is_absolute() else root / args.split_root
    with captions.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if args.limit > 0:
        records = records[: args.limit]

    processor = AudioProcessor(
        sample_rate=args.sample_rate,
        segment_window_sec=args.window_sec,
        segment_hop_sec=args.hop_sec,
    )
    builder = MusicGraphBuilder()
    pairs: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        ytid = record.get("ytid", "").strip()
        audio_path = audio_root / f"{ytid}.wav"
        if not ytid or not audio_path.exists():
            continue
        try:
            segments = processor.segment_file(audio_path)
            vectors = [processor.summarize_segment(segment) for segment in segments]
            graph = builder.build_graph(vectors)
            graph_path = output_root / "graphs" / f"{ytid}.pt"
            feature_path = output_root / "audio_features" / f"{ytid}.pt"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph, graph_path)
            torch.save({"track_id": ytid, "segments": [
                {"start_sec": segment.start_sec, "end_sec": segment.end_sec, "node_feature": vector}
                for segment, vector in zip(segments, vectors)
            ]}, feature_path)
            pairs.append({
                "track_id": ytid,
                "audio_path": str(audio_path).replace("\\", "/"),
                "graph_path": str(graph_path).replace("\\", "/"),
                "feature_path": str(feature_path).replace("\\", "/"),
                "caption": record["caption"],
                "text_context": record["caption"],
                "text_source": "musiccaps",
                "caption_start_s": float(record["start_s"]),
                "caption_end_s": float(record["end_s"]),
                "num_nodes": int(graph.num_nodes),
                "num_edges": int(graph.edge_index.size(1)),
            })
            print(f"[{index}] processed {ytid}")
        except Exception as error:
            print(f"[{index}] skipped {ytid}: {error}", file=sys.stderr)

    if len(pairs) < 3:
        raise RuntimeError("At least three verified MusicCaps audio-caption pairs are required.")
    train_end = max(1, int(len(pairs) * 0.8))
    val_end = min(len(pairs) - 1, max(train_end + 1, int(len(pairs) * 0.9)))
    split_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", pairs[:train_end]), ("val", pairs[train_end:val_end]), ("test", pairs[val_end:])):
        (split_root / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (split_root / "alignment_report.json").write_text(json.dumps({
        "dataset": "MusicCaps",
        "caption_records_considered": len(records),
        "verified_audio_caption_pairs": len(pairs),
        "audio_ids": [pair["track_id"] for pair in pairs],
    }, indent=2), encoding="utf-8")
    print(f"Verified MusicCaps pairs: {len(pairs)}")


if __name__ == "__main__":
    main()