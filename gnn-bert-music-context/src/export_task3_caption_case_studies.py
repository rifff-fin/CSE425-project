from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import torch

def main() -> None:
    parser = argparse.ArgumentParser(description="Export verified MusicCaps caption/graph case studies for Task 3 alignment analysis.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits/musiccaps"))
    parser.add_argument("--output", type=Path, default=Path("results/task3_caption_case_studies.json"))
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    output = args.output if args.output.is_absolute() else root / args.output
    records: List[Dict[str, Any]] = json.loads((manifest_root / "test.json").read_text(encoding="utf-8"))
    studies = []
    for record in records[: args.count]:
        graph_path = Path(record["graph_path"])
        if not graph_path.is_absolute():
            graph_path = root / graph_path
        graph = torch.load(graph_path, weights_only=False)
        studies.append({
            "track_id": record["track_id"],
            "caption": record.get("caption", record.get("text_context", "")),
            "text_source": record.get("text_source", "musiccaps"),
            "audio_path": record.get("audio_path"),
            "graph_path": str(graph_path.resolve().relative_to(root.resolve())).replace("\\", "/"),
            "caption_graph_alignment": {
                "shared_track_id": record["track_id"],
                "caption_start_s": record.get("caption_start_s"),
                "caption_end_s": record.get("caption_end_s"),
                "num_nodes": int(graph.num_nodes),
                "num_edges": int(graph.edge_index.size(1)),
                "edge_preview": graph.edge_index[:, : min(10, graph.edge_index.size(1))].T.tolist(),
            },
        })
    result = {
        "task": "task3",
        "analysis": "caption_graph_alignment_case_studies",
        "dataset": "MusicCaps verified audio-caption subset",
        "samples": len(studies),
        "case_studies": studies,
        "note": "These are verified caption/graph alignment cases for Task 3 analysis; FMA supervised tags and DEAM emotion regression remain separate task paths.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
