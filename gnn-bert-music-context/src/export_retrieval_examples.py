from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from fma_dataset import load_manifest_label_names
from fma_paired_dataset import FMAPairedDataset
from train import FusionTrainingModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Export qualitative MusicCaps caption/audio retrieval examples.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits/musiccaps"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/retrieval_examples/musiccaps_examples.json"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output = args.output if args.output.is_absolute() else root / args.output
    dataset = FMAPairedDataset(manifest_root / "test.json", load_manifest_label_names(manifest_root))
    rows = [dataset[index] for index in range(len(dataset))]
    model = FusionTrainingModel("task4", num_tags=0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        graph, hidden = model.encode(Batch.from_data_list([row["graph"] for row in rows]), [row["texts"][0] for row in rows])
        audio = F.normalize(model.head.g_proj(graph), dim=-1)
        text = F.normalize(model.head.text_proj(hidden.mean(dim=1)), dim=-1)
        similarity = text @ audio.T
    examples = []
    for index, record in enumerate(dataset.records[:args.limit]):
        ranking = torch.argsort(similarity[index], descending=True)[:3].tolist()
        examples.append({
            "query_caption": record["caption"],
            "query_audio_id": record["track_id"],
            "top_3_retrieved_audio": [
                {"rank": rank + 1, "track_id": dataset.records[item]["track_id"], "audio_path": dataset.records[item]["audio_path"], "similarity": float(similarity[index, item])}
                for rank, item in enumerate(ranking)
            ],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": "MusicCaps", "examples": examples}, indent=2), encoding="utf-8")
    print(f"Exported {len(examples)} retrieval examples to {output}")


if __name__ == "__main__":
    main()