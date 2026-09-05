from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class FMAGraphDataset(Dataset):
    """Load processed FMA graphs and genre targets from a split manifest."""

    def __init__(self, manifest_path: str | Path, label_names: Sequence[str]) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = next(
            parent for parent in self.manifest_path.parents if (parent / "src").is_dir()
        )
        self.records: List[Dict[str, Any]] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.label_names = list(label_names)
        self.label_to_index = {label: index for index, label in enumerate(self.label_names)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        graph_path = Path(record["graph_path"])
        if not graph_path.is_absolute():
            graph_path = self.project_root / graph_path
        graph = torch.load(graph_path, weights_only=False)
        if not isinstance(graph, Data):
            raise TypeError(f"Expected PyG Data in {graph_path}, got {type(graph).__name__}")

        target = torch.zeros(len(self.label_names), dtype=torch.float32)
        for tag in record.get("tags", []):
            label_index = self.label_to_index.get(str(tag).lower())
            if label_index is not None:
                target[label_index] = 1.0

        return {"graph": graph, "tags": target, "track_id": record["track_id"]}


def load_fma_label_names(split_root: str | Path) -> List[str]:
    """Collect the normalized labels present across all JSON manifests."""
    root = Path(split_root)
    labels = {
        str(tag).lower()
        for path in root.glob("*.json")
        for record in json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict)
        for tag in record.get("tags", [])
    }
    if not labels:
        raise ValueError(f"No labels found in manifests under {root}")
    return sorted(labels)


def load_manifest_label_names(split_root: str | Path) -> List[str]:
    """Collect labels when present, allowing caption-only manifests."""
    root = Path(split_root)
    labels = {
        str(tag).lower()
        for path in root.glob("*.json")
        if path.name != "alignment_report.json"
        for record in json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict)
        for tag in record.get("tags", [])
    }
    return sorted(labels)
