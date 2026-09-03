from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Sequence
import torch
from torch.utils.data import Dataset
class FMATextDataset(Dataset):
    # Load FMA metadata text contexts and multi-label targets.

    def __init__(self, manifest_path: str | Path, label_names: Sequence[str]) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.label_names = list(label_names)
        self.label_to_index = {label.lower(): index for index, label in enumerate(self.label_names)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        target = torch.zeros(len(self.label_names), dtype=torch.float32)
        for tag in record.get("tags", []):
            label_index = self.label_to_index.get(str(tag).lower())
            if label_index is not None:
                target[label_index] = 1.0
        return {
            "texts": [record.get("text_context", "")],
            "tags": target,
            "track_id": record["track_id"],
        }
