from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Sequence
import torch
from torch.utils.data import Dataset
class MusicCapsProxyTextDataset(Dataset):
    # MusicCaps captions with deterministic caption-derived proxy labels.
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
        return {"texts": [record.get("caption", record.get("text_context", ""))], "tags": target, "track_id": record["track_id"]}

def load_proxy_label_names(split_root: str | Path) -> list[str]:
    root = Path(split_root)
    report_path = root / "proxy_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        labels = [str(label).lower() for label in report.get("labels", [])]
        if labels:
            return labels
    labels = set()
    for path in root.glob("*.json"):
        if path.name == "proxy_report.json":
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            labels.update(str(tag).lower() for tag in record.get("tags", []))
    if not labels:
        raise ValueError(f"No proxy labels found under {root}")
    return sorted(labels)
