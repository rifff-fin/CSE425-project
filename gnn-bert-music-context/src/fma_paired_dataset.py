from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Sequence
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
class FMAPairedDataset(Dataset):
    # Load aligned FMA graphs, metadata text, tags, and optional emotion targets.

    def __init__(self, manifest_path: str | Path, label_names: Sequence[str]) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = next(
            parent for parent in self.manifest_path.parents if (parent / "src").is_dir()
        )
        self.records = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.label_names = list(label_names)
        self.label_to_index = {name.lower(): i for i, name in enumerate(self.label_names)}

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

        tags = torch.zeros(len(self.label_names), dtype=torch.float32)
        for tag in record.get("tags", []):
            label_index = self.label_to_index.get(str(tag).lower())
            if label_index is not None:
                tags[label_index] = 1.0
        # FMA has no emotion annotations. NaN-free neutral placeholders keep the
        # multitask interface usable; emotion loss is disabled by the trainer below.
        valence = float(record.get("valence", 0.0))
        arousal = float(record.get("arousal", 0.0))
        has_emotion = "valence" in record and "arousal" in record
        return {
            "graph": graph,
            "texts": [record.get("text_context", "")],
            "tag_labels": tags,
            "valence": torch.tensor([valence], dtype=torch.float32),
            "arousal": torch.tensor([arousal], dtype=torch.float32),
            "has_emotion": torch.tensor(has_emotion, dtype=torch.bool),
            "track_id": record["track_id"],
        }
