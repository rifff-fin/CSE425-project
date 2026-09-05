from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.manifold import TSNE
from torch_geometric.data import Batch

from bert_encoder import BERTTextEncoder
from evaluate import evaluate_tagging
from fma_dataset import load_fma_label_names
from fma_paired_dataset import FMAPairedDataset
from gnn_model import MusicGNNEncoder


def load_split(root: Path, name: str, labels: List[str]) -> FMAPairedDataset:
    return FMAPairedDataset(root / f"{name}.json", labels)


def encode_dataset(dataset: FMAPairedDataset, gnn: MusicGNNEncoder, bert: BERTTextEncoder, batch_size: int) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    graph_vectors: List[torch.Tensor] = []
    text_vectors: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    records: List[Dict[str, object]] = []
    bert.eval(); gnn.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            graph = Batch.from_data_list([row["graph"] for row in rows])
            graph_vectors.append(gnn(graph))
            _, pooled = bert([row["texts"][0] for row in rows])
            text_vectors.append(pooled)
            targets.append(torch.stack([row["tag_labels"] for row in rows]))
            records.extend(dataset.records[start:start + len(rows)])
    return torch.cat(graph_vectors).numpy(), torch.cat(text_vectors).numpy(), records


def train_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=500, class_weight="balanced"))
    classifier.fit(train_x, train_y)
    return torch.tensor(classifier.predict_proba(test_x), dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 3 ablations, t-SNE, and case-study export.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/task3_analysis.json"))
    parser.add_argument("--plot", type=Path, default=Path("results/plots/task3_tsne.png"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else root / args.manifest_root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output = args.output if args.output.is_absolute() else root / args.output
    labels = load_fma_label_names(manifest_root)
    train_ds = load_split(manifest_root, "train", labels)
    test_ds = load_split(manifest_root, "test", labels)
    gnn = MusicGNNEncoder(in_channels=32, hidden_dim=128, num_layers=2, model_type="sage")
    bert = BERTTextEncoder(model_name="distilbert-base-uncased")
    model_state = torch.load(checkpoint, map_location="cpu")
    gnn.load_state_dict({key.removeprefix("gnn."): value for key, value in model_state.items() if key.startswith("gnn.")})
    bert.load_state_dict({key.removeprefix("text_encoder."): value for key, value in model_state.items() if key.startswith("text_encoder.")}, strict=False)
    train_g, train_t, _ = encode_dataset(train_ds, gnn, bert, args.batch_size)
    test_g, test_t, test_records = encode_dataset(test_ds, gnn, bert, args.batch_size)
    train_y = np.array([row.get("tags", []) for row in train_ds.records], dtype=object)
    test_y = np.array([row.get("tags", []) for row in test_ds.records], dtype=object)
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_targets = np.zeros((len(train_y), len(labels)), dtype=np.float32)
    test_targets = np.zeros((len(test_y), len(labels)), dtype=np.float32)
    for matrix, rows in ((train_targets, train_y), (test_targets, test_y)):
        for row_index, tags in enumerate(rows):
            for tag in tags:
                if str(tag).lower() in label_to_index:
                    matrix[row_index, label_to_index[str(tag).lower()]] = 1.0

    predictions = {
        "bert_only": train_probe(train_t, train_targets, test_t),
        "gnn_only": train_probe(train_g, train_targets, test_g),
        "early_concat": train_probe(np.concatenate([train_g, train_t], axis=1), train_targets, np.concatenate([test_g, test_t], axis=1)),
    }
    results = {name: evaluate_tagging(torch.logit(pred.clamp(1e-5, 1 - 1e-5)), torch.tensor(test_targets)) for name, pred in predictions.items()}

    fused = np.concatenate([test_g, test_t], axis=1)
    if len(fused) >= 3:
        embedding = TSNE(n_components=2, perplexity=min(5, len(fused) - 1), random_state=42, init="random").fit_transform(fused)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 5))
        colors = [int(np.argmax(test_targets[index])) if test_targets[index].any() else -1 for index in range(len(test_targets))]
        plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="tab20", s=55)
        for index, record in enumerate(test_records):
            plt.annotate(str(record["track_id"]), (embedding[index, 0], embedding[index, 1]), fontsize=8)
        plt.title("Task 3 multimodal embedding t-SNE")
        plt.tight_layout(); plt.savefig(args.plot, dpi=160); plt.close()
    case_studies = []
    for index, record in enumerate(test_records[:3]):
        graph = test_ds[index]["graph"]
        edge_preview = graph.edge_index[:, : min(10, graph.edge_index.size(1))].T.tolist()
        case_studies.append({
            "track_id": record["track_id"],
            "text_context": record.get("text_context", ""),
            "true_tags": record.get("tags", []),
            "predicted_top_tags": [labels[item] for item in torch.argsort(predictions["early_concat"][index], descending=True)[:5].tolist()],
            "graph_path": record["graph_path"],
            "num_nodes": record.get("num_nodes"),
            "num_edges": record.get("num_edges"),
            "graph_edge_preview": edge_preview,
            "text_tag_alignment": {"metadata_text": record.get("text_context", ""), "supervised_tags": record.get("tags", [])},
        })
    result = {"task": "task3", "split": "test", "samples": len(test_ds), "labels": labels, "ablations": results, "tsne_plot": str(args.plot).replace("\\", "/"), "case_studies": case_studies}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()