from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch

from bert_encoder import BERTTextEncoder
from contrastive import ContrastiveDualEncoder
from fusion_model import GNNBERTFusionModel
from gnn_model import MusicGNNEncoder
from fma_dataset import FMAGraphDataset, load_fma_label_names
from fma_text_dataset import FMATextDataset


class MusicDataset(Dataset):
    """Minimal task-specific dataset wrapper for local development and experimentation."""

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_task_samples(samples: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    """Batch tensors and PyG graphs while preserving text lists."""
    batch: Dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        if key == "graph":
            if task == "task2":
                batch[key] = Batch.from_data_list(values)
            continue
        if key == "texts":
            batch[key] = [text for value in values for text in value]
            continue
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch


@dataclass
class TrainingConfig:
    task: str = "task1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    alpha: float = 1.0
    beta: float = 0.5
    tau: float = 0.07
    checkpoint_dir: str = "./checkpoints"


def make_linear_warmup_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Construct a simple linear warmup scheduler."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps + 1) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_task_1(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: TrainingConfig) -> None:
    """Fine-tune BERT for multi-label tagging with BCEWithLogitsLoss."""
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = make_linear_warmup_scheduler(optimizer, total_steps=total_steps, warmup_steps=config.warmup_steps)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            texts = batch["texts"]
            labels = batch["tags"].to(config.device)
            _, t = model(texts)
            logits = model.classifier(t)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                texts = batch["texts"]
                labels = batch["tags"].to(config.device)
                _, t = model(texts)
                logits = model.classifier(t)
                val_loss += criterion(logits, labels).item()

        val_loss = val_loss / max(1, len(val_loader))
        print(f"[Task 1] Epoch {epoch}/{config.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f"task1_epoch_{epoch}.pt"))


def run_task_2(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: TrainingConfig) -> None:
    """Train graph encoder with BCEWithLogitsLoss over segment graph classification targets."""
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = make_linear_warmup_scheduler(optimizer, total_steps=total_steps, warmup_steps=config.warmup_steps)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            graph = batch["graph"]
            graph = graph.to(config.device)
            labels = batch["tags"].to(config.device)
            embedding = model(graph)
            logits = model.classifier(embedding)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                graph = batch["graph"].to(config.device)
                labels = batch["tags"].to(config.device)
                embedding = model(graph)
                logits = model.classifier(embedding)
                val_loss += criterion(logits, labels).item()

        val_loss = val_loss / max(1, len(val_loader))
        print(f"[Task 2] Epoch {epoch}/{config.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f"task2_epoch_{epoch}.pt"))


def run_task_3(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: TrainingConfig) -> None:
    """Jointly train fusion model with multi-task tag + emotion regression objective."""
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = make_linear_warmup_scheduler(optimizer, total_steps=total_steps, warmup_steps=config.warmup_steps)

    tag_criterion = nn.BCEWithLogitsLoss()
    regression_criterion = nn.MSELoss()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            g = batch["g"].to(config.device)
            H_text = batch["H_text"].to(config.device)
            tag_labels = batch["tag_labels"].to(config.device)
            valence = batch["valence"].to(config.device)
            arousal = batch["arousal"].to(config.device)

            tag_logits, valence_pred, arousal_pred = model(g, H_text)
            tag_loss = tag_criterion(tag_logits, tag_labels)
            reg_loss = regression_criterion(valence_pred, valence) + regression_criterion(arousal_pred, arousal)
            loss = config.alpha * tag_loss + config.beta * reg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                g = batch["g"].to(config.device)
                H_text = batch["H_text"].to(config.device)
                tag_labels = batch["tag_labels"].to(config.device)
                valence = batch["valence"].to(config.device)
                arousal = batch["arousal"].to(config.device)

                tag_logits, valence_pred, arousal_pred = model(g, H_text)
                tag_loss = tag_criterion(tag_logits, tag_labels)
                reg_loss = regression_criterion(valence_pred, valence) + regression_criterion(arousal_pred, arousal)
                val_loss += (config.alpha * tag_loss + config.beta * reg_loss).item()

        val_loss = val_loss / max(1, len(val_loader))
        print(f"[Task 3] Epoch {epoch}/{config.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f"task3_epoch_{epoch}.pt"))


def run_task_4(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: TrainingConfig) -> None:
    """Train the dual encoder with InfoNCE retrieval objective."""
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = make_linear_warmup_scheduler(optimizer, total_steps=total_steps, warmup_steps=config.warmup_steps)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            g = batch["g"].to(config.device)
            H_text = batch["H_text"].to(config.device)
            logits = model(g, H_text)
            loss = model.info_nce_loss(logits)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                g = batch["g"].to(config.device)
                H_text = batch["H_text"].to(config.device)
                logits = model(g, H_text)
                val_loss += model.info_nce_loss(logits).item()

        val_loss = val_loss / max(1, len(val_loader))
        print(f"[Task 4] Epoch {epoch}/{config.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f"task4_epoch_{epoch}.pt"))


def build_dummy_task_data(task: str) -> Tuple[MusicDataset, MusicDataset]:
    """Create small synthetic datasets for pipeline testing and code validation."""
    samples: List[Dict[str, Any]] = []
    for i in range(12):
        sample = {
            "texts": [f"music sample {i} with warm smooth melody"],
            "tags": torch.randint(0, 2, (20,), dtype=torch.float32),
            "g": torch.randn(128, dtype=torch.float32),
            "H_text": torch.randn(8, 768, dtype=torch.float32),
            "tag_labels": torch.randint(0, 2, (20,), dtype=torch.float32),
            "valence": torch.tensor([float(i % 5) / 5.0], dtype=torch.float32),
            "arousal": torch.tensor([float((i + 2) % 6) / 6.0], dtype=torch.float32),
            "graph": Data(
                x=torch.randn(6, 32),
                edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long),
                batch=torch.zeros(6, dtype=torch.long),
            ),
        }
        samples.append(sample)

    split = int(0.8 * len(samples))
    return MusicDataset(samples[:split]), MusicDataset(samples[split:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training loop for music-context ML tasks.")
    parser.add_argument("--task", type=str, default="task1", choices=["task1", "task2", "task3", "task4"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--real-data", action="store_true", help="Use processed FMA data for Task 1 or Task 2.")
    parser.add_argument("--manifest-root", type=str, default="./data/splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.real_data and args.task not in {"task1", "task2"}:
        raise ValueError("--real-data is currently supported for task1 and task2.")

    if args.real_data:
        manifest_root = os.path.abspath(args.manifest_root)
        label_names = load_fma_label_names(manifest_root)
        if args.task == "task1":
            train_ds = FMATextDataset(os.path.join(manifest_root, "train.json"), label_names)
            val_ds = FMATextDataset(os.path.join(manifest_root, "val.json"), label_names)
        else:
            train_ds = FMAGraphDataset(os.path.join(manifest_root, "train.json"), label_names)
            val_ds = FMAGraphDataset(os.path.join(manifest_root, "val.json"), label_names)
    else:
        train_ds, val_ds = build_dummy_task_data(args.task)
    collate_fn = lambda samples: collate_task_samples(samples, args.task)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.task == "task1":
        model = BERTTextEncoder(model_name="distilbert-base-uncased")
        num_tags = len(label_names) if args.real_data else 20
        model.classifier = nn.Linear(768, num_tags)
        run_task_1(model, train_loader, val_loader, config)
    elif args.task == "task2":
        model = MusicGNNEncoder(in_channels=32, hidden_dim=128, num_layers=2, model_type="sage")
        num_tags = len(label_names) if args.real_data else 20
        model.classifier = nn.Linear(128, num_tags)
        run_task_2(model, train_loader, val_loader, config)
    elif args.task == "task3":
        model = GNNBERTFusionModel(graph_dim=128, text_dim=768, fusion_dim=256, num_tags=20)
        run_task_3(model, train_loader, val_loader, config)
    elif args.task == "task4":
        model = ContrastiveDualEncoder(g_dim=128, text_dim=768, embed_dim=256, tau=config.tau)
        run_task_4(model, train_loader, val_loader, config)
    else:
        raise ValueError(f"Unsupported task selection: {args.task}")


if __name__ == "__main__":
    main()
