from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bert_encoder import BertTextEncoder
from contrastive import DualEncoderContrastive
from fusion_model import CrossAttentionFusion
from gnn_model import GraphAudioEncoder


class ExampleDataset(Dataset):
    """Placeholder dataset for structured training samples."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


@dataclass
class TrainingConfig:
    task: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 2e-5
    alpha: float = 1.0
    beta: float = 0.5


def run_task_1(model: nn.Module, loader: DataLoader, device: str) -> None:
    """Placeholder training loop for Task 1: BERT tag classifier."""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        model.train()
        for batch in loader:
            texts = batch["texts"]
            labels = batch["labels"].to(device)
            _, cls_tokens = model(texts)
            logits = model.classifier(cls_tokens)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Task 1 | Epoch {epoch} | loss: {loss.item():.4f}")


def run_task_2(model: nn.Module, loader: DataLoader, device: str) -> None:
    """Placeholder training loop for Task 2: GNN audio encoder."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        model.train()
        for batch in loader:
            graph = batch["graph"]
            target = batch["target"].to(device)
            graph_embedding = model(graph)
            loss = criterion(graph_embedding, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Task 2 | Epoch {epoch} | loss: {loss.item():.4f}")


def run_task_3(model: nn.Module, loader: DataLoader, device: str, alpha: float = 1.0, beta: float = 0.5) -> None:
    """Placeholder training loop for Task 3: multi-task GNN-BERT fusion."""
    model.to(device)
    classification_loss = nn.BCEWithLogitsLoss()
    regression_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        model.train()
        for batch in loader:
            g = batch["graph_embedding"].to(device)
            text = batch["text_hidden"].to(device)
            labels = batch["tag_labels"].to(device)
            valence = batch["valence"].to(device)
            arousal = batch["arousal"].to(device)

            tag_logits, pred_valence, pred_arousal = model(g, text)
            loss_cls = classification_loss(tag_logits, labels)
            loss_reg = regression_loss(pred_valence, valence) + regression_loss(pred_arousal, arousal)
            loss = alpha * loss_cls + beta * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Task 3 | Epoch {epoch} | total_loss: {loss.item():.4f}")


def run_task_4(model: nn.Module, loader: DataLoader, device: str) -> None:
    """Placeholder training loop for Task 4: contrastive retrieval."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        model.train()
        for batch in loader:
            graph_emb = batch["graph_embedding"].to(device)
            text_emb = batch["text_hidden"].to(device)
            logits = model(graph_emb, text_emb)
            loss = model.info_nce_loss(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Task 4 | Epoch {epoch} | loss: {loss.item():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN-BERT music context models.")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3, 4], help="Task number to run.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(task=args.task, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    device = config.device

    dummy_samples = [{
        "texts": ["calm musical texture with soft synths"],
        "labels": torch.ones(1, 20),
        "graph": {"x": torch.randn(5, 128), "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])},
        "target": torch.randn(1, 128),
        "graph_embedding": torch.randn(1, 128),
        "text_hidden": torch.randn(1, 12, 768),
        "tag_labels": torch.ones(1, 20),
        "valence": torch.randn(1, 1),
        "arousal": torch.randn(1, 1),
    }]

    loader = DataLoader(ExampleDataset(dummy_samples), batch_size=config.batch_size, shuffle=True)

    if config.task == 1:
        bert_model = BertTextEncoder(model_name="distilbert-base-uncased")
        run_task_1(bert_model, loader, device)
    elif config.task == 2:
        gnn_model = GraphAudioEncoder(in_channels=128, hidden_dim=128, num_layers=2, model_type="sage")
        run_task_2(gnn_model, loader, device)
    elif config.task == 3:
        fusion_model = CrossAttentionFusion(graph_dim=128, text_dim=768, fusion_dim=256, num_heads=4, num_labels=20)
        run_task_3(fusion_model, loader, device, alpha=config.alpha, beta=config.beta)
    elif config.task == 4:
        contrastive_model = DualEncoderContrastive(graph_dim=128, text_dim=768, hidden_dim=256, tau=0.07)
        run_task_4(contrastive_model, loader, device)
    else:
        raise ValueError(f"Unsupported task: {config.task}")


if __name__ == "__main__":
    main()
