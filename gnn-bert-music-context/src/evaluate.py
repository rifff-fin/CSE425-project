from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve


def macro_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Macro-F1 for multilabel predictions."""
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape}, targets {targets.shape}")
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= threshold).astype(np.float32)
    targets_np = targets.detach().cpu().numpy().astype(np.float32)
    return float(f1_score(targets_np, preds, average="macro", zero_division=0))


def micro_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Micro-F1 for multilabel predictions."""
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape}, targets {targets.shape}")
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= threshold).astype(np.float32)
    targets_np = targets.detach().cpu().numpy().astype(np.float32)
    return float(f1_score(targets_np, preds, average="micro", zero_division=0))


def auc_pr(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Area under the Precision-Recall curve for multilabel classification."""
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape}, targets {targets.shape}")
    y_true = targets.detach().cpu().numpy().astype(np.float32).ravel()
    y_score = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    return float(average_precision_score(y_true, y_score))


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error for continuous emotion regression."""
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
    return float(torch.mean(torch.abs(predictions - targets)).item())


def recall_at_k(similarity: torch.Tensor, k_values: Iterable[int] = (1, 5, 10)) -> Dict[int, float]:
    """Compute Recall@K for retrieval tasks using a similarity matrix."""
    if similarity.dim() != 2:
        raise ValueError(f"Expected [B, B] similarity matrix, got {similarity.shape}")

    results: Dict[int, float] = {}
    device = similarity.device
    n = similarity.size(0)
    targets = torch.arange(n, device=device)
    ranking = torch.argsort(similarity, dim=1, descending=True)

    for k in k_values:
        hits = 0
        for i in range(n):
            retrieved = ranking[i, :k]
            if targets[i] in retrieved:
                hits += 1
        results[int(k)] = hits / max(1, n)
    return results


def evaluate_tagging(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Return all multi-label tagging metrics."""
    return {
        "macro_f1": macro_f1(logits, targets),
        "micro_f1": micro_f1(logits, targets),
        "auc_pr": auc_pr(logits, targets),
    }


def evaluate_regression(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Return regression metric summaries."""
    return {"mae": mae(predictions, targets)}


def evaluate_retrieval(similarity: torch.Tensor, k_values: Iterable[int] = (1, 5, 10)) -> Dict[str, float]:
    """Return retrieval Recall@K metrics."""
    scores = recall_at_k(similarity, k_values=k_values)
    return {f"R@{k}": float(v) for k, v in scores.items()}


if __name__ == "__main__":
    logits = torch.randn(8, 20)
    targets = torch.randint(0, 2, (8, 20)).float()
    valence_pred = torch.randn(8, 1)
    valence_true = torch.randn(8, 1)
    sim = torch.randn(8, 8)

    print("Tagging metrics:", evaluate_tagging(logits, targets))
    print("Regression metrics:", evaluate_regression(valence_pred, valence_true))
    print("Retrieval metrics:", evaluate_retrieval(sim, k_values=(1, 5, 10)))
