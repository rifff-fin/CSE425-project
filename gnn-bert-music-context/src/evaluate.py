from __future__ import annotations

from typing import Dict, List

import torch


def evaluate_tag_predictions(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute a simple accuracy proxy for multilabel classification."""
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape}, targets {targets.shape}")
    preds = (torch.sigmoid(logits) > 0.5).float()
    accuracy = (preds == targets).float().mean().item()
    return float(accuracy)


def evaluate_regression(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean squared error for valence/arousal regression."""
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
    mse = torch.mean((predictions - targets) ** 2).item()
    return float(mse)


if __name__ == "__main__":
    logits = torch.randn(8, 20)
    targets = torch.randint(0, 2, (8, 20)).float()
    valence_pred = torch.randn(8, 1)
    valence_true = torch.randn(8, 1)

    print(f"Tag accuracy: {evaluate_tag_predictions(logits, targets):.4f}")
    print(f"Regression MSE: {evaluate_regression(valence_pred, valence_true):.4f}")
