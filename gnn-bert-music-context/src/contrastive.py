from __future__ import annotations

import torch
from torch import nn


class ContrastiveDualEncoder(nn.Module):
    """Dual encoder for graph-text retrieval using normalized cosine similarity and InfoNCE."""

    def __init__(self, g_dim: int, text_dim: int, embed_dim: int = 256, tau: float = 0.07) -> None:
        super().__init__()
        self.g_dim = g_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.tau = tau

        self.g_proj = nn.Sequential(
            nn.Linear(g_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the feature dimension."""
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    def forward(self, g: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarities between graph and caption embeddings."""
        if g.dim() != 2:
            raise ValueError(f"Expected graph embedding [B, d_g], got {g.shape}")
        if H_text.dim() != 3:
            raise ValueError(f"Expected text hidden states [B, L, d_t], got {H_text.shape}")

        graph_emb = self.g_proj(g)
        text_emb = self.text_proj(H_text.mean(dim=1))

        assert graph_emb.shape[0] == text_emb.shape[0], "Batch mismatch in dual encoder"
        graph_emb = self._normalize(graph_emb)
        text_emb = self._normalize(text_emb)

        logits = (graph_emb @ text_emb.T) / self.tau
        assert logits.dim() == 2 and logits.shape[0] == logits.shape[1], (
            f"Expected [B, B] similarity matrix, got {logits.shape}"
        )
        return logits

    def info_nce_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute symmetric InfoNCE loss using a square similarity matrix."""
        if logits.dim() != 2 or logits.shape[0] != logits.shape[1]:
            raise ValueError(f"Expected square logits [B, B], got {logits.shape}")

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2.0


if __name__ == "__main__":
    g = torch.randn(8, 128)
    H_text = torch.randn(8, 16, 768)
    model = ContrastiveDualEncoder(g_dim=128, text_dim=768, embed_dim=256, tau=0.07)
    logits = model(g, H_text)
    loss = model.info_nce_loss(logits)
    print(f"logits shape: {logits.shape}")
    print(f"loss: {loss.item():.4f}")
