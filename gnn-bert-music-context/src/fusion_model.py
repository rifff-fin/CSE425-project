from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class GNNBERTFusionModel(nn.Module):
    """Cross-attention fusion model combining graph and text representations."""

    def __init__(
        self,
        graph_dim: int,
        text_dim: int,
        fusion_dim: int = 256,
        num_heads: int = 4,
        num_tags: int = 20,
    ) -> None:
        super().__init__()
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim

        self.g_proj = nn.Linear(graph_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)

        self.tag_head = nn.Linear(fusion_dim, num_tags)
        self.valence_head = nn.Linear(fusion_dim, 1)
        self.arousal_head = nn.Linear(fusion_dim, 1)

    def forward(self, g: torch.Tensor, H_text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute tag logits and valence/arousal regression predictions from g and H_text."""
        if g.dim() != 2:
            raise ValueError(f"Expected graph embedding [B, d_g], got {g.shape}")
        if H_text.dim() != 3:
            raise ValueError(f"Expected text hidden states [B, L, d_t], got {H_text.shape}")

        batch_size, seq_len, _ = H_text.shape
        assert g.size(0) == batch_size, "Batch mismatch between g and H_text"

        g_expanded = g.unsqueeze(1).repeat(1, seq_len, 1)
        q = self.g_proj(g_expanded)
        k = self.text_proj(H_text)
        v = self.text_proj(H_text)

        fused, _ = self.cross_attn(query=q, key=k, value=v)
        fused = self.norm(fused)
        pooled = fused.mean(dim=1)

        tag_logits = self.tag_head(pooled)
        valence = self.valence_head(pooled)
        arousal = self.arousal_head(pooled)

        assert tag_logits.ndim == 2, f"tag_logits must be [B, num_tags], got {tag_logits.shape}"
        assert valence.ndim == 2 and valence.size(-1) == 1, (
            f"valence must be [B, 1], got {valence.shape}"
        )
        assert arousal.ndim == 2 and arousal.size(-1) == 1, (
            f"arousal must be [B, 1], got {arousal.shape}"
        )

        return tag_logits, valence, arousal


if __name__ == "__main__":
    g = torch.randn(4, 128)
    H_text = torch.randn(4, 16, 768)
    model = GNNBERTFusionModel(graph_dim=128, text_dim=768, fusion_dim=256, num_tags=20)
    logits, valence, arousal = model(g, H_text)
    print(f"tag_logits: {logits.shape}")
    print(f"valence: {valence.shape}")
    print(f"arousal: {arousal.shape}")
