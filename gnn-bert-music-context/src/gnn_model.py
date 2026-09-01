from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, SAGEConv, global_mean_pool


class MusicGNNEncoder(nn.Module):
    """PyTorch Geometric encoder for graph-based music representation learning."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        model_type: str = "sage",
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()

        current_dim = in_channels
        for _ in range(num_layers):
            if self.model_type == "gat":
                conv = GATv2Conv(current_dim, hidden_dim, heads=heads, dropout=dropout)
                self.convs.append(conv)
                current_dim = hidden_dim * heads
            else:
                conv = SAGEConv(current_dim, hidden_dim)
                self.convs.append(conv)
                current_dim = hidden_dim

        self.output_proj = nn.Linear(current_dim, hidden_dim)

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """Process PyG graph batches and aggregate node states into track embedding g."""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") else None

        if x.dim() != 2:
            raise ValueError(f"Expected node feature tensor [N, F], got {x.shape}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Expected edge_index [2, E], got {edge_index.shape}")

        for conv in self.convs:
            if self.model_type == "gat":
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

            assert x.dim() == 2, f"Node features must remain [N, F], got {x.shape}"

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        g = global_mean_pool(x, batch)
        g = self.output_proj(g)

        assert g.dim() == 2, f"g must be [B, hidden_dim], got {g.shape}"
        assert g.size(-1) == self.hidden_dim, (
            f"Unexpected pooled embedding dim: {g.size(-1)} != {self.hidden_dim}"
        )

        return g


if __name__ == "__main__":
    from torch_geometric.data import Data

    x = torch.randn(8, 32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    model = MusicGNNEncoder(in_channels=32, hidden_dim=64, num_layers=2, model_type="sage")
    g = model(data)
    print(f"g shape: {g.shape}")
