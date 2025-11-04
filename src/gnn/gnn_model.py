"""Graph neural network architecture for AuditNet."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "torch-geometric is required for the GNN model. "
        "Install the appropriate wheel from https://data.pyg.org/whl/."
    ) from exc


class GNN(torch.nn.Module):
    """Two-layer graph convolutional network with dropout."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1: Final[GCNConv] = GCNConv(in_channels, hidden_channels)
        self.conv2: Final[GCNConv] = GCNConv(hidden_channels, out_channels)
        self.dropout: float = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute logits for each node."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
