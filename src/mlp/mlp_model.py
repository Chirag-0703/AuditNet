"""Stacked multi-layer perceptron for ensemble predictions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMLP(nn.Module):
    """Two-layer MLP with dropout for stacked feature inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for stacked classifier inputs."""
        hidden = F.relu(self.fc1(features))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.relu(self.fc2(hidden))
        return self.out(hidden)
