"""LSTM-based classifier for sequential ledger data."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Simple LSTM classifier followed by a dense projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return class logits for each batch element."""
        _, (hidden, _) = self.lstm(sequences)
        return self.fc(hidden[-1])
