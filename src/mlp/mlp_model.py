# src/mlp/mlp_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the stacked MLP model
class MetaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
