# src/lstm/lstm_model.py

import torch
import torch.nn as nn

# Define the LSTM model 
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
