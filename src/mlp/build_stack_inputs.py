# src/mlp/build_stack_inputs.py

import torch
import torch.nn.functional as F
import os

GRAPH_PATH        = "data/processed/graph.pt"
SEQS_PATH         = "data/processed/sequences.pt"
GNN_MODEL_PATH    = "models/gnn_checkpoint.pt"
LSTM_MODEL_PATH   = "models/lstm_checkpoint.pt"
OUT_PATH          = "data/processed/stack_inputs.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Graph and GNN model
graph = torch.load(GRAPH_PATH,weights_only=False)
x, edge_index, y = graph.x.to(device), graph.edge_index.to(device), graph.y.to(device)

from src.gnn.gnn_model import GCN
gnn = GCN(in_channels=x.size(1), hidden_channels=64, out_channels=2).to(device)
gnn.load_state_dict(torch.load(GNN_MODEL_PATH))
gnn.eval()

# Load Sequences and LSTM model
seq_data = torch.load(SEQS_PATH)
sequences = seq_data['sequences'].to(device)
labels = seq_data['labels']
account_ids = seq_data['account_ids']  # Not used here but retained

from src.lstm.lstm_model import LSTMClassifier
lstm = LSTMClassifier(input_dim=sequences.size(2), hidden_dim=64).to(device)
lstm.load_state_dict(torch.load(LSTM_MODEL_PATH))
lstm.eval()

# Inference from both models
with torch.no_grad():
    probs_gnn  = F.softmax(gnn(x, edge_index), dim=1).cpu()
    probs_lstm = F.softmax(lstm(sequences), dim=1).cpu()

# Combine GNN and LSTM predictions as input features
stack_inputs = torch.cat([probs_gnn, probs_lstm], dim=1)
stack_labels = y.cpu()

# Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
torch.save({
    'features': stack_inputs,
    'labels': stack_labels
}, OUT_PATH)


print(f"Stacked inputs saved to {OUT_PATH}")
print(f"Features shape: {stack_inputs.shape}, Labels shape: {stack_labels.shape}")
