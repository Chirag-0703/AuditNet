# src/gnn/train_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random
import os
from gnn_model import GNN 

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load graph
graph = torch.load("data/processed/graph.pt", weights_only=False)
x, edge_index, edge_attr, y = graph.x, graph.edge_index, graph.edge_attr, graph.y
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, edge_index, edge_attr, y = x.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)

# Split
indices = torch.arange(y.size(0))
train_idx, val_test_idx, y_train, y_valtest = train_test_split(
    indices, y, test_size=0.2, stratify=y.cpu(), random_state=SEED
)
val_idx, test_idx, _, _ = train_test_split(
    val_test_idx, y_valtest, test_size=0.5, stratify=y_valtest.cpu(), random_state=SEED
)

y_np = y.cpu().numpy()
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_np), y=y_np)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Model Training
model = GNN(in_channels=x.size(1), hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss(weight=weights)
print("Training GNN...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = loss_fn(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = out[val_idx].argmax(dim=1)
            val_true = y[val_idx]
            acc = accuracy_score(val_true.cpu(), val_pred.cpu())
            f1 = f1_score(val_true.cpu(), val_pred.cpu(), zero_division=0)
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

# Final evaluation on test set
model.eval()
with torch.no_grad():
    out = model(x, edge_index)
    test_pred = out[test_idx].argmax(dim=1)
    test_true = y[test_idx]
    acc = accuracy_score(test_true.cp
    u(), test_pred.cpu())
    f1 = f1_score(test_true.cpu(), test_pred.cpu(), zero_division=0)
    cm = confusion_matrix(test_true.cpu(), test_pred.cpu())

print(f"\nFinal Test Accuracy: {acc:.4f}")
print(f"Final Test F1 Score : {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/gnn_checkpoint.pt")
