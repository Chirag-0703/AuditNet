# src/mlp/train_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random
import os
from mlp_model import MetaMLP

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load stacked inputs
data = torch.load("data/processed/stack_inputs.pt")
X = data['features'].to(device) 
y = data['labels'].to(device)    

# Stratified split
indices = torch.arange(len(y))
train_idx, val_test_idx, y_train, y_valtest = train_test_split(
    indices.cpu(), y.cpu(), test_size=0.2, stratify=y.cpu(), random_state=SEED
)
val_idx, test_idx, _, _ = train_test_split(
    val_test_idx, y_valtest, test_size=0.5, stratify=y_valtest, random_state=SEED
)
train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)

y_np = y.cpu().numpy()
weights_np = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_np)
weights = torch.tensor(weights_np, dtype=torch.float32).to(device)

# Model Training
model = MetaMLP(input_dim=4, hidden_dim=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(weight=weights)
print("🚀 Training stacked MLP...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    logits = model(X[train_idx])
    loss = criterion(logits, y[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X[val_idx])
            val_preds = val_logits.argmax(dim=1)
            val_labels = y[val_idx]
            acc = accuracy_score(val_labels.cpu(), val_preds.cpu())
            f1 = f1_score(val_labels.cpu(), val_preds.cpu())
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

# Final evaluation on test set
model.eval()
with torch.no_grad():
    test_logits = model(X[test_idx])
    test_preds = test_logits.argmax(dim=1)
    test_labels = y[test_idx]
    test_acc = accuracy_score(test_labels.cpu(), test_preds.cpu())
    test_f1 = f1_score(test_labels.cpu(), test_preds.cpu())
    test_cm = confusion_matrix(test_labels.cpu(), test_preds.cpu())

print(f"\nFinal Test Accuracy: {test_acc:.4f} | F1 Score: {test_f1:.4f}")
print(f"Confusion Matrix:\n{test_cm}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mlp_checkpoint.pt")