# src/lstm/train_lstm.py

import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from lstm_model import LSTMClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load sequences
data = torch.load("data/processed/sequences.pt")
X = data['sequences']
y = data['labels']
feature_cols = data.get('feature_cols', ['amount', 'timestamp_unix'])

INPUT_DIM = len(feature_cols)

# Split
indices = torch.arange(len(y))
train_idx, val_test_idx, y_train, y_valtest = train_test_split(indices, y, test_size=0.2, stratify=y, random_state=SEED)
val_idx, test_idx, _, _ = train_test_split(val_test_idx, y_valtest, test_size=0.5, stratify=y_valtest, random_state=SEED)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(SequenceDataset(X[train_idx], y[train_idx]), batch_size=32, shuffle=True)
val_loader = DataLoader(SequenceDataset(X[val_idx], y[val_idx]), batch_size=32)
test_loader = DataLoader(SequenceDataset(X[test_idx], y[test_idx]), batch_size=32)


weights = compute_class_weight(class_weight="balanced", classes=np.unique(y.numpy()), y=y.numpy())
weights = torch.tensor(weights, dtype=torch.float)

# Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=INPUT_DIM, hidden_dim=64).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Training LSTM...")
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = loss_fn(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                trues.extend(y_batch.tolist())
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds)
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Val Acc = {acc:.4f}, Val F1 = {f1:.4f}")

# Final evaluation on test set
model.eval()
preds, trues = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        trues.extend(y_batch.tolist())

acc = accuracy_score(trues, preds)
f1 = f1_score(trues, preds)
cm = confusion_matrix(trues, preds)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1 Score : {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# Save Model 
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lstm_checkpoint.pt")
