# src/gnn/build_graph.py

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
from collections import Counter

RAW_PATH = "data/raw/ledger.csv"
SAVE_PATH = "data/processed/graph.pt"
df = pd.read_csv(RAW_PATH)

# Timestamps coversion
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp_unix'] = df['timestamp'].astype(np.int64) // 10**9

# Label
df['label'] = df['anomaly_type'].apply(lambda x: 0 if x == 'normal' else 1)

# Account indexing
accounts = pd.Index(sorted(set(df['from_acct']) | set(df['to_acct'])))
acct2idx = {acct: i for i, acct in enumerate(accounts)}

# Edge index
edge_index = torch.tensor([
    [acct2idx[a] for a in df['from_acct']],
    [acct2idx[a] for a in df['to_acct']]
], dtype=torch.long)

# Edge features
edge_features = df[['amount', 'timestamp_unix']]
edge_attr = torch.tensor(StandardScaler().fit_transform(edge_features), dtype=torch.float)

# Node features
node_features = []
for acct in accounts:
    in_txns = df[df['to_acct'] == acct]
    out_txns = df[df['from_acct'] == acct]

    node_features.append([
        len(in_txns), len(out_txns),
        in_txns['amount'].mean() if not in_txns.empty else 0.0,
        out_txns['amount'].mean() if not out_txns.empty else 0.0,
        in_txns['amount'].std() if len(in_txns) > 1 else 0.0,
        out_txns['amount'].std() if len(out_txns) > 1 else 0.0,
    ])

x = torch.tensor(StandardScaler().fit_transform(node_features), dtype=torch.float)

# Labels
fraud_accounts = set(df[df['label'] == 1]['from_acct'].unique())
account_labels = {acct: int(acct in fraud_accounts) for acct in accounts}
y = torch.tensor([account_labels[acct] for acct in accounts], dtype=torch.long)

# PyG data
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Save graph
os.makedirs("data/processed", exist_ok=True)
torch.save(graph, SAVE_PATH)
print(f"Graph saved: {SAVE_PATH}")
print(f"Nodes: {graph.num_nodes} | Edges: {graph.num_edges}")
