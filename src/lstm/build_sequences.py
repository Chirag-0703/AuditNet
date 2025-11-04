# src/lstm/build_sequences.py

import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/ledger.csv"
SAVE_PATH = "data/processed/sequences.pt"
FEATURE_COLS = ['amount', 'timestamp_unix']  

def preprocess_ledger(path: str):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_unix'] = df['timestamp'].astype(np.int64) // 10**9
    df['label'] = df['anomaly_type'].apply(lambda x: 0 if x == 'normal' else 1)
    return df

def build_sequences(df: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    sequence_data, sequence_labels, account_ids = [], [], []
    grouped = df.groupby('from_acct')

    for acct, group in grouped:
        sequence = torch.tensor(group[feature_cols].values, dtype=torch.float)
        label = 1 if group['label'].any() else 0
        sequence_data.append(sequence)
        sequence_labels.append(label)
        account_ids.append(acct)

    padded = pad_sequence(sequence_data, batch_first=True)
    labels = torch.tensor(sequence_labels, dtype=torch.long)

    return padded, labels, account_ids, feature_cols

if __name__ == "__main__":
    df = preprocess_ledger(RAW_PATH)
    padded, labels, account_ids, features = build_sequences(df, FEATURE_COLS)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        'sequences': padded,
        'labels': labels,
        'account_ids': account_ids,
        'feature_cols': features
    }, SAVE_PATH)

    print(f"Saved sequences to {SAVE_PATH}, shape = {padded.shape}")
