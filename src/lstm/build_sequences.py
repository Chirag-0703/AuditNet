"""Prepare sequence tensors for LSTM training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence

try:
    from src.config import PROCESSED_DIR, RAW_LEDGER_PATH, get_logger
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import PROCESSED_DIR, RAW_LEDGER_PATH, get_logger  # type: ignore  # noqa: E402

LOGGER = get_logger(__name__)
SEQUENCE_OUTPUT_PATH = PROCESSED_DIR / "sequences.pt"
FEATURE_COLS = ["amount", "timestamp_unix"]


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ledger file not found at {path}. Run simulate_data.py first.")
    dataframe = pd.read_csv(path)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    dataframe["timestamp_unix"] = dataframe["timestamp"].view("int64") // 10**9
    if "label" not in dataframe.columns:
        dataframe["label"] = (dataframe["anomaly_type"] != "normal").astype(int)
    return dataframe


def build_sequences(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Convert ledger rows into padded account sequences."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    sequence_tensors: List[torch.Tensor] = []
    sequence_labels: List[int] = []
    account_ids: List[str] = []

    for account, group in df.groupby("from_acct"):
        sequence = torch.tensor(group[feature_cols].values, dtype=torch.float)
        label = int(group["label"].any())
        sequence_tensors.append(sequence)
        sequence_labels.append(label)
        account_ids.append(account)

    padded_sequences = pad_sequence(sequence_tensors, batch_first=True)
    labels = torch.tensor(sequence_labels, dtype=torch.long)
    return padded_sequences, labels, account_ids, feature_cols


def save_sequences(
    sequences: torch.Tensor,
    labels: torch.Tensor,
    accounts: List[str],
    features: List[str],
    destination: Path = SEQUENCE_OUTPUT_PATH,
) -> None:
    """Persist sequence artifacts to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequences": sequences,
            "labels": labels,
            "account_ids": accounts,
            "feature_cols": features,
        },
        destination,
    )
    LOGGER.info("Saved sequence tensors to %s | shape=%s", destination, tuple(sequences.shape))


def main() -> None:
    """CLI entry-point."""
    ledger = _load_ledger(RAW_LEDGER_PATH)
    sequences, labels, accounts, features = build_sequences(ledger, FEATURE_COLS)
    save_sequences(sequences, labels, accounts, features)


if __name__ == "__main__":
    main()
