"""Build PyTorch Geometric graph artifacts from the synthetic ledger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

try:
    from src.config import PROCESSED_DIR, RAW_LEDGER_PATH, get_logger
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import PROCESSED_DIR, RAW_LEDGER_PATH, get_logger  # type: ignore  # noqa: E402

try:
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError(
        "torch-geometric is required to build the transaction graph. "
        "Install the appropriate wheel from https://data.pyg.org/whl/."
    ) from exc

LOGGER = get_logger(__name__)
GRAPH_SAVE_PATH: Path = PROCESSED_DIR / "graph.pt"


@dataclass(frozen=True)
class GraphArtifacts:
    """Container for PyG graph components."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor


def _load_ledger(path: Path) -> pd.DataFrame:
    """Load the ledger CSV and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Ledger file not found at {path}. Run simulate_data.py first.")

    ledger = pd.read_csv(path)
    expected_columns = {"txn_id", "timestamp", "from_acct", "to_acct", "amount", "anomaly_type", "label"}
    missing = expected_columns - set(ledger.columns)
    if missing:
        raise ValueError(f"Ledger missing required columns: {', '.join(sorted(missing))}")

    ledger["timestamp"] = pd.to_datetime(ledger["timestamp"])
    ledger["timestamp_unix"] = ledger["timestamp"].view("int64") // 10**9
    return ledger


def _build_accounts(df: pd.DataFrame) -> Tuple[pd.Index, Dict[str, int]]:
    accounts = pd.Index(sorted(set(df["from_acct"]) | set(df["to_acct"])))
    acct_to_idx = {acct: idx for idx, acct in enumerate(accounts)}
    return accounts, acct_to_idx


def _build_edge_index(df: pd.DataFrame, mapping: Dict[str, int]) -> torch.Tensor:
    sources = [mapping[acct] for acct in df["from_acct"]]
    targets = [mapping[acct] for acct in df["to_acct"]]
    return torch.tensor([sources, targets], dtype=torch.long)


def _build_edge_features(df: pd.DataFrame) -> torch.Tensor:
    scaler = StandardScaler()
    features = df[["amount", "timestamp_unix"]]
    scaled = scaler.fit_transform(features)
    return torch.tensor(scaled, dtype=torch.float)


def _build_node_features(df: pd.DataFrame, accounts: pd.Index) -> torch.Tensor:
    feature_rows = []
    for account in accounts:
        inbound = df[df["to_acct"] == account]
        outbound = df[df["from_acct"] == account]
        feature_rows.append(
            [
                len(inbound),
                len(outbound),
                inbound["amount"].mean() if not inbound.empty else 0.0,
                outbound["amount"].mean() if not outbound.empty else 0.0,
                inbound["amount"].std(ddof=0) if len(inbound) > 1 else 0.0,
                outbound["amount"].std(ddof=0) if len(outbound) > 1 else 0.0,
            ]
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_rows)
    return torch.tensor(scaled, dtype=torch.float)


def _build_labels(df: pd.DataFrame, accounts: pd.Index) -> torch.Tensor:
    fraud_accounts = set(df.loc[df["label"] == 1, "from_acct"].unique())
    labels = [int(account in fraud_accounts) for account in accounts]
    return torch.tensor(labels, dtype=torch.long)


def build_pyg_graph(path: Path = RAW_LEDGER_PATH) -> Data:
    """Construct a PyG Data object from the ledger."""
    ledger_df = _load_ledger(path)
    accounts, mapping = _build_accounts(ledger_df)

    edge_index = _build_edge_index(ledger_df, mapping)
    edge_attr = _build_edge_features(ledger_df)
    node_features = _build_node_features(ledger_df, accounts)
    labels = _build_labels(ledger_df, accounts)

    LOGGER.info(
        "Constructed graph | nodes=%d | edges=%d",
        node_features.size(0),
        edge_index.size(1),
    )
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)


def save_graph(graph: Data, destination: Path = GRAPH_SAVE_PATH) -> None:
    """Persist the graph artifact to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, destination)
    LOGGER.info("Saved graph artifact to %s", destination)


def main() -> None:
    """Entry-point for CLI usage."""
    graph = build_pyg_graph()
    save_graph(graph)


if __name__ == "__main__":
    main()
