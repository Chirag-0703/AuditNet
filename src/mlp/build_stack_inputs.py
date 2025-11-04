"""Build stacked model inputs by combining GNN and LSTM predictions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

try:
    from src.config import PROCESSED_DIR, MODELS_DIR, get_logger  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import PROCESSED_DIR, MODELS_DIR, get_logger  # type: ignore  # noqa: E402
from src.gnn.gnn_model import GNN  # noqa: E402
from src.lstm.lstm_model import LSTMClassifier  # noqa: E402

LOGGER = get_logger(__name__)
GRAPH_PATH = PROCESSED_DIR / "graph.pt"
SEQUENCES_PATH = PROCESSED_DIR / "sequences.pt"
STACK_OUTPUT_PATH = PROCESSED_DIR / "stack_inputs.pt"
GNN_MODEL_PATH = MODELS_DIR / "gnn_checkpoint.pt"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_checkpoint.pt"
HIDDEN_DIM = 64


def _load_graph() -> Dict[str, torch.Tensor]:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph artifact not found at {GRAPH_PATH}. Run gnn/build_graph.py first.")
    graph = torch.load(GRAPH_PATH, weights_only=False, map_location="cpu")
    if not all(hasattr(graph, attr) for attr in ("x", "edge_index", "y")):
        raise ValueError("Graph artifact missing required attributes.")
    return {"x": graph.x, "edge_index": graph.edge_index, "y": graph.y}


def _load_sequences() -> Dict[str, torch.Tensor]:
    if not SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"Sequence data not found at {SEQUENCES_PATH}. Run lstm preprocessing first.")
    data = torch.load(SEQUENCES_PATH, map_location="cpu")
    required = {"sequences", "labels"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Sequence tensor missing keys: {', '.join(sorted(missing))}")
    return data


def build_stack_inputs() -> None:
    """Combine GNN and LSTM probabilities into stacked model features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    graph = _load_graph()
    seq_data = _load_sequences()

    gnn = GNN(in_channels=graph["x"].size(1), hidden_channels=HIDDEN_DIM, out_channels=2).to(device)
    if not GNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"GNN checkpoint not found at {GNN_MODEL_PATH}. Train the model first.")
    gnn.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device))
    gnn.eval()

    lstm = LSTMClassifier(input_dim=seq_data["sequences"].size(2), hidden_dim=HIDDEN_DIM).to(device)
    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found at {LSTM_MODEL_PATH}. Train the model first.")
    lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    lstm.eval()

    with torch.no_grad():
        probs_gnn = F.softmax(gnn(graph["x"].to(device), graph["edge_index"].to(device)), dim=1).cpu()
        probs_lstm = F.softmax(lstm(seq_data["sequences"].to(device)), dim=1).cpu()

    features = torch.cat([probs_gnn, probs_lstm], dim=1)
    labels = graph["y"]

    STACK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels}, STACK_OUTPUT_PATH)
    LOGGER.info(
        "Saved stacked inputs to %s | features_shape=%s",
        STACK_OUTPUT_PATH,
        tuple(features.shape),
    )


def main() -> None:
    """CLI entry-point."""
    build_stack_inputs()


if __name__ == "__main__":
    main()
