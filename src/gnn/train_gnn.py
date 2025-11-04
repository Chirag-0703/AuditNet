"""Training routine for the AuditNet graph neural network."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    from src.config import (
        DEFAULT_RANDOM_SEED,
        GNN_TRAINING_PARAMS,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DIR,
        get_logger,
    )
    from src.utils.metrics import compute_classification_metrics
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import (  # type: ignore  # noqa: E402
        DEFAULT_RANDOM_SEED,
        GNN_TRAINING_PARAMS,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DIR,
        get_logger,
    )
    from src.utils.metrics import compute_classification_metrics  # type: ignore  # noqa: E402

from src.gnn.gnn_model import GNN

LOGGER = get_logger(__name__)
GRAPH_PATH = PROCESSED_DIR / "graph.pt"
LOSS_HISTORY_PATH = METRICS_DIR / "gnn_loss.pt"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "gnn_checkpoint.pt"
HIDDEN_CHANNELS = 64


def _load_graph(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Graph artifact not found at {path}. Run build_graph.py first.")
    graph = torch.load(path, weights_only=False)
    return graph.x, graph.edge_index, graph.edge_attr, graph.y


def _prepare_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    return device


def _split_indices(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.arange(labels.size(0))
    train_idx, val_test_idx, _, y_valtest = train_test_split(
        indices, labels, test_size=0.2, stratify=labels.cpu(), random_state=DEFAULT_RANDOM_SEED
    )
    val_idx, test_idx, _, _ = train_test_split(
        val_test_idx, y_valtest, test_size=0.5, stratify=y_valtest.cpu(), random_state=DEFAULT_RANDOM_SEED
    )
    return train_idx, val_idx, test_idx


def train() -> Dict[str, float]:
    """Train the GNN model and persist artifacts."""
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)

    x, edge_index, edge_attr, y = _load_graph(GRAPH_PATH)
    device = _prepare_device()
    x, edge_index, edge_attr, y = (
        x.to(device),
        edge_index.to(device),
        edge_attr.to(device),
        y.to(device),
    )

    train_idx, val_idx, test_idx = _split_indices(y)
    y_np = y.cpu().numpy()
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_np), y=y_np)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    model = GNN(in_channels=x.size(1), hidden_channels=HIDDEN_CHANNELS, out_channels=2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GNN_TRAINING_PARAMS["learning_rate"],
        weight_decay=GNN_TRAINING_PARAMS["weight_decay"],
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    train_losses = []
    for epoch in range(1, int(GNN_TRAINING_PARAMS["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        loss = loss_fn(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == GNN_TRAINING_PARAMS["epochs"]:
            model.eval()
            with torch.no_grad():
                val_logits = logits[val_idx]
                val_pred = val_logits.argmax(dim=1)
                metrics = compute_classification_metrics(y[val_idx].cpu().numpy(), val_pred.cpu().numpy())
            LOGGER.info(
                "Epoch %d | loss=%.4f | val_accuracy=%.4f | val_f1=%.4f",
                epoch,
                loss.item(),
                metrics["accuracy"],
                metrics["f1"],
            )

    model.eval()
    with torch.no_grad():
        test_logits = model(x, edge_index)
        test_pred = test_logits[test_idx].argmax(dim=1).cpu().numpy()
        test_true = y[test_idx].cpu().numpy()
        test_metrics = compute_classification_metrics(test_true, test_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    LOGGER.info("Saved model checkpoint to %s", MODEL_CHECKPOINT_PATH)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(train_losses, LOSS_HISTORY_PATH)
    LOGGER.info("Saved training loss history to %s", LOSS_HISTORY_PATH)

    return test_metrics


def main() -> None:
    """CLI entry-point."""
    metrics = train()
    LOGGER.info(
        "Final test metrics | accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )


if __name__ == "__main__":
    main()
