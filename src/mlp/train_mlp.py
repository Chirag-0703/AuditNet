"""Training pipeline for the stacked MLP ensemble in AuditNet."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    from src.config import (
        DEFAULT_RANDOM_SEED,
        METRICS_DIR,
        MODELS_DIR,
        MLP_TRAINING_PARAMS,
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
        METRICS_DIR,
        MODELS_DIR,
        MLP_TRAINING_PARAMS,
        PROCESSED_DIR,
        get_logger,
    )
    from src.utils.metrics import compute_classification_metrics  # type: ignore  # noqa: E402

from src.mlp.mlp_model import MetaMLP

LOGGER = get_logger(__name__)
STACK_INPUTS_PATH = PROCESSED_DIR / "stack_inputs.pt"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "mlp_checkpoint.pt"
LOSS_HISTORY_PATH = METRICS_DIR / "stacked_mlp_loss.pt"
HIDDEN_DIM = 8


def _load_stack_inputs(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Stacked input tensor missing at {path}. Run build_stack_inputs.py first.")
    data = torch.load(path, map_location="cpu")
    if "features" not in data or "labels" not in data:
        raise ValueError("Stack inputs file missing 'features' and 'labels'.")
    return data["features"], data["labels"]


def train() -> Dict[str, float]:
    """Train the stacked MLP model."""
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)

    features, labels = _load_stack_inputs(STACK_INPUTS_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    features, labels = features.to(device), labels.to(device)

    indices = torch.arange(len(labels))
    train_idx, val_test_idx, y_train, y_valtest = train_test_split(
        indices.cpu(),
        labels.cpu(),
        test_size=0.2,
        stratify=labels.cpu(),
        random_state=DEFAULT_RANDOM_SEED,
    )
    val_idx, test_idx, _, _ = train_test_split(
        val_test_idx,
        y_valtest,
        test_size=0.5,
        stratify=y_valtest,
        random_state=DEFAULT_RANDOM_SEED,
    )
    train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)

    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels.cpu().numpy())
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = MetaMLP(input_dim=features.size(1), hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_TRAINING_PARAMS["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    train_losses = []
    for epoch in range(1, int(MLP_TRAINING_PARAMS["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[train_idx])
        loss = criterion(logits, labels[train_idx])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == MLP_TRAINING_PARAMS["epochs"]:
            model.eval()
            with torch.no_grad():
                val_logits = model(features[val_idx])
                val_pred = val_logits.argmax(dim=1).cpu().numpy()
                val_true = labels[val_idx].cpu().numpy()
                metrics = compute_classification_metrics(val_true, val_pred)
            LOGGER.info(
                "Epoch %d | loss=%.4f | val_accuracy=%.4f | val_f1=%.4f",
                epoch,
                loss.item(),
                metrics["accuracy"],
                metrics["f1"],
            )

    model.eval()
    with torch.no_grad():
        test_logits = model(features[test_idx])
        test_pred = test_logits.argmax(dim=1).cpu().numpy()
        test_true = labels[test_idx].cpu().numpy()
        test_metrics = compute_classification_metrics(test_true, test_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    LOGGER.info("Saved stacked MLP checkpoint to %s", MODEL_CHECKPOINT_PATH)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(train_losses, LOSS_HISTORY_PATH)
    LOGGER.info("Persisted training loss history to %s", LOSS_HISTORY_PATH)

    return test_metrics


def main() -> None:
    """CLI entry-point."""
    metrics = train()
    LOGGER.info(
        "Final stacked MLP metrics | accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )


if __name__ == "__main__":
    main()
