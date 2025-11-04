"""Training pipeline for the AuditNet LSTM sequence model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

try:
    from src.config import (
        DEFAULT_RANDOM_SEED,
        LSTM_TRAINING_PARAMS,
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
        LSTM_TRAINING_PARAMS,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DIR,
        get_logger,
    )
    from src.utils.metrics import compute_classification_metrics  # type: ignore  # noqa: E402

try:
    from src.lstm.lstm_model import LSTMClassifier
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.lstm.lstm_model import LSTMClassifier  # type: ignore  # noqa: E402

LOGGER = get_logger(__name__)
SEQUENCES_PATH = PROCESSED_DIR / "sequences.pt"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "lstm_checkpoint.pt"
LOSS_HISTORY_PATH = METRICS_DIR / "lstm_loss.pt"
HIDDEN_DIM = 64


@dataclass(frozen=True)
class SequenceBatch:
    """Container for sequence tensors."""

    sequences: torch.Tensor
    labels: torch.Tensor


class SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset for batched sequence training."""

    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor) -> None:
        self._sequences = sequences
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._sequences[index], self._labels[index]


def _load_sequences(path: Path) -> SequenceBatch:
    if not path.exists():
        raise FileNotFoundError(f"Sequence tensor not found at {path}. Run the preprocessing steps first.")
    data = torch.load(path, map_location="cpu")
    if "sequences" not in data or "labels" not in data:
        raise ValueError("Sequences file missing required keys: 'sequences' and 'labels'.")
    return SequenceBatch(sequences=data["sequences"], labels=data["labels"])


def train() -> Dict[str, float]:
    """Train the LSTM model and persist artifacts."""
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)

    batch = _load_sequences(SEQUENCES_PATH)
    feature_cols = len(batch.sequences[0][0])

    indices = torch.arange(len(batch.labels))
    train_idx, val_test_idx, y_train, y_valtest = train_test_split(
        indices,
        batch.labels,
        test_size=0.2,
        stratify=batch.labels,
        random_state=DEFAULT_RANDOM_SEED,
    )
    val_idx, test_idx, _, _ = train_test_split(
        val_test_idx,
        y_valtest,
        test_size=0.5,
        stratify=y_valtest,
        random_state=DEFAULT_RANDOM_SEED,
    )

    train_ds = SequenceDataset(batch.sequences[train_idx], batch.labels[train_idx])
    val_ds = SequenceDataset(batch.sequences[val_idx], batch.labels[val_idx])
    test_ds = SequenceDataset(batch.sequences[test_idx], batch.labels[test_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(LSTM_TRAINING_PARAMS["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(val_ds, batch_size=int(LSTM_TRAINING_PARAMS["batch_size"]))
    test_loader = DataLoader(test_ds, batch_size=int(LSTM_TRAINING_PARAMS["batch_size"]))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(batch.labels.numpy()),
        y=batch.labels.numpy(),
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = LSTMClassifier(input_dim=feature_cols, hidden_dim=HIDDEN_DIM).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    for epoch in range(1, int(LSTM_TRAINING_PARAMS["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequences)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(epoch_loss)

        if epoch % 10 == 0 or epoch == LSTM_TRAINING_PARAMS["epochs"]:
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for sequences, labels in val_loader:
                    logits = model(sequences.to(device))
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_trues.extend(labels.numpy())
            metrics = compute_classification_metrics(np.array(val_trues), np.array(val_preds))
            LOGGER.info(
                "Epoch %d | loss=%.4f | val_accuracy=%.4f | val_f1=%.4f",
                epoch,
                epoch_loss,
                metrics["accuracy"],
                metrics["f1"],
            )

    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            logits = model(sequences.to(device))
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_trues.extend(labels.numpy())
    test_metrics = compute_classification_metrics(np.array(test_trues), np.array(test_preds))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    LOGGER.info("Saved LSTM checkpoint to %s", MODEL_CHECKPOINT_PATH)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(train_losses, LOSS_HISTORY_PATH)
    LOGGER.info("Persisted training loss history to %s", LOSS_HISTORY_PATH)

    return test_metrics


def main() -> None:
    """CLI entry-point."""
    metrics = train()
    LOGGER.info(
        "Final LSTM metrics | accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )


if __name__ == "__main__":
    main()
