"""Global configuration and logging utilities for AuditNet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_LEDGER_PATH: Final[Path] = DATA_DIR / "raw" / "ledger.csv"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
METRICS_DIR: Final[Path] = DATA_DIR / "metrics"

DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_THRESHOLD: Final[float] = 0.5
DEFAULT_TEST_SIZE: Final[float] = 0.2
DEFAULT_VAL_SPLIT: Final[float] = 0.5

GNN_TRAINING_PARAMS: Final[Dict[str, float]] = {
    "epochs": 100,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
}

LSTM_TRAINING_PARAMS: Final[Dict[str, int]] = {
    "epochs": 50,
    "batch_size": 32,
}

MLP_TRAINING_PARAMS: Final[Dict[str, float]] = {
    "epochs": 100,
    "learning_rate": 0.01,
}

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for the application.

    Parameters
    ----------
    level:
        Logging level to use for the root logger.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance.

    Parameters
    ----------
    name:
        Name for the logger.

    Returns
    -------
    logging.Logger
        Logger configured with global formatting defaults.
    """
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
