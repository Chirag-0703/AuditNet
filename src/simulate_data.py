"""Synthetic ledger simulation for AuditNet."""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Iterable, List

import numpy as np
import pandas as pd

try:
    from src.config import DEFAULT_RANDOM_SEED, RAW_LEDGER_PATH, get_logger
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path
    ROOT = _Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import DEFAULT_RANDOM_SEED, RAW_LEDGER_PATH, get_logger  # type: ignore  # noqa: E402

NUM_ACCOUNTS: int = 500
NUM_TXNS: int = 23_000
FRAUD_ACCOUNTS: int = 48
START_DATE: datetime = datetime(2024, 3, 31)
END_DATE: datetime = datetime(2025, 3, 31)

random.seed(DEFAULT_RANDOM_SEED)
np.random.seed(DEFAULT_RANDOM_SEED)

LOGGER = get_logger(__name__)


def generate_account_ids(count: int) -> List[str]:
    """Return formatted account identifiers."""
    return [f"A{index:04d}" for index in range(1, count + 1)]


def fiscal_timestamp(start: datetime, end: datetime, *, after_hours: bool = False) -> str:
    """Generate a timestamp that respects fiscal patterns."""
    delta = end - start
    while True:
        random_day = start + timedelta(days=random.randint(0, delta.days))
        if random.random() < 0.15 and random_day.month in {3, 6, 9, 12} and random_day.day >= 25:
            random_day = random_day.replace(day=random.randint(25, min(31, random_day.day)))

        if random.random() < 0.2 and random_day.day >= 28:
            random_day = random_day.replace(hour=random.randint(15, 23), minute=random.randint(0, 59))
        else:
            if random.random() < 0.9:
                while random_day.weekday() >= 5:
                    random_day = start + timedelta(days=random.randint(0, delta.days))

            hour = random.randint(0, 5) if after_hours else int(
                np.random.choice([*range(9, 18)] * 4 + [*range(6, 9)] + [*range(18, 21)])
            )
            minute = random.randint(0, 59)
            random_day = random_day.replace(hour=hour, minute=minute)

        return random_day.strftime("%Y-%m-%d %H:%M:%S")


def _simulate_normal_transactions(accounts: Iterable[str], limit: int, txn_id: int) -> List[List[str]]:
    """Generate baseline transactions between random accounts."""
    records: List[List[str]] = []
    for _ in range(limit):
        from_acct, to_acct = random.sample(list(accounts), 2)
        amount = round(random.uniform(100, 10_000), 2)
        timestamp = fiscal_timestamp(START_DATE, END_DATE)
        records.append([f"T{txn_id}", timestamp, from_acct, to_acct, amount, "normal"])
        txn_id += 1
    return records


def simulate_transactions() -> pd.DataFrame:
    """Simulate a synthetic ledger and persist to the raw data directory."""
    accounts = generate_account_ids(NUM_ACCOUNTS)
    fraud_accounts = random.sample(accounts, FRAUD_ACCOUNTS)
    txn_id = 1

    transactions: List[List[str]] = []
    normal_limit = int(NUM_TXNS * 0.85)
    transactions.extend(_simulate_normal_transactions(accounts, normal_limit, txn_id))
    txn_id += normal_limit

    for _ in range(500):
        path = random.sample(fraud_accounts, 3)
        amount = round(random.uniform(5_000, 10_000), 2)
        timestamp = fiscal_timestamp(START_DATE, END_DATE)
        for idx in range(3):
            transactions.append(
                [f"T{txn_id}", timestamp, path[idx], path[(idx + 1) % 3], amount - idx * 100, "circular_flow"]
            )
            txn_id += 1

    for _ in range(1500):
        account = random.choice(fraud_accounts)
        amount = round(random.uniform(1_000, 9_000), 2)
        timestamp = fiscal_timestamp(START_DATE, END_DATE)
        transactions.append([f"T{txn_id}", timestamp, account, account, amount, "self_loop"])
        txn_id += 1

    for _ in range(1500):
        sender = random.choice(fraud_accounts)
        receiver = random.choice([acct for acct in accounts if acct != sender])
        amount = round(random.uniform(1_000, 9_000), 2)
        timestamp = fiscal_timestamp(START_DATE, END_DATE, after_hours=True)
        transactions.append([f"T{txn_id}", timestamp, sender, receiver, amount, "after_hours"])
        txn_id += 1

    for _ in range(500):
        sender = random.choice(fraud_accounts)
        receivers = random.sample([acct for acct in accounts if acct != sender], 3)
        for receiver in receivers:
            amount = round(random.uniform(9_000, 9_999), 2)
            timestamp = fiscal_timestamp(START_DATE, END_DATE)
            transactions.append([f"T{txn_id}", timestamp, sender, receiver, amount, "structuring"])
            txn_id += 1

    ledger_df = pd.DataFrame(
        transactions,
        columns=["txn_id", "timestamp", "from_acct", "to_acct", "amount", "anomaly_type"],
    ).sort_values("timestamp").reset_index(drop=True)
    ledger_df["label"] = (ledger_df["anomaly_type"] != "normal").astype(int)

    os.makedirs(RAW_LEDGER_PATH.parent, exist_ok=True)
    ledger_df.to_csv(RAW_LEDGER_PATH, index=False)
    LOGGER.info(
        "Saved ledger to %s | rows=%d | fraud_accounts=%d",
        RAW_LEDGER_PATH,
        len(ledger_df),
        len(fraud_accounts),
    )
    return ledger_df


if __name__ == "__main__":
    simulate_transactions()
