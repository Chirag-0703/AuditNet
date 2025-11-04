import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

NUM_ACCOUNTS = 500
NUM_TXNS = 23000
FRAUD_ACCOUNTS = 48
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def generate_account_ids(n):
    return [f"A{str(i).zfill(4)}" for i in range(1, n + 1)]

def fiscal_timestamp(start_date, end_date, after_hours=False):
    while True:
        delta = end_date - start_date
        random_day = start_date + timedelta(days=random.randint(0, delta.days))

        # Inject quarter-end bursts
        if random.random() < 0.15 and random_day.month in [3, 6, 9, 12]:
            if random_day.day >= 25:
                random_day = random_day.replace(day=random.randint(25, min(31, random_day.day)))

        # Inject end-of-month bursts
        if random.random() < 0.2 and random_day.day >= 28:
            random_day = random_day.replace(hour=random.randint(15, 23), minute=random.randint(0, 59))
        else:
            # Prefer weekdays
            if random.random() < 0.9:
                while random_day.weekday() >= 5:
                    random_day = start_date + timedelta(days=random.randint(0, delta.days))

            # Prefer business hours
            if after_hours:
                hour = random.randint(0, 5)
            else:
                hour = np.random.choice(
                    [*range(9, 18)] * 4 + [*range(6, 9)] + [*range(18, 21)]
                )
            minute = random.randint(0, 59)
            random_day = random_day.replace(hour=hour, minute=minute)

        return random_day.strftime("%Y-%m-%d %H:%M:%S")

def simulate_transactions():
    accounts = generate_account_ids(NUM_ACCOUNTS)
    fraud_accounts = random.sample(accounts, FRAUD_ACCOUNTS)

    start_date = datetime(2024, 3, 31)
    end_date = datetime(2025, 3, 31)

    txns = []
    txn_id = 1

    # ----- Normal transactions (85%) -----
    for _ in range(int(NUM_TXNS * 0.85)):
        from_acct, to_acct = random.sample(accounts, 2)
        amount = round(random.uniform(100, 10000), 2)
        ts = fiscal_timestamp(start_date, end_date)
        txns.append([f"T{txn_id}", ts, from_acct, to_acct, amount, "normal"])
        txn_id += 1

    # ---- Anomalies (~3750 per category = 15000 total = 15%) ----
    # Use only fraud_accounts for at least one side of each anomaly txn

    # Circular Flow (3 txns per cycle)
    for _ in range(500):  # 500 cycles = 1500 txns
        path = random.sample(fraud_accounts, 3)
        amt = round(random.uniform(5000, 10000), 2)
        ts = fiscal_timestamp(start_date, end_date)
        for i in range(3):
            txns.append([f"T{txn_id}", ts, path[i], path[(i + 1) % 3], amt - i * 100, "circular_flow"])
            txn_id += 1

    # Self-loop (fraud_account sends to itself)
    for _ in range(1500):
        acct = random.choice(fraud_accounts)
        amount = round(random.uniform(1000, 9000), 2)
        ts = fiscal_timestamp(start_date, end_date)
        txns.append([f"T{txn_id}", ts, acct, acct, amount, "self_loop"])
        txn_id += 1

    # After-hours (fraud_account as sender or receiver)
    for _ in range(1500):
        from_acct = random.choice(fraud_accounts)
        to_acct = random.choice([a for a in accounts if a != from_acct])
        amount = round(random.uniform(1000, 9000), 2)
        ts = fiscal_timestamp(start_date, end_date, after_hours=True)
        txns.append([f"T{txn_id}", ts, from_acct, to_acct, amount, "after_hours"])
        txn_id += 1

    # Structuring (1 sender, 3 receivers = 1500 txns from 500 senders)
    for _ in range(500):
        sender = random.choice(fraud_accounts)
        receivers = random.sample([a for a in accounts if a != sender], 3)
        for recv in receivers:
            amount = round(random.uniform(9000, 9999), 2)
            ts = fiscal_timestamp(start_date, end_date)
            txns.append([f"T{txn_id}", ts, sender, recv, amount, "structuring"])
            txn_id += 1

    df = pd.DataFrame(txns, columns=["txn_id", "timestamp", "from_acct", "to_acct", "amount", "anomaly_type"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/ledger.csv", index=False)
    print(f"Saved: data/raw/ledger.csv | Rows: {len(df)} | Fraud accounts: {len(fraud_accounts)}")

if __name__ == "__main__":
    simulate_transactions()
