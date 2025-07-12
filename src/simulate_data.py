import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set constants
NUM_ACCOUNTS = 500
NUM_TXNS = 10000
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
    start_date = datetime(2024, 3, 31)
    end_date = datetime(2025, 3, 31)

    txns = []
    txn_id = 1

    # Creating Normal transactions
    for _ in range(int(NUM_TXNS * 0.93)):
        from_acct, to_acct = random.sample(accounts, 2)
        amount = round(random.uniform(100, 10000), 2)
        ts = fiscal_timestamp(start_date, end_date)
        txns.append([f"T{txn_id}", ts, from_acct, to_acct, amount, "normal", "normal"])
        txn_id += 1

    # Targeting Circular flow Transactions
    for _ in range(int(NUM_TXNS * 0.02)):
        path = random.sample(accounts, 3)
        amt = round(random.uniform(5000, 10000), 2)
        ts = fiscal_timestamp(start_date, end_date)
        for i in range(3):
            from_acct = path[i]
            to_acct = path[(i + 1) % 3]
            txns.append([f"T{txn_id}", ts, from_acct, to_acct, amt - i * 100, "circular", "circular_flow"])
            txn_id += 1

    # Targeting Self-loop Transactions
    for _ in range(int(NUM_TXNS * 0.01)):
        acct = random.choice(accounts)
        amount = round(random.uniform(1000, 9000), 2)
        ts = fiscal_timestamp(start_date, end_date)
        txns.append([f"T{txn_id}", ts, acct, acct, amount, "self-loop", "self_loop"])
        txn_id += 1

    # Targeting After-hours Transactions
    for _ in range(int(NUM_TXNS * 0.02)):
        from_acct, to_acct = random.sample(accounts, 2)
        amount = round(random.uniform(1000, 9000), 2)
        ts = fiscal_timestamp(start_date, end_date, after_hours=True)
        txns.append([f"T{txn_id}", ts, from_acct, to_acct, amount, "after-hours", "after_hours"])
        txn_id += 1

    # Targeting Structuring Transactions
    for _ in range(int(NUM_TXNS * 0.02)):
        sender = random.choice(accounts)
        receivers = random.sample([a for a in accounts if a != sender], 3)
        for recv in receivers:
            amount = round(random.uniform(9000, 9999), 2)
            ts = fiscal_timestamp(start_date, end_date)
            txns.append([f"T{txn_id}", ts, sender, recv, amount, "structured", "structuring"])
            txn_id += 1

    df = pd.DataFrame(txns, columns=["txn_id", "timestamp", "from_acct", "to_acct", "amount", "description", "anomaly_type"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    os.makedirs("../data/raw", exist_ok=True)
    df.to_csv("../data/raw/ledger.csv", index=False)
    print("Saved: data/raw/ledger.csv | Rows:", len(df))

if __name__ == "__main__":
    simulate_transactions()