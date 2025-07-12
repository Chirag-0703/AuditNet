# AuditNet: ML-Driven Financial Audit Automation

AuditNet is a full-cycle machine learning pipeline designed to automate financial audits by detecting anomalies in simulated double-entry ledgers using graph neural networks (GNNs) and sequence models (LSTMs).


## Features

- Simulates realistic double-entry ledgers over a fiscal year
- Injects labeled anomalies (circular flows, structuring, self-loops, after-hours)
- Adds financial realism (month-end spikes, business hours, weekday bias)
- Prepares transaction graphs for GNNs and sequences for LSTMs
- Supports supervised anomaly detection
- Designed for modular development and visualization

---

## Project Structure

```
AuditNet/
├── data/
│   └── raw/              # Generated CSV ledger files
├── src/
│   ├── simulate_data.py  # Generates synthetic ledger with anomalies
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Chirag-0703/AuditNet.git
cd AuditNet
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### ▶️ Step 1: Generate synthetic ledger

This creates a full year of transactions with embedded anomalies:

```bash
python src/simulate_data.py
```





## 📦 Dependencies

AuditNet uses the following core libraries:

- `pandas`, `numpy` — for data manipulation
- `torch`, `torch-geometric`, `pytorch-lightning` — for GNN and LSTM models
- `networkx` — graph creation and visualization
- `matplotlib` — basic plotting
- `streamlit` — anomaly dashboard
- `tqdm` — progress bars

All packages are listed in `requirements.txt`.



