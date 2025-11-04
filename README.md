# AuditNet

## Project Summary

AuditNet is a production-ready fraud anomaly detection platform that synthesizes double-entry ledgers, injects labeled irregularities, and benchmarks graph, sequence, and tabular deep-learning models (GNN, LSTM, stacked MLP) for evaluator-ready insight. Synthetic transactions replicate seasonality, working-hour cadence, and risk typologies so auditors can validate models without client data exposure.

## Key Features

- **Multi-model fraud comparison** – benchmark GNN, LSTM, and stacked MLP scores side-by-side.
- **Ego-centric node investigation** – inspect high-risk accounts and near neighbors in isolation.
- **Anomaly probability scoring** – surface calibrated risk scores per account.
- **Recent transaction drill-down** – review counterparty flows, timestamps, and amounts.
- **ROC/PR curves and loss trends** – quantify discrimination power and training stability.
- **Threshold-based filtering** – tune risk tolerance by probability cutoffs.
- **Time-based transaction filtering** – constrain analyses to auditor-selected windows.
- **Interactive auditor workflows** – streamlit UX designed for risk review teams.

## Architecture Overview

- **Data simulation pipeline** (`src/simulate_data.py`): generates fiscal-year ledger, anomaly injection, feature engineering.
- **Graph construction** (`src/gnn/build_graph.py`): builds PyG `Data` objects with node/edge features.
- **Model training modules** (`src/gnn/train_gnn.py`, `src/lstm/train_lstm.py`, `src/mlp/train_mlp.py`): reproducible, weighted, stratified training.
- **Dashboard UI** (`src/dashboard.py`): streamlit app with performance analytics and investigative workflows.

```mermaid
graph TD
    A[Synthetic Ledger Simulation] --> B[Graph & Sequence Builders]
    B --> C[Model Training (GNN / LSTM / MLP)]
    C --> D[Inference Artifacts]
    D --> E[Streamlit Dashboard & Decisions]
```

## Directory Structure

```
AuditNet/
├── data/                  # raw and processed ledgers, graphs, sequences
├── src/
│   ├── simulate_data.py   # synthetic ledger generation
│   ├── gnn/               # GNN models, trainers, graph builders
│   ├── lstm/              # LSTM models, trainers, sequence builders
│   ├── mlp/               # Stacked MLP models and utilities
│   └── dashboard.py       # Streamlit UI entry point
├── requirements.txt
└── README.md
```

## Installation

1. **Python 3.9+** – ensure the interpreter is ≥ 3.9.
2. **Virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .\\.venv\\Scripts\\activate
   ```
3. **Dependencies**
   ```bash
   pip install --upgrade pip wheel
   pip install -r requirements.txt
   ```
4. **macOS runtime** – install BLAS/OpenMP backend for torch-geometric:
   ```bash
   brew install libomp
   ```
5. **PyTorch Geometric wheels** – if the default install fails, fetch the appropriate wheel from https://data.pyg.org/whl and install the build matching `torch==2.8.0` and your platform, e.g.
   ```bash
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
     --index-url https://download.pytorch.org/whl/cpu
   pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
     --find-links https://data.pyg.org/whl/torch-2.8.0+cpu.html
   ```
   (Swap `+cpu` for the appropriate CUDA wheel if you have GPU support.)

## Data & Model Pipeline

All CLI utilities live under `src/`. Run each step from the project root:

```bash
python src/simulate_data.py            # generate synthetic ledger
python src/lstm/build_sequences.py     # prepare LSTM-ready sequences
python src/gnn/build_graph.py          # construct PyG transaction graph
python src/gnn/train_gnn.py            # train GNN model & persist metrics
python src/lstm/train_lstm.py          # train LSTM model
python src/mlp/build_stack_inputs.py   # assemble stacked model inputs
python src/mlp/train_mlp.py            # train stacked MLP ensemble
```

Each script auto-inserts the repo root on `sys.path`, so `python src/...` works both in CLI and CI environments. All checkpoints and loss curves are saved under `models/` and `data/metrics/` for dashboard ingestion.

## Running the Dashboard

```bash
streamlit run src/dashboard.py
```

- **Model Performance tab** – compare metrics, confusion matrices, ROC/PR curves, loss curves, and training stability notes.
- **Audit Dashboard tab** – select an account, review probability scores, inbound/outbound summaries, anomaly rationale, recent transactions, and fraudulent-amount histograms.

## Metrics Interpretation Guide

- **Accuracy** – overall correctness; best for balanced datasets.
- **Precision** – proportion of predicted fraud that is real; prioritize when false positives are costly (e.g., manual reviews).
- **Recall** – proportion of actual fraud detected; emphasize when missing fraud has high liability.
- **F1 Score** – harmonic mean of precision and recall; use for imbalanced data to balance false positives/negatives.

## Limitations & Future Work

- Real ledger ingestion connectors (ERP/GL integrations).
- Concept drift detection and model re-calibration.
- Temporal GNN architectures for inter-day dependencies.
- Auto-encoder and reconstruction-based anomaly scoring.
- Model attribution tooling (SHAP, GNNExplainer) for auditor transparency.

## Recommended Auditor Workflow

1. Run the dashboard and open the **Model Performance** tab to identify the best-performing model for the current dataset.
2. Switch to **Audit Dashboard**, select the highest-risk account, and review probability and transaction summaries.
3. Inspect anomaly explanations and recent transactions to validate the flag.
4. Adjust probability thresholds and time window (if required) to stress-test findings.
5. Export insights or escalate to manual review workflows.

## Model Comparison Summary

- **GNN** – excels at capturing relational fraud patterns; strongest when graph density is high but requires consistent graph quality.
- **LSTM** – captures temporal behavior; effective for accounts with long sequential histories but sensitive to sequence length.
- **Stacked MLP** – fast, tabular-friendly ensembler; great baseline and ensemble component, but less explanatory without feature attribution.

## Badges

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Last Commit](https://img.shields.io/github/last-commit/Chirag-0703/AuditNet)

## License

MIT License – see `LICENSE` (placeholder).
