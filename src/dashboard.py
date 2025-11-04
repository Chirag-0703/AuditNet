"""AuditNet Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import torch
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

try:
    from src.config import (
        DEFAULT_THRESHOLD,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DIR,
        RAW_LEDGER_PATH,
        get_logger,
    )
    from src.utils.metrics import compute_classification_metrics
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config import (  # type: ignore  # noqa: E402
        DEFAULT_THRESHOLD,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DIR,
        RAW_LEDGER_PATH,
        get_logger,
    )
    from src.utils.metrics import compute_classification_metrics  # type: ignore  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

LOGGER = get_logger(__name__)

st.set_page_config(layout="wide")
st.title("🕵️ AuditNet Dashboard")
st.markdown("A unified dashboard for GNN, LSTM, and Stacked MLP fraud detection models.")

# Sidebar navigation (kept minimal per requirements)
st.sidebar.title("🧭 Navigation")
FOCUS_MODEL_OPTIONS = ["GNN", "LSTM", "Stacked MLP"]
focus_model = st.sidebar.radio("Focus model", FOCUS_MODEL_OPTIONS, index=0)

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Normal", "Fraud"]
GRAPH_PATH = PROCESSED_DIR / "graph.pt"
SEQUENCE_PATH = PROCESSED_DIR / "sequences.pt"
STACK_PATH = PROCESSED_DIR / "stack_inputs.pt"
GNN_MODEL_PATH = MODELS_DIR / "gnn_checkpoint.pt"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_checkpoint.pt"
MLP_MODEL_PATH = MODELS_DIR / "mlp_checkpoint.pt"
LOSS_PATTERNS = (
    METRICS_DIR / "{model}_loss.pt",
    METRICS_DIR / "{model}_loss.npy",
    METRICS_DIR / "{model}_loss.json",
    MODELS_DIR / "{model}_loss.pt",
    MODELS_DIR / "{model}_loss.npy",
    MODELS_DIR / "{model}_history.json",
)


def _safe_torch_load(path: Path, **kwargs) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found at {path}.")
    return torch.load(path, map_location="cpu", **kwargs)


@st.cache_resource(show_spinner=False)
def load_gnn_artifacts(device_str: str) -> Dict[str, torch.Tensor]:
    """Load graph, model, and inference outputs for the GNN."""
    device = torch.device(device_str)
    from src.gnn.gnn_model import GNN  # pylint: disable=import-outside-toplevel

    graph = _safe_torch_load(GRAPH_PATH, weights_only=False)
    model = GNN(in_channels=graph.x.size(1), hidden_channels=64, out_channels=2).to(device)
    state_dict = _safe_torch_load(GNN_MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(graph.x.to(device), graph.edge_index.to(device))

    return {
        "graph": graph,
        "labels": graph.y.cpu(),
        "logits": logits.cpu(),
        "preds": torch.argmax(logits, dim=1).cpu(),
        "probs": torch.softmax(logits, dim=1)[:, 1].cpu(),
    }


@st.cache_resource(show_spinner=False)
def load_lstm_artifacts(device_str: str) -> Dict[str, torch.Tensor]:
    """Load sequences, model, and inference outputs for the LSTM."""
    device = torch.device(device_str)
    from src.lstm.lstm_model import LSTMClassifier  # pylint: disable=import-outside-toplevel

    seq_data = _safe_torch_load(SEQUENCE_PATH)
    sequences = seq_data["sequences"].to(device)
    labels = seq_data["labels"].to(device)

    model = LSTMClassifier(input_dim=sequences.size(2), hidden_dim=64).to(device)
    state_dict = _safe_torch_load(LSTM_MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(sequences)

    return {
        "labels": labels.cpu(),
        "logits": logits.cpu(),
        "preds": torch.argmax(logits, dim=1).cpu(),
        "probs": torch.softmax(logits, dim=1)[:, 1].cpu(),
    }


@st.cache_resource(show_spinner=False)
def load_mlp_artifacts(device_str: str) -> Dict[str, torch.Tensor]:
    """Load stacked inputs, model, and inference outputs for the MetaMLP."""
    device = torch.device(device_str)
    from src.mlp.mlp_model import MetaMLP  # pylint: disable=import-outside-toplevel

    stack_data = _safe_torch_load(STACK_PATH)
    features = stack_data["features"].to(device)
    labels = stack_data["labels"].to(device)

    model = MetaMLP(input_dim=features.size(1), hidden_dim=8, num_classes=2, dropout=0.3).to(device)
    state_dict = _safe_torch_load(MLP_MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(features)

    return {
        "labels": labels.cpu(),
        "logits": logits.cpu(),
        "preds": torch.argmax(logits, dim=1).cpu(),
        "probs": torch.softmax(logits, dim=1)[:, 1].cpu(),
    }


@st.cache_data(show_spinner=False)
def load_ledger() -> pd.DataFrame:
    """Load raw ledger data for contextual insights and filtering."""
    if not RAW_LEDGER_PATH.exists():
        raise FileNotFoundError(f"Ledger not found at {RAW_LEDGER_PATH}. Run simulate_data.py first.")
    df = pd.read_csv(RAW_LEDGER_PATH, parse_dates=["timestamp"])
    if "label" not in df.columns and "anomaly_type" in df.columns:
        df["label"] = (df["anomaly_type"] != "normal").astype(int)
    df.sort_values("timestamp", inplace=True)
    return df


def render_confusion_matrix(cm: torch.Tensor, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def metric_colorize(column: pd.Series) -> List[str]:
    """Return styles for metric dataframe highlighting."""
    styles: List[str] = []
    for value in column:
        if pd.isna(value):
            styles.append("")
        elif value >= 0.99:
            styles.append("background-color: #2ecc71; color: black;")
        elif value >= 0.95:
            styles.append("background-color: #f1c40f; color: black;")
        else:
            styles.append("background-color: #e74c3c; color: white;")
    return styles


def build_loss_figure(history: Sequence[float], title: str) -> go.Figure:
    """Build a loss curve with rolling-average overlay."""
    epochs = list(range(1, len(history) + 1))
    series = pd.Series(history, index=epochs, dtype=float)
    rolling = series.rolling(window=5, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=series.values,
            mode="lines+markers",
            name="Loss",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=rolling.values,
            mode="lines",
            name="Rolling Avg (w=5)",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_roc_pr_figure(labels: np.ndarray, probs: np.ndarray, model_name: str) -> go.Figure:
    """Render ROC and PR curves for the supplied probabilities."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    positive_rate = float(np.mean(labels))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"ROC Curve (AUC={roc_auc:.3f})",
            f"Precision-Recall Curve (AUC={pr_auc:.3f})",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(color="#888888", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Precision-Recall",
            line=dict(color="#ff7f0e"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[positive_rate, positive_rate],
            mode="lines",
            name="Baseline",
            line=dict(color="#888888", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        title=f"{model_name} ROC & PR Curves",
        hovermode="closest",
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )

    return fig


def summarize_training_stability(history: Sequence[float]) -> str:
    """Summarize loss trends for auditor-facing messaging."""
    if not history:
        return "Insufficient data to assess stability."

    arr = np.asarray(history, dtype=float)
    if arr.size < 2:
        return "Only one epoch recorded so far; trend is inconclusive."

    start = float(arr[0])
    end = float(arr[-1])
    drop_pct = ((start - end) / start * 100) if start else 0.0
    volatility = float(arr.std())
    mean_loss = float(arr.mean())

    if drop_pct > 50:
        trend_descriptor = "Strong improvement"
    elif drop_pct > 25:
        trend_descriptor = "Steady improvement"
    elif drop_pct > 5:
        trend_descriptor = "Modest improvement"
    elif drop_pct > -5:
        trend_descriptor = "Flat trajectory"
    else:
        trend_descriptor = "Loss is trending upward"

    if mean_loss:
        vol_ratio = volatility / mean_loss
    else:
        vol_ratio = 0.0

    if vol_ratio < 0.05:
        variance_descriptor = "very low volatility"
    elif vol_ratio < 0.15:
        variance_descriptor = "controlled volatility"
    else:
        variance_descriptor = "high variance to monitor"

    recent = arr[-5:] if arr.size >= 5 else arr
    recent_span = float(recent.max() - recent.min())
    recent_mean = float(recent.mean())
    baseline = max(recent_mean, 1e-6)
    if recent_span <= 0.02 * baseline:
        recent_descriptor = "has effectively plateaued in the last few epochs"
    elif recent[-1] < recent[0]:
        recent_descriptor = "continues to trend downward late in training"
    elif recent[-1] > recent[0]:
        recent_descriptor = "shows a late uptick — watch for overfitting"
    else:
        recent_descriptor = "shows mild oscillations near convergence"

    return (
        f"{trend_descriptor} (~{drop_pct:.1f}% from {start:.3f} → {end:.3f}) with {variance_descriptor}; "
        f"recent window {recent_descriptor}."
    )



def _loss_filename_key(model_key: str) -> str:
    """Create a filesystem-friendly key for loss file lookups."""
    return model_key.lower().replace(" ", "_")


def load_loss_history(model_key: str) -> Optional[Sequence[float]]:
    """Attempt to load a loss history for the given model key."""
    for pattern in LOSS_PATTERNS:
        candidate = Path(str(pattern).format(model=_loss_filename_key(model_key)))
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".pt":
                history = torch.load(candidate)
            elif candidate.suffix == ".npy":
                history = np.load(candidate)
            else:
                history = pd.read_json(candidate)
        except (ValueError, json.JSONDecodeError, OSError, RuntimeError):
            continue

        if isinstance(history, torch.Tensor):
            history = history.tolist()
        elif isinstance(history, np.ndarray):
            history = history.tolist()
        elif isinstance(history, pd.DataFrame):
            if "loss" in history.columns:
                history = history["loss"].tolist()
            else:
                history = history.iloc[:, 0].tolist()
        elif isinstance(history, dict):
            history = history.get("loss") or history.get("train_loss")

        if history:
            return list(history)
    return None




def get_node_mapping(ledger: pd.DataFrame) -> Tuple[List[str], Dict[int, str]]:
    """Return account ordering and index-to-account map."""
    accounts = sorted(set(ledger["from_acct"]) | set(ledger["to_acct"]))
    idx_to_account = {idx: acct for idx, acct in enumerate(accounts)}
    return accounts, idx_to_account


# --- Load model outputs once ---
gnn_data = load_gnn_artifacts(DEVICE_STR)
lstm_data = load_lstm_artifacts(DEVICE_STR)
mlp_data = load_mlp_artifacts(DEVICE_STR)

MODEL_RESULTS = {
    "GNN": gnn_data,
    "LSTM": lstm_data,
    "Stacked MLP": mlp_data,
}
ordered_models = [focus_model] + [name for name in MODEL_RESULTS.keys() if name != focus_model]

# Tabs for the two primary views
performance_tab, audit_tab = st.tabs(["Model Performance", "Audit Dashboard"])

METRIC_DESCRIPTIONS = {
    "Accuracy": "Overall rate of correct predictions across both fraud and normal classes.",
    "Precision": "Of the accounts flagged as fraud, how many were truly fraudulent (controls false positives).",
    "Recall": "Of the fraudulent accounts, how many did the model catch (controls false negatives).",
    "F1": "Harmonic mean of precision and recall, balancing false alarms with misses.",
}

ALL_METRICS = list(METRIC_DESCRIPTIONS.keys())
metric_selector_key = "metric_selector"
if metric_selector_key not in st.session_state:
    st.session_state[metric_selector_key] = ALL_METRICS

with performance_tab:
    st.markdown("### Metrics Overview")
    selected_metrics = st.multiselect(
        "Select metrics to display",
        options=ALL_METRICS,
        default=st.session_state[metric_selector_key],
        key=metric_selector_key,
    )

    if not selected_metrics:
        st.warning("Select at least one metric to compare models.")
    else:
        metric_rows = []
        for model_name in ordered_models:
            result = MODEL_RESULTS[model_name]
            metrics = compute_classification_metrics(
                result["labels"].numpy(),
                result["preds"].numpy(),
            )
            formatted_metrics = {key.title(): value for key, value in metrics.items()}
            metric_rows.append(
                {"Model": model_name, **{metric: formatted_metrics[metric] for metric in selected_metrics}}
            )
        metrics_df = pd.DataFrame(metric_rows).set_index("Model")
        subset = selected_metrics or list(metrics_df.columns)
        styled_metrics = metrics_df.style.format(lambda value: f"{value:.4f}").apply(
            metric_colorize, axis=0, subset=subset
        )
        st.dataframe(styled_metrics, width="stretch")

    st.markdown("#### Metric Interpretations")
    for metric_name in selected_metrics or ALL_METRICS:
        st.caption(f"**{metric_name}:** {METRIC_DESCRIPTIONS[metric_name]}")

    st.markdown("### Confusion Matrices")
    cm_cols = st.columns(3)
    for col, model_name in zip(cm_cols, ordered_models):
        with col:
            result = MODEL_RESULTS[model_name]
            cm = confusion_matrix(result["labels"].numpy(), result["preds"].numpy())
            fig = render_confusion_matrix(torch.tensor(cm), f"{model_name} Confusion Matrix")
            st.pyplot(fig)

    with st.expander("ROC & PR Curves"):
        for model_name in ordered_models:
            result = MODEL_RESULTS[model_name]
            labels = result["labels"].numpy()
            probs = result["probs"].numpy()
            if len(np.unique(labels)) < 2:
                st.warning(f"Insufficient class diversity to compute curves for {model_name}.")
                continue
            fig = build_roc_pr_figure(labels, probs, model_name)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Loss Curves")
    loss_cols = st.columns(3)
    loss_histories = {model_name: load_loss_history(model_name) for model_name in ordered_models}
    for col, model_name in zip(loss_cols, ordered_models):
        with col:
            history = loss_histories[model_name]
            if history:
                fig = build_loss_figure(history, f"{model_name} Training Loss")
                st.plotly_chart(fig, use_container_width=True)
            else:
                slug = _loss_filename_key(model_name)
                st.info(
                    f"No training loss history found for {model_name}. "
                    "Add an array at data/metrics/"
                    f"{slug}_loss.(pt|npy|json) or models/{slug}_history.json to display it here."
                )

    st.markdown("### Training Stability Notes")
    for model_name in ordered_models:
        history = loss_histories.get(model_name)
        if history:
            summary = summarize_training_stability(history)
        else:
            summary = "No training history available yet."
        st.markdown(f"- **{model_name}:** {summary}")

with audit_tab:
    ledger_df = load_ledger()
    if ledger_df.empty:
        st.warning("Ledger is empty — load data to explore the network.")
    else:
        _, idx_to_account = get_node_mapping(ledger_df)
        threshold = DEFAULT_THRESHOLD
        min_timestamp = ledger_df["timestamp"].min().to_pydatetime()
        max_timestamp = ledger_df["timestamp"].max().to_pydatetime()
        time_window = (min_timestamp, max_timestamp)

        gnn_probs = gnn_data["probs"].numpy()
        all_nodes = list(range(len(gnn_probs)))
        if not all_nodes:
            st.warning("No nodes available to visualize.")
        else:
            default_focus = st.session_state.get(
                "selected_node_focus",
                int(np.argmax(gnn_probs)),
            )
            if default_focus not in all_nodes:
                default_focus = all_nodes[0]
            focus_index = all_nodes.index(default_focus)

            if "selected_node_focus" not in st.session_state:
                st.session_state["selected_node_focus"] = default_focus

            selected_node = st.selectbox(
                "Focus node",
                options=all_nodes,
                index=focus_index,
                key="selected_node_focus",
            )

            st.markdown(
                """
                <style>
                .insight-grid {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.75rem;
                    margin: 0.5rem 0 1rem 0;
                }
                .insight-card {
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                    padding: 0.9rem 1.1rem;
                    min-width: 240px;
                    font-family: "Inter", "Segoe UI", sans-serif;
                    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
                }
                .insight-card strong {
                    display: block;
                    color: #334155;
                    font-size: 0.78rem;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    margin-bottom: 0.45rem;
                }
                .insight-card span {
                    display: block;
                    color: #0f172a;
                    font-size: 1.05rem;
                    line-height: 1.45rem;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Node Insights")
            st.caption("Details reflect the focused node and current filter settings.")
            selected_account = idx_to_account.get(selected_node, f"Account {selected_node}")
            prob = gnn_probs[selected_node].item()

            account_in = ledger_df[ledger_df["to_acct"] == selected_account]
            account_out = ledger_df[ledger_df["from_acct"] == selected_account]
            inbound_count = len(account_in)
            outbound_count = len(account_out)
            total_received = account_in["amount"].sum()
            total_sent = account_out["amount"].sum()

            st.markdown(
                f"""
                <div class="insight-grid">
                    <div class="insight-card">
                        <strong>Fraud Score</strong>
                        <span>Fraud probability: {prob:.2f}</span>
                    </div>
                    <div class="insight-card">
                        <strong>Transaction Counts</strong>
                        <span>Inbound transactions: {inbound_count:,} | Outbound transactions: {outbound_count:,}</span>
                    </div>
                    <div class="insight-card">
                        <strong>Transaction Totals</strong>
                        <span>Total received: ${total_received:,.2f} | Total sent: ${total_sent:,.2f}</span>
                    </div>
                    <div class="insight-card">
                        <strong>Account Identifier</strong>
                        <span>{selected_account}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Anomaly explanation"):
                explanation_parts = []
                if prob >= threshold:
                    explanation_parts.append("Probability exceeds the anomaly threshold.")
                if len(account_out) > len(account_in) * 2:
                    explanation_parts.append("Outbound activity dominates inbound traffic.")
                if not account_out.empty and account_out["amount"].max() > ledger_df["amount"].quantile(0.95):
                    explanation_parts.append("Large transfer detected relative to peers.")
                if not explanation_parts:
                    explanation_parts.append("No specific anomaly signals detected beyond base probability.")
                for item in explanation_parts:
                    st.write(f"- {item}")

            st.markdown("#### Recent Transactions")
            recent_activity = ledger_df[
                (ledger_df["from_acct"] == selected_account) | (ledger_df["to_acct"] == selected_account)
            ]
            recent_activity = recent_activity[
                (recent_activity["timestamp"] >= time_window[0]) & (recent_activity["timestamp"] <= time_window[1])
            ]
            st.dataframe(
                recent_activity.sort_values("timestamp", ascending=False).head(25),
                width="stretch",
            )

            label_series = ledger_df.get("label")
            if label_series is not None:
                fraud_transactions = ledger_df[label_series == 1]
            else:
                fraud_transactions = pd.DataFrame()
            if not fraud_transactions.empty:
                st.markdown("### Fraudulent Transaction Distribution")
                fig_hist = px.histogram(
                    fraud_transactions,
                    x="amount",
                    nbins=30,
                    title="Histogram of Fraudulent Transaction Amounts",
                    labels={"amount": "Transaction Amount"},
                )
                fig_hist.update_layout(
                    template="plotly_dark",
                    bargap=0.05,
                    margin=dict(l=40, r=20, t=60, b=40),
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No fraudulent transactions available to plot.")
