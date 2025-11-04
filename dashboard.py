# dashboard.py

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import pyvis
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# Set wide layout and title
st.set_page_config(layout="wide")
st.title("🕵️ AuditNet Dashboard")
st.markdown("A unified dashboard for GNN, LSTM, and Stacked MLP fraud detection models.")

# ----- Sidebar -----
st.sidebar.title("🧠 Select Model")
model_choice = st.sidebar.radio("Choose a model", ["GNN", "LSTM", "Stacked MLP"])

# ----- Load Data & Models -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_choice == "GNN":
    from src.gnn.gnn_model import GNN
    graph_data = torch.load("data/processed/graph.pt", weights_only=False)
    x, edge_index, y = graph_data.x.to(device), graph_data.edge_index.to(device), graph_data.y.to(device)

    model = GNN(in_channels=x.size(1), hidden_channels=64, out_channels=2).to(device)
    model.load_state_dict(torch.load("models/gnn_checkpoint.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(x, edge_index)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]
    labels = y

elif model_choice == "LSTM":
    from src.lstm.lstm_model import LSTMClassifier
    seq_data = torch.load("data/processed/sequences.pt")
    sequences = seq_data['sequences'].to(device)
    labels = seq_data['labels'].to(device)
    account_ids = seq_data['account_ids']

    model = LSTMClassifier(input_dim=sequences.size(2), hidden_dim=64).to(device)
    model.load_state_dict(torch.load("models/lstm_checkpoint.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(sequences)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

elif model_choice == "Stacked MLP":
    from src.mlp.mlp_model import MetaMLP 
    data = torch.load("data/processed/stack_inputs.pt")
    X, labels = data['features'].to(device), data['labels'].to(device)

    input_dim = X.shape[1]
    model = MetaMLP(input_dim=input_dim, hidden_dim=8, num_classes=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load("models/mlp_checkpoint.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]


# ----- Metrics -----
st.subheader("📈 Model Metrics")

acc = accuracy_score(labels.cpu(), preds.cpu())
f1 = f1_score(labels.cpu(), preds.cpu())
cm = confusion_matrix(labels.cpu(), preds.cpu())

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("F1 Score", f"{f1:.4f}")

st.markdown("#### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# ----- GNN Graph Visualization -----

if model_choice == "GNN":
    st.subheader("🕸️ GNN Interactive Transaction Graph")

    edge_list = edge_index.cpu().numpy().T
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black", directed=False)

    # Add nodes with proper sizing and labeling
    for i in range(x.shape[0]):
        fraud_prob = probs[i].item()
        label = f"Account {i}<br>Fraud Probability: {fraud_prob:.2f}"
        color = "red" if fraud_prob > 0.5 else "green"
        size = 15 + (fraud_prob * 30)  # Larger if more suspicious
        net.add_node(i, label=str(i), title=label, color=color, size=size)

    # Add edges
    for src, dst in edge_list:
        net.add_edge(int(src), int(dst), color="#888888")

    # Set layout with physics enabled (but stable)
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "timestep": 0.5,
        "stabilization": {
          "iterations": 150
        }
      },
      "nodes": {
        "shape": "dot",
        "font": { "size": 14, "color": "black" }
      },
      "edges": {
        "color": { "color": "#888888" }
      }
    }
    """)

    # Save and render without writing permanently
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False) as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_file.seek(0)
        components.html(tmp_file.read(), height=700, scrolling=True)
