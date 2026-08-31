"""
Stage 3B (Dataset C): GraphSAGE Inductive Benchmark on Elliptic Bitcoin Dataset
==============================================================================
Validates the GraphSAGE GNN architecture on real-world transaction data (Elliptic Bitcoin DAG).

GUARDRAIL NOTICE (Guardrails #1, #2, #3, #4):
- Isolated node-classification benchmark on Bitcoin transactions.
- NEVER blended with synthetic or IBM datasets for composite claims.
- Evaluated with exact sample size N, class distribution, 95% Confidence Intervals,
  and multi-seed stability checks.

Dataset Structure (Elliptic):
- Graph: 203,769 transaction nodes, 234,355 directed payment edges.
- Features: 165 continuous graph/flow features (standardized).
- Train Split: Timesteps 1..34 (N = 29,894 labeled nodes: 3,462 illicit, 26,432 licit; 11.58% positive).
- Test Split:  Timesteps 35..49 (N = 16,670 labeled nodes: 1,083 illicit, 15,587 licit; 6.50% positive).
- Unlabeled: 157,205 nodes (y=2) excluded from supervised loss evaluation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from src.adapters.elliptic_adapter import EllipticAdapter

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUTPUT_CSV = DATA_DIR / "elliptic_graphsage_evaluation.csv"
OUTPUT_MULTI_SEED_CSV = DATA_DIR / "elliptic_multi_seed_evaluation.csv"


class EllipticGraphSAGE(nn.Module):
    def __init__(self, in_channels: int = 165, hidden_channels: int = 128, out_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.classifier = nn.Linear(out_channels, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.classifier(h).squeeze(-1)
        return logits, h


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = successes / total
    denom = 1 + (z**2) / total
    centre = p + (z**2) / (2 * total)
    adj_std = np.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)
    lower = (centre - z * adj_std) / denom
    upper = (centre + z * adj_std) / denom
    return (max(0.0, float(lower)), min(1.0, float(upper)))


def train_and_eval_elliptic(seed: int = 42, epochs: int = 60) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    adapter = EllipticAdapter()
    data, train_mask, test_mask = adapter.get_train_test_split(split_timestep=34)
    
    device = torch.device("cpu")
    model = EllipticGraphSAGE(in_channels=data.x.size(1), hidden_channels=128, out_channels=64).to(device)
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.binary_y.float().to(device)
    
    # Class imbalance weighting on train set
    n_train_pos = int(((y == 1) & train_mask).sum().item())
    n_train_neg = int(((y == 0) & train_mask).sum().item())
    pos_weight = torch.tensor([n_train_neg / max(1, n_train_pos)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out, _ = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    
    # Evaluation on test split (T > 34)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, edge_index)
        probs = torch.sigmoid(logits)
        
        y_test_true = y[test_mask].cpu().numpy().astype(int)
        y_test_prob = probs[test_mask].cpu().numpy()
        y_test_pred = (y_test_prob >= 0.50).astype(int)
    
    acc = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, zero_division=0)
    rec = recall_score(y_test_true, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_true, y_test_prob)
    pr_auc = average_precision_score(y_test_true, y_test_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()
    prec_ci = wilson_score_interval(tp, tp + fp)
    rec_ci = wilson_score_interval(tp, tp + fn)
    acc_ci = wilson_score_interval(tp + tn, len(y_test_true))
    
    return {
        "seed": seed,
        "n_train": int(train_mask.sum().item()),
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,
        "n_test": len(y_test_true),
        "n_test_pos": int((y_test_true == 1).sum()),
        "n_test_neg": int((y_test_true == 0).sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": acc, "accuracy_ci": acc_ci,
        "precision": prec, "precision_ci": prec_ci,
        "recall": rec, "recall_ci": rec_ci,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def main():
    print("=" * 65)
    print("   STAGE 3B (DATASET C) ? GRAPHSAGE ON ELLIPTIC BITCOIN GRAPH")
    print("=" * 65)
    
    print("Evaluating GraphSAGE inductive node classification on Elliptic DAG...")
    m_main = train_and_eval_elliptic(seed=42, epochs=50)
    
    print("\nRunning multi-seed evaluation across 5 seeds...")
    seeds = [42, 101, 2024, 7, 99]
    multi_seed_results = []
    for s in seeds:
        res = train_and_eval_elliptic(seed=s, epochs=50)
        multi_seed_results.append({
            "seed": s,
            "n_test": res["n_test"],
            "positives": res["n_test_pos"],
            "negatives": res["n_test_neg"],
            "accuracy": round(res["accuracy"], 4),
            "precision": round(res["precision"], 4),
            "recall": round(res["recall"], 4),
            "f1": round(res["f1"], 4),
            "roc_auc": round(res["roc_auc"], 4),
            "pr_auc": round(res["pr_auc"], 4)
        })
    
    df_multi = pd.DataFrame(multi_seed_results)
    OUTPUT_MULTI_SEED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_multi.to_csv(OUTPUT_MULTI_SEED_CSV, index=False)
    
    df_main = pd.DataFrame([{
        "dataset": "Elliptic Bitcoin Transaction DAG",
        "task": "Inductive Node Classification (Illicit vs Licit)",
        "n_train_nodes": m_main["n_train"],
        "n_train_illicit": m_main["n_train_pos"],
        "n_test_nodes": m_main["n_test"],
        "n_test_illicit": m_main["n_test_pos"],
        "accuracy": round(m_main["accuracy"], 4),
        "precision": round(m_main["precision"], 4),
        "recall": round(m_main["recall"], 4),
        "f1": round(m_main["f1"], 4),
        "roc_auc": round(m_main["roc_auc"], 4),
        "pr_auc": round(m_main["pr_auc"], 4),
        "tp": m_main["tp"], "fp": m_main["fp"],
        "tn": m_main["tn"], "fn": m_main["fn"]
    }])
    df_main.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "=" * 65)
    print("       STAGE 3B ? ELLIPTIC BENCHMARK PERFORMANCE REPORT")
    print("=" * 65)
    print(f"Dataset                 : Dataset C (Elliptic Bitcoin Transaction Graph)")
    print(f"Task                    : Inductive Node Classification (Illicit vs Licit)")
    print(f"Train Partition (T<=34) : {m_main['n_train']:,} nodes (Illicit: {m_main['n_train_pos']:,} | 11.58%)")
    print(f"Test Partition  (T>34)  : {m_main['n_test']:,} nodes (Illicit: {m_main['n_test_pos']:,} | 6.50%)")
    print(f"Decision Threshold      : 0.50")
    print("-" * 65)
    print(f"Accuracy                : {m_main['accuracy']*100:.2f}%  [95% CI: {m_main['accuracy_ci'][0]*100:.2f}% - {m_main['accuracy_ci'][1]*100:.2f}%]")
    print(f"Precision               : {m_main['precision']*100:.2f}%  [95% CI: {m_main['precision_ci'][0]*100:.2f}% - {m_main['precision_ci'][1]*100:.2f}%]  ({m_main['tp']}/{m_main['tp']+m_main['fp']})")
    print(f"Recall                  : {m_main['recall']*100:.2f}%  [95% CI: {m_main['recall_ci'][0]*100:.2f}% - {m_main['recall_ci'][1]*100:.2f}%]  ({m_main['tp']}/{m_main['tp']+m_main['fn']})")
    print(f"F1 Score                : {m_main['f1']*100:.2f}%")
    print(f"ROC-AUC                 : {m_main['roc_auc']:.4f}")
    print(f"PR-AUC                  : {m_main['pr_auc']:.4f}")
    print("-" * 65)
    print(f"Confusion Matrix        : TP={m_main['tp']}, FP={m_main['fp']}, TN={m_main['tn']}, FN={m_main['fn']}")
    print("-" * 65)
    print(f"5-Seed Stability Summary (Mean +/- Std, N=5 seeds):")
    print(f"  F1 Score              : {df_multi['f1'].mean()*100:.2f}% +/- {df_multi['f1'].std()*100:.2f}%")
    print(f"  Precision             : {df_multi['precision'].mean()*100:.2f}% +/- {df_multi['precision'].std()*100:.2f}%")
    print(f"  Recall                : {df_multi['recall'].mean()*100:.2f}% +/- {df_multi['recall'].std()*100:.2f}%")
    print(f"  PR-AUC                : {df_multi['pr_auc'].mean():.4f} +/- {df_multi['pr_auc'].std():.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
