"""
Stages 5, 6, 7 (IBM AML): Tactical Intelligence Layers (Real GraphSAGE Model)
=============================================================================
Computes:
1. Item 7: Confidence Tiers from actual IBM GraphSAGE probabilities + structural signals.
2. Item 8: Rule-based explainability for IBM AML subgraphs.
3. Item 9: Tunable Alert Threshold Policy directly evaluated on the IBM GraphSAGE holdout test set.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ibm_graphsage_classifier import load_or_create_ibm_pyg_dataset, IBMGraphSAGE, normalize_node_features, TARGET_COL
from torch_geometric.loader import DataLoader

DATA_DIR = Path("data")
IBM_SUMMARY_FILE = DATA_DIR / "ibm_graph_summary.csv"
IBM_TIERS_FILE = DATA_DIR / "ibm_confidence_tiers.csv"
IBM_TIER_EVAL_FILE = DATA_DIR / "ibm_confidence_tier_evaluation.csv"
IBM_EXPLANATIONS_JSON = DATA_DIR / "ibm_explainability_examples.json"
IBM_POLICY_FILE = DATA_DIR / "ibm_threshold_policy_analysis.csv"

def wilson_ci(pos, n, conf=0.95):
    if n == 0: return (0.0, 0.0)
    z = 1.95996
    p = pos / n
    denom = 1 + (z**2)/n
    centre = p + (z**2)/(2*n)
    adj_std = np.sqrt((p*(1-p) + (z**2)/(4*n))/n)
    return (max(0.0, float((centre - z*adj_std)/denom)), min(1.0, float((centre + z*adj_std)/denom)))

def main():
    print("=" * 70)
    print("   STAGES 5, 6, 7 (IBM AML) ? TACTICAL INTELLIGENCE (GRAPHSAGE NATIVE)")
    print("=" * 70)
    
    raw_dataset, df_summary = load_or_create_ibm_pyg_dataset()
    
    train_ids, test_ids = train_test_split(
        df_summary["subgraph_id"].tolist(),
        test_size=0.20,
        random_state=42,
        stratify=df_summary[TARGET_COL]
    )
    test_set = set(test_ids)
    
    train_raw = [d for d in raw_dataset if d.subgraph_id not in test_set]
    test_raw = [d for d in raw_dataset if d.subgraph_id in test_set]
    train_norm, test_norm = normalize_node_features(train_raw, test_raw)
    
    train_loader = DataLoader(train_norm, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)
    
    # Train Seed 42 GraphSAGE
    torch.manual_seed(42)
    np.random.seed(42)
    n_pos = sum(int(d.y.item()) for d in train_norm)
    n_neg = len(train_norm) - n_pos
    pos_weight = float(n_neg / max(1, n_pos))
    
    model = IBMGraphSAGE(input_dim=7, hidden_dim=64, dropout=0.20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    for epoch in range(1, 31):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()
            
    # Inference on all 1,000 subgraphs
    full_loader = DataLoader(train_norm + test_norm, batch_size=64, shuffle=False)
    all_sub_objs = train_norm + test_norm
    model.eval()
    
    sub_id_to_prob = {}
    with torch.no_grad():
        for d in all_sub_objs:
            out, _ = model(d.x, d.edge_index, torch.zeros(d.num_nodes, dtype=torch.long))
            prob = float(torch.sigmoid(out).item())
            sub_id_to_prob[d.subgraph_id] = prob
            
    df_summary["graphsage_probability"] = df_summary["subgraph_id"].map(sub_id_to_prob)
    
    # -------------------------------------------------------------
    # 1. ITEM 7: IBM Confidence Tiers
    # -------------------------------------------------------------
    df_train_summary = df_summary[df_summary["subgraph_id"].isin(set(train_ids))]
    p75_vel = float(df_train_summary["velocity_tph"].quantile(0.75))
    p75_nodes = float(df_train_summary["num_nodes"].quantile(0.75))
    
    tier_records = []
    for _, row in df_summary.iterrows():
        p = row["graphsage_probability"]
        s_signals = 0
        if row["velocity_tph"] >= p75_vel: s_signals += 1
        if row["num_nodes"] >= p75_nodes: s_signals += 1
        if row["num_terminal_sinks"] > 0: s_signals += 1
        if row["fan_out_ratio"] >= 1.0: s_signals += 1
        
        if p >= 0.70 and s_signals >= 2:
            tier = "HIGH_CONFIDENCE"
        elif p >= 0.50:
            tier = "MEDIUM_CONFIDENCE"
        else:
            tier = "NORMAL"
            
        tier_records.append({
            "subgraph_id": row["subgraph_id"],
            "seed_account": row["seed_account"],
            "contains_laundering": int(row[TARGET_COL]),
            "graphsage_probability": round(p, 4),
            "confidence_tier": tier,
            "structural_signals_count": s_signals,
            "num_nodes": row["num_nodes"],
            "num_edges": row["num_edges"],
            "total_flow": row["total_transaction_value"],
            "num_terminal_sinks": row["num_terminal_sinks"]
        })
        
    df_tiers = pd.DataFrame(tier_records)
    df_tiers.to_csv(IBM_TIERS_FILE, index=False)
    
    # Evaluate Tiers
    total_laundering = int(df_tiers["contains_laundering"].sum())
    eval_records = []
    for t_name in ["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "NORMAL"]:
        sub_t = df_tiers[df_tiers["confidence_tier"] == t_name]
        cnt = len(sub_t)
        pos = int(sub_t["contains_laundering"].sum())
        prec = float(pos / cnt) if cnt > 0 else 0.0
        rec = float(pos / total_laundering) if total_laundering > 0 else 0.0
        ci = wilson_ci(pos, cnt)
        
        eval_records.append({
            "confidence_tier": t_name,
            "total_assigned": cnt,
            "true_laundering": pos,
            "true_normal": cnt - pos,
            "precision": round(prec * 100, 2),
            "precision_95ci": f"[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]",
            "laundering_capture_rate": round(rec * 100, 2),
            "pct_of_dataset": round(cnt / len(df_tiers) * 100, 2)
        })
    df_tier_eval = pd.DataFrame(eval_records)
    df_tier_eval.to_csv(IBM_TIER_EVAL_FILE, index=False)
    print("\n[1/3] IBM Confidence Tier Evaluation (N=1,000):")
    print(df_tier_eval.to_string(index=False))
    
    # -------------------------------------------------------------
    # 2. ITEM 8: IBM Explainability Layer
    # -------------------------------------------------------------
    exp_dict = {}
    for _, row in df_tiers.head(100).iterrows():
        sub_id = row["subgraph_id"]
        seed = row["seed_account"]
        bullets = [
            f"Evaluated multi-bank transaction subnetwork around root account {seed}.",
            f"Subnetwork connects {int(row['num_nodes'])} accounts across {int(row['num_edges'])} directed payment ledger transfers.",
            f"Cumulative transaction volume reached ${row['total_flow']:,.2f}.",
            f"Observed {int(row['num_terminal_sinks'])} terminal absorbing sink accounts (out_degree=0) within the observation window."
        ]
        if row["structural_signals_count"] >= 2 and row["graphsage_probability"] >= 0.70:
            bullets.append("Unusual concentration of multi-hop rapid fund distribution observed.")
            summary_txt = f"High-risk inter-bank laundering flow pattern detected around account {seed} across {int(row['num_nodes'])} accounts."
        elif row["graphsage_probability"] >= 0.50:
            bullets.append("Moderate structural branching detected, with partial sink accumulation.")
            summary_txt = f"Elevated transaction activity detected around account {seed}; partial multi-hop evidence."
        else:
            bullets.append("Transaction activity consistent with standard commercial or bilateral transfers.")
            summary_txt = f"Sub-threshold transaction pattern around account {seed}."
            
        exp_dict[sub_id] = {
            "subgraph_id": sub_id,
            "seed_account": seed,
            "risk_probability": row["graphsage_probability"],
            "confidence_tier": row["confidence_tier"],
            "executive_summary": summary_txt,
            "investigative_evidence_bullets": bullets,
            "top_terminal_details": {
                "terminal_type": "Bank Account Sink (out_degree=0)",
                "rationale": f"Flow reaches {int(row['num_terminal_sinks'])} terminal sink accounts with no subsequent observed outbound transfers."
            }
        }
    with open(IBM_EXPLANATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(exp_dict, f, indent=2)
    print(f"\n[2/3] Saved IBM explanations to: {IBM_EXPLANATIONS_JSON}")
    
    # -------------------------------------------------------------
    # 3. ITEM 9: IBM Threshold Policy (Test Set Only N=200)
    # -------------------------------------------------------------
    df_test_summary = df_summary[df_summary["subgraph_id"].isin(test_set)].copy()
    y_test_true = df_test_summary[TARGET_COL].values
    y_test_prob = df_test_summary["graphsage_probability"].values
    n_test_pos = int(sum(y_test_true))
    
    thresholds = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90]
    policy_rows = []
    for t in thresholds:
        y_pred = (y_test_prob >= t).astype(int)
        tp = int(np.sum((y_test_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_test_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_test_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_test_true == 0) & (y_pred == 0)))
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / n_test_pos) if n_test_pos > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        prec_ci = wilson_ci(tp, tp + fp)
        rec_ci = wilson_ci(tp, tp + fn)
        
        tier_label = "HIGH_SENSITIVITY" if t <= 0.2 else ("BALANCED_TRIAGE" if t <= 0.6 else "HIGH_PRECISION")
        if t >= 0.90: tier_label = "HIGH_CONFIDENCE_ALERT"
        
        policy_rows.append({
            "threshold": t,
            "classifier_model": "IBM GraphSAGE (Seed 42)",
            "policy_tier": tier_label,
            "alerts": tp + fp,
            "alert_rate_pct": round((tp + fp) / len(df_test_summary) * 100, 2),
            "precision": round(prec, 4),
            "precision_95ci": f"[{prec_ci[0]*100:.2f}%, {prec_ci[1]*100:.2f}%]",
            "recall": round(rec, 4),
            "recall_95ci": f"[{rec_ci[0]*100:.2f}%, {rec_ci[1]*100:.2f}%]",
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })
    df_policy = pd.DataFrame(policy_rows)
    df_policy.to_csv(IBM_POLICY_FILE, index=False)
    print("\n[3/3] Real GraphSAGE Holdout Test Policy Curve (N=200, Pos=59, Neg=141):")
    print(df_policy.to_string(index=False))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
