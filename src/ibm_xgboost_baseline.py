"""
Stage 3A (IBM AML): Scoped Baseline Classifier (XGBoost)
=========================================================
Trains and evaluates XGBoost on IBM AML subgraphs with strict leakage audit,
95% Confidence Intervals, and 5-seed stability testing.

Schema & Guardrail Compliance:
- Target: contains_laundering (1 = Subgraph contains illicit transfer, 0 = Benign).
- Features: num_nodes, num_edges, in_degree_seed, out_degree_seed, fan_out_ratio,
  density, average_degree, num_terminal_sinks, total_transaction_value,
  max_transaction_value, velocity_tph, velocity_vph.
- Explicitly dropped: account_age_days, dormancy_score, num_cash_out_edges (unavailable).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = Path("data")
SUMMARY_FILE = DATA_DIR / "ibm_graph_summary.csv"
EVAL_FILE = DATA_DIR / "ibm_xgboost_evaluation.csv"
MULTI_SEED_FILE = DATA_DIR / "ibm_xgboost_multi_seed_evaluation.csv"

FEATURE_COLS = [
    "num_nodes",
    "num_edges",
    "in_degree_seed",
    "out_degree_seed",
    "fan_out_ratio",
    "density",
    "average_degree",
    "num_terminal_sinks",
    "total_transaction_value",
    "max_transaction_value",
    "velocity_tph",
    "velocity_vph"
]
TARGET_COL = "contains_laundering"

def wilson_ci(pos, n, conf=0.95):
    if n == 0: return (0.0, 0.0)
    z = 1.95996
    p = pos / n
    denom = 1 + (z**2)/n
    centre = p + (z**2)/(2*n)
    adj_std = np.sqrt((p*(1-p) + (z**2)/(4*n))/n)
    return (max(0.0, float((centre - z*adj_std)/denom)), min(1.0, float((centre + z*adj_std)/denom)))

def bootstrap_ci(y_true, y_pred, y_prob, metric_name, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        if metric_name == "f1":
            scores.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
        elif metric_name == "roc_auc":
            scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        elif metric_name == "pr_auc":
            scores.append(average_precision_score(y_true[idx], y_prob[idx]))
    if not scores:
        return (0.0, 0.0)
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))

def train_and_eval_ibm_xgboost(seed=42):
    df = pd.read_csv(SUMMARY_FILE)
    
    # 80/20 train/test split stratified
    df_train, df_test = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df[TARGET_COL]
    )
    
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL].values
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL].values
    
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = float(n_neg / max(1, n_pos))
    
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    
    n_test = len(y_test)
    tp = int(np.sum((y_test == 1) & (y_pred == 1)))
    fp = int(np.sum((y_test == 0) & (y_pred == 1)))
    fn = int(np.sum((y_test == 1) & (y_pred == 0)))
    tn = int(np.sum((y_test == 0) & (y_pred == 0)))
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    acc_ci = wilson_ci(tp + tn, n_test)
    prec_ci = wilson_ci(tp, tp + fp)
    rec_ci = wilson_ci(tp, tp + fn)
    f1_ci = bootstrap_ci(y_test, y_pred, y_prob, "f1", seed=seed)
    roc_ci = bootstrap_ci(y_test, y_pred, y_prob, "roc_auc", seed=seed)
    pr_ci = bootstrap_ci(y_test, y_pred, y_prob, "pr_auc", seed=seed)
    
    return {
        "seed": seed,
        "n_test": n_test,
        "n_pos": int(sum(y_test)),
        "n_neg": int(len(y_test) - sum(y_test)),
        "accuracy": acc,
        "acc_ci_lower": acc_ci[0],
        "acc_ci_upper": acc_ci[1],
        "precision": prec,
        "prec_ci_lower": prec_ci[0],
        "prec_ci_upper": prec_ci[1],
        "recall": rec,
        "rec_ci_lower": rec_ci[0],
        "rec_ci_upper": rec_ci[1],
        "f1": f1,
        "f1_ci_lower": f1_ci[0],
        "f1_ci_upper": f1_ci[1],
        "roc_auc": roc_auc,
        "roc_ci_lower": roc_ci[0],
        "roc_ci_upper": roc_ci[1],
        "pr_auc": pr_auc,
        "pr_ci_lower": pr_ci[0],
        "pr_ci_upper": pr_ci[1],
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }, clf

def main():
    print("=" * 70)
    print("   STAGE 3A (IBM AML) ? XGBOOST BASELINE CLASSIFIER")
    print("=" * 70)
    
    # 1. Single Seed 42 run
    metrics_42, clf = train_and_eval_ibm_xgboost(seed=42)
    df_eval = pd.DataFrame([metrics_42])
    df_eval.to_csv(EVAL_FILE, index=False)
    
    print(f"Sample Size (N_test) : {metrics_42['n_test']} (Pos: {metrics_42['n_pos']} | 20%, Neg: {metrics_42['n_neg']} | 80%)")
    print(f"Accuracy             : {metrics_42['accuracy']*100:.2f}% [95% CI: {metrics_42['acc_ci_lower']*100:.2f}% - {metrics_42['acc_ci_upper']*100:.2f}%]")
    print(f"Precision            : {metrics_42['precision']*100:.2f}% [95% CI: {metrics_42['prec_ci_lower']*100:.2f}% - {metrics_42['prec_ci_upper']*100:.2f}%] ({metrics_42['tp']}/{metrics_42['tp']+metrics_42['fp']})")
    print(f"Recall               : {metrics_42['recall']*100:.2f}% [95% CI: {metrics_42['rec_ci_lower']*100:.2f}% - {metrics_42['rec_ci_upper']*100:.2f}%] ({metrics_42['tp']}/{metrics_42['tp']+metrics_42['fn']})")
    print(f"F1 Score             : {metrics_42['f1']*100:.2f}% [95% CI: {metrics_42['f1_ci_lower']*100:.2f}% - {metrics_42['f1_ci_upper']*100:.2f}%]")
    print(f"ROC-AUC              : {metrics_42['roc_auc']:.4f} [95% CI: {metrics_42['roc_ci_lower']:.4f} - {metrics_42['roc_ci_upper']:.4f}]")
    print(f"PR-AUC               : {metrics_42['pr_auc']:.4f} [95% CI: {metrics_42['pr_ci_lower']:.4f} - {metrics_42['pr_ci_upper']:.4f}]")
    print(f"Confusion Matrix     : TP={metrics_42['tp']}, FP={metrics_42['fp']}, TN={metrics_42['tn']}, FN={metrics_42['fn']}")
    
    # 2. 5-Seed Stability Run
    seeds = [42, 101, 2024, 7, 99]
    multi_seed_records = []
    for s in seeds:
        m, _ = train_and_eval_ibm_xgboost(seed=s)
        multi_seed_records.append(m)
        
    df_multi = pd.DataFrame(multi_seed_records)
    df_multi.to_csv(MULTI_SEED_FILE, index=False)
    
    print("\n" + "-" * 70)
    print("5-Seed Stability Summary (N=5 independent splits):")
    print(f"  ? F1 Score : {df_multi['f1'].mean()*100:.2f}% +/- {df_multi['f1'].std()*100:.2f}%")
    print(f"  ? Precision: {df_multi['precision'].mean()*100:.2f}% +/- {df_multi['precision'].std()*100:.2f}%")
    print(f"  ? Recall   : {df_multi['recall'].mean()*100:.2f}% +/- {df_multi['recall'].std()*100:.2f}%")
    print(f"  ? PR-AUC   : {df_multi['pr_auc'].mean():.4f} +/- {df_multi['pr_auc'].std():.4f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
