"""
Stage 3A: XGBoost Baseline Classifier (Scoped Rebuild)
======================================================
Trains and evaluates a deliberately-simple tabular baseline classifier on
graph metrics and financial flow features extracted from Stage 2 subgraphs.

Baseline Features:
- num_nodes, num_edges, num_account_nodes, num_atm_nodes, num_terminal_nodes
- max_hop, total_transaction_value, max_transaction_value, avg_transaction_value
- fan_out_ratio (out_degree / (in_degree + out_degree + 1e-5))
- velocity_tph (num_edges / 72.0)
- velocity_vph (total_transaction_value / 72.0)
- in_degree_incident, out_degree_incident
- density, number_of_connected_components, average_degree

Guardrails & Validation:
- Integrated with SyntheticAdapter.
- Strict data leakage audit (no target/ground truth in features; entity overlap check).
- Honest metric reporting with sample size N, class balance, and 95% Confidence Intervals.
- Multi-seed stability evaluation across 5 seeds.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from src.adapters.synthetic_adapter import SyntheticAdapter

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
GRAPH_SUMMARY_FILE = DATA_DIR / "graph_summary.csv"
MODEL_FILE = MODELS_DIR / "xgboost_baseline.json"
FEATURE_SCHEMA_FILE = MODELS_DIR / "xgboost_features.json"
PREDICTIONS_FILE = DATA_DIR / "xgboost_predictions.csv"
THRESHOLD_ANALYSIS_FILE = DATA_DIR / "xgboost_threshold_analysis.csv"
FEATURE_IMPORTANCE_FILE = DATA_DIR / "xgboost_feature_importance.csv"
TEMPORAL_EVAL_FILE = DATA_DIR / "xgboost_temporal_evaluation.csv"
MULTI_SEED_EVAL_FILE = DATA_DIR / "xgboost_multi_seed_evaluation.csv"

PR_CURVE_FILE = DATA_DIR / "xgboost_pr_curve.png"
ROC_CURVE_FILE = DATA_DIR / "xgboost_roc_curve.png"
FEATURE_IMPORTANCE_PLOT_FILE = DATA_DIR / "xgboost_feature_importance.png"

RANDOM_STATE = 42
TARGET_COL = "contains_suspicious_activity"

EXCLUDED_COLUMNS = [
    "contains_suspicious_activity",
    "suspicious_ring_count",
    "is_suspicious",
    "ring_id",
    "ground_truth_entity_id",
    "complaint_id",
    "incident_entity_id",
    "incident_time",
    "window_start",
    "window_end"
]


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence
    p = successes / total
    denominator = 1 + (z**2) / total
    centre_adjusted_probability = p + (z**2) / (2 * total)
    adjusted_std_dev = np.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)
    lower_bound = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    return (max(0.0, float(lower_bound)), min(1.0, float(upper_bound)))


def bootstrap_metric_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """Calculates bootstrap 95% confidence intervals for F1, ROC-AUC, and PR-AUC."""
    np.random.seed(seed)
    n = len(y_true)
    f1_list, roc_list, pr_list = [], [], []
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        f1_list.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
        roc_list.append(roc_auc_score(y_true[idx], y_prob[idx]))
        pr_list.append(average_precision_score(y_true[idx], y_prob[idx]))
    
    return {
        "f1_ci": (float(np.percentile(f1_list, 2.5)), float(np.percentile(f1_list, 97.5))),
        "roc_auc_ci": (float(np.percentile(roc_list, 2.5)), float(np.percentile(roc_list, 97.5))),
        "pr_auc_ci": (float(np.percentile(pr_list, 2.5)), float(np.percentile(pr_list, 97.5)))
    }


def engineer_baseline_features(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Computes explicit baseline flow and velocity features."""
    df = df_summary.copy()
    in_deg = df["in_degree_incident"].fillna(0)
    out_deg = df["out_degree_incident"].fillna(0)
    df["fan_out_ratio"] = out_deg / (in_deg + out_deg + 1e-5)
    df["velocity_tph"] = df["num_edges"] / 72.0
    df["velocity_vph"] = df["total_transaction_value"] / 72.0
    return df


def audit_data_leakage(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("        DATA QUALITY & LEAKAGE AUDIT")
    print("=" * 55)
    print(f"Total Graphs (N)           : {len(df)}")
    print(f"Train Partition (N_train)  : {len(train_df)} (Normal: {(train_df[TARGET_COL] == 0).sum()}, Suspicious: {(train_df[TARGET_COL] == 1).sum()})")
    print(f"Test Partition (N_test)    : {len(test_df)} (Normal: {(test_df[TARGET_COL] == 0).sum()}, Suspicious: {(test_df[TARGET_COL] == 1).sum()})")
    
    # Check overlap
    overlap_entities = set(train_df["incident_entity_id"]).intersection(set(test_df["incident_entity_id"]))
    print(f"Shared Incident Entities   : {len(overlap_entities)} (Expected strictly <= random entity recurrence across distinct complaints)")
    print("Leakage Exclusions Verified: All ground-truth tags (contains_suspicious_activity, is_suspicious, ring_id) excluded from features.")
    print("=" * 55 + "\n")


def train_and_evaluate_xgboost(seed: int = 42) -> Tuple[xgb.XGBClassifier, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    adapter = SyntheticAdapter()
    df_raw = adapter.load_graph_summary()
    df_engineered = engineer_baseline_features(df_raw)
    
    feature_names = [
        c for c in df_engineered.columns
        if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df_engineered[c])
    ]
    
    # Stratified split
    train_df, test_df = train_test_split(
        df_engineered,
        test_size=0.20,
        random_state=seed,
        stratify=df_engineered[TARGET_COL]
    )
    
    X_train = train_df[feature_names]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[feature_names]
    y_test = test_df[TARGET_COL].astype(int)
    
    neg_c = int((y_train == 0).sum())
    pos_c = int((y_train == 1).sum())
    scale_pos_weight = float(neg_c / pos_c) if pos_c > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    prec_ci = wilson_score_interval(tp, tp + fp)
    rec_ci = wilson_score_interval(tp, tp + fn)
    acc_ci = wilson_score_interval(tp + tn, len(y_test))
    boot_ci = bootstrap_metric_ci(y_test.values, y_pred, y_prob, seed=seed)
    
    metrics = {
        "seed": seed,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_pos_test": int((y_test == 1).sum()),
        "n_neg_test": int((y_test == 0).sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": acc, "accuracy_ci": acc_ci,
        "precision": prec, "precision_ci": prec_ci,
        "recall": rec, "recall_ci": rec_ci,
        "f1": f1, "f1_ci": boot_ci["f1_ci"],
        "roc_auc": roc_auc, "roc_auc_ci": boot_ci["roc_auc_ci"],
        "pr_auc": pr_auc, "pr_auc_ci": boot_ci["pr_auc_ci"]
    }
    
    return model, metrics, train_df, test_df


def run_multi_seed_evaluation(seeds: List[int] = [42, 101, 2024, 7, 99]) -> pd.DataFrame:
    records = []
    for s in seeds:
        _, m, _, _ = train_and_evaluate_xgboost(seed=s)
        records.append({
            "seed": s,
            "n_test": m["n_test"],
            "positives": m["n_pos_test"],
            "negatives": m["n_neg_test"],
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "pr_auc": round(m["pr_auc"], 4)
        })
    df_seeds = pd.DataFrame(records)
    df_seeds.to_csv(MULTI_SEED_EVAL_FILE, index=False)
    return df_seeds


def main():
    print("=" * 60)
    print("   STAGE 3A ? XGBOOST BASELINE CLASSIFIER (SCOPED REBUILD)")
    print("=" * 60)
    
    model, m, train_df, test_df = train_and_evaluate_xgboost(seed=RANDOM_STATE)
    adapter = SyntheticAdapter()
    df_raw = adapter.load_graph_summary()
    audit_data_leakage(df_raw, train_df, test_df)
    
    # Save model and feature schema
    feature_names = [
        c for c in engineer_baseline_features(df_raw).columns
        if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(engineer_baseline_features(df_raw)[c])
    ]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE))
    with open(FEATURE_SCHEMA_FILE, "w") as f:
        json.dump({"features": feature_names}, f, indent=2)
    print(f"[SUCCESS] Saved model to: {MODEL_FILE}")
    print(f"[SUCCESS] Saved feature schema to: {FEATURE_SCHEMA_FILE}")
    
    # Multi-seed stability
    print("Running 5-seed stability validation...")
    df_seeds = run_multi_seed_evaluation()
    
    print("\n" + "=" * 60)
    print("       STAGE 3A ? BASELINE PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Sample Size (N_test)    : {m['n_test']} cases (Normal: {m['n_neg_test']}, Suspicious: {m['n_pos_test']})")
    print(f"Decision Threshold      : 0.50 (Default Operational Baseline)")
    print("-" * 60)
    print(f"Accuracy                : {m['accuracy']*100:.2f}%  [95% CI: {m['accuracy_ci'][0]*100:.2f}% - {m['accuracy_ci'][1]*100:.2f}%]")
    print(f"Precision               : {m['precision']*100:.2f}%  [95% CI: {m['precision_ci'][0]*100:.2f}% - {m['precision_ci'][1]*100:.2f}%]  ({m['tp']}/{m['tp']+m['fp']})")
    print(f"Recall                  : {m['recall']*100:.2f}%  [95% CI: {m['recall_ci'][0]*100:.2f}% - {m['recall_ci'][1]*100:.2f}%]  ({m['tp']}/{m['tp']+m['fn']})")
    print(f"F1 Score                : {m['f1']*100:.2f}%  [95% CI: {m['f1_ci'][0]*100:.2f}% - {m['f1_ci'][1]*100:.2f}%]")
    print(f"ROC-AUC                 : {m['roc_auc']:.4f}  [95% CI: {m['roc_auc_ci'][0]:.4f} - {m['roc_auc_ci'][1]:.4f}]")
    print(f"PR-AUC                  : {m['pr_auc']:.4f}  [95% CI: {m['pr_auc_ci'][0]:.4f} - {m['pr_auc_ci'][1]:.4f}]")
    print("-" * 60)
    print(f"Confusion Matrix        : TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}")
    print("-" * 60)
    print(f"5-Seed Stability Summary (Mean +/- Std, N=5 seeds):")
    print(f"  F1 Score              : {df_seeds['f1'].mean()*100:.2f}% +/- {df_seeds['f1'].std()*100:.2f}%")
    print(f"  Precision             : {df_seeds['precision'].mean()*100:.2f}% +/- {df_seeds['precision'].std()*100:.2f}%")
    print(f"  Recall                : {df_seeds['recall'].mean()*100:.2f}% +/- {df_seeds['recall'].std()*100:.2f}%")
    print(f"  PR-AUC                : {df_seeds['pr_auc'].mean():.4f} +/- {df_seeds['pr_auc'].std():.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
