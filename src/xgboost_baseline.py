"""
Stage 3A: XGBoost Baseline Classifier
====================================
This module trains and evaluates an XGBoost baseline classifier on tabular graph
and financial features extracted from Stage 2 (data/graph_summary.csv).

Objective:
Classify cybercrime incident subgraphs as:
  0 = normal incident graph
  1 = suspicious incident graph / potential mule network

Features Used (Structural & Financial):
- num_nodes
- num_edges
- num_account_nodes
- num_atm_nodes
- num_terminal_nodes
- max_hop
- total_transaction_value
- max_transaction_value
- avg_transaction_value
- num_cash_out_edges
- in_degree_incident
- out_degree_incident
- density
- number_of_connected_components
- average_degree

Strict Leakage Prevention:
Ground-truth evaluation labels and identifiers are strictly excluded from model features:
- contains_suspicious_activity (target)
- suspicious_ring_count (ground truth)
- is_suspicious (ground truth)
- ring_id (ground truth)
- ground_truth_entity_id (offline ground truth)
- complaint_id, incident_entity_id, incident_time, window_start, window_end

Outputs:
1. models/xgboost_baseline.json - Serialized trained XGBoost model
2. models/xgboost_features.json - Feature schema
3. data/xgboost_predictions.csv - Test set predictions with actual, probability, predicted label
4. data/xgboost_threshold_analysis.csv - Precision, Recall, F1 across thresholds
5. data/xgboost_feature_importance.csv - Gain-based feature importances
6. data/xgboost_pr_curve.png - Precision-Recall curve
7. data/xgboost_roc_curve.png - ROC curve
8. data/xgboost_feature_importance.png - Feature importance bar plot
9. data/xgboost_temporal_evaluation.csv - Chronological evaluation robustness check
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


# ==============================================================================
# Configuration & Paths
# ==============================================================================

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
GRAPH_SUMMARY_FILE = DATA_DIR / "graph_summary.csv"

MODEL_FILE = MODELS_DIR / "xgboost_baseline.json"
FEATURE_SCHEMA_FILE = MODELS_DIR / "xgboost_features.json"
PREDICTIONS_FILE = DATA_DIR / "xgboost_predictions.csv"
THRESHOLD_ANALYSIS_FILE = DATA_DIR / "xgboost_threshold_analysis.csv"
FEATURE_IMPORTANCE_FILE = DATA_DIR / "xgboost_feature_importance.csv"
TEMPORAL_EVAL_FILE = DATA_DIR / "xgboost_temporal_evaluation.csv"

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


# ==============================================================================
# Data Loading & Quality Checks
# ==============================================================================

def load_graph_summary(summary_path: Path = GRAPH_SUMMARY_FILE) -> pd.DataFrame:
    """Loads graph summary dataset."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required graph summary file: {summary_path}")
    return pd.read_csv(summary_path)


def validate_data_quality(df: pd.DataFrame) -> None:
    """
    Performs comprehensive data-quality checks on the input summary table.
    """
    print("\n" + "=" * 55)
    print("        DATA QUALITY & LEAKAGE AUDIT")
    print("=" * 55)
    print(f"Dataset Shape              : {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Missing values check
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Total Missing Values       : {total_missing}")
    if total_missing > 0:
        print("Missing by column:")
        print(missing_counts[missing_counts > 0])
        
    # Duplicate check
    dup_complaints = df["complaint_id"].duplicated().sum()
    print(f"Duplicate complaint_ids    : {dup_complaints}")
    assert dup_complaints == 0, "Duplicate complaint_ids detected in graph_summary.csv!"

    # Target distribution
    assert TARGET_COL in df.columns, f"Target column {TARGET_COL} missing!"
    target_counts = df[TARGET_COL].value_counts().to_dict()
    neg_count = target_counts.get(0, 0)
    pos_count = target_counts.get(1, 0)
    pos_pct = (pos_count / len(df)) * 100
    print(f"Target Distribution        : Normal (0)={neg_count}, Suspicious (1)={pos_count} ({pos_pct:.1f}%)")
    assert pos_count > 0 and neg_count > 0, "Target does not contain both classes!"
    print("-" * 55)


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Selects valid numerical graph/financial features while strictly excluding labels.
    """
    excluded_found = [col for col in df.columns if col in EXCLUDED_COLUMNS or "suspicious" in col.lower() or "ring" in col.lower()]
    
    # Feature candidate list
    candidate_features = [
        "num_nodes",
        "num_edges",
        "num_account_nodes",
        "num_atm_nodes",
        "num_terminal_nodes",
        "max_hop",
        "total_transaction_value",
        "max_transaction_value",
        "avg_transaction_value",
        "num_cash_out_edges",
        "in_degree_incident",
        "out_degree_incident",
        "density",
        "number_of_connected_components",
        "average_degree"
    ]
    
    # Filter features that exist in df and are numeric
    selected_features = [f for f in candidate_features if f in df.columns and f not in excluded_found]
    
    print("\nFeatures Excluded (Labels / Identifiers / Leakage Prevention):")
    for col in sorted(list(set(excluded_found))):
        print(f"  - {col}")
        
    print("\nFinal Model Features:")
    for idx, f in enumerate(selected_features, 1):
        print(f"  {idx:>2}. {f}")
    print("-" * 55 + "\n")
    
    # Verify no excluded label accidentally leaked
    for col in selected_features:
        assert col not in EXCLUDED_COLUMNS, f"Data leakage detected! Label column {col} in feature list!"
        assert not np.issubdtype(df[col].dtype, np.object_), f"Non-numeric feature column {col}!"

    X = df[selected_features].copy()
    # Handle missing with median imputation if any
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            
    y = df[TARGET_COL].copy().astype(int)
    return X, y, selected_features, excluded_found


# ==============================================================================
# Model Training & Class Imbalance Handling
# ==============================================================================

def train_xgboost_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE
) -> Tuple[xgb.XGBClassifier, float]:
    """
    Trains an XGBoost baseline classifier with scale_pos_weight derived from training set.
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    print("=" * 55)
    print("         MODEL CONFIGURATION & TRAINING")
    print("=" * 55)
    print(f"Training Samples (Negative) : {neg_count}")
    print(f"Training Samples (Positive) : {pos_count}")
    print(f"Class Imbalance Ratio       : {scale_pos_weight:.2f}:1")
    print(f"scale_pos_weight Applied    : {scale_pos_weight:.4f}")

    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "eval_metric": "logloss"
    }

    print("Model Parameters:")
    for k, v in params.items():
        print(f"  - {k:<20}: {v}")
    print("-" * 55)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    print("Model training completed successfully.\n")
    return model, scale_pos_weight


# ==============================================================================
# Model Evaluation & Metrics
# ==============================================================================

def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    default_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates the model on the untouched test set and computes comprehensive metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= default_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_prob": y_prob,
        "y_pred": y_pred
    }
    return metrics


def analyze_thresholds(
    y_test: pd.Series,
    y_prob: np.ndarray,
    output_path: Path = THRESHOLD_ANALYSIS_FILE
) -> pd.DataFrame:
    """
    Evaluates precision, recall, and F1 across various decision thresholds.
    """
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    records = []

    for t in thresholds:
        y_p = (y_prob >= t).astype(int)
        p = precision_score(y_test, y_p, zero_division=0)
        r = recall_score(y_test, y_p, zero_division=0)
        f = f1_score(y_test, y_p, zero_division=0)
        records.append({
            "threshold": round(t, 2),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4)
        })

    df_thresh = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_thresh.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved threshold analysis to: {output_path}")
    return df_thresh


# ==============================================================================
# Visualizations: PR Curve, ROC Curve, Feature Importance
# ==============================================================================

def plot_precision_recall_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
    pr_auc: float,
    output_path: Path = PR_CURVE_FILE
) -> None:
    """Renders and saves the Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    baseline = (y_test == 1).mean()

    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.plot(recall, precision, color="#1D3557", lw=2.5, label=f"XGBoost Baseline (PR-AUC = {pr_auc:.4f})")
    plt.axhline(y=baseline, color="#E63946", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline:.3f})")
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curve — XGBoost Baseline", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved Precision-Recall curve to: {output_path}")


def plot_roc_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
    roc_auc: float,
    output_path: Path = ROC_CURVE_FILE
) -> None:
    """Renders and saves the ROC Curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.plot(fpr, tpr, color="#2A9D8F", lw=2.5, label=f"XGBoost Baseline (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#6C757D", linestyle="--", lw=1.5, label="Random Guess (0.5000)")
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved ROC curve to: {output_path}")


def calculate_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    output_csv: Path = FEATURE_IMPORTANCE_FILE,
    output_png: Path = FEATURE_IMPORTANCE_PLOT_FILE
) -> pd.DataFrame:
    """
    Extracts gain-based feature importances, exports CSV, and plots horizontal bar chart.
    """
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type="gain")
    
    # Map feature names
    records = []
    for f in feature_names:
        records.append({
            "feature": f,
            "importance": float(importance_dict.get(f, 0.0))
        })

    df_imp = pd.DataFrame(records).sort_values("importance", ascending=False).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved feature importance to: {output_csv}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.clf()
    top_features = df_imp.head(12).sort_values("importance", ascending=True)
    plt.barh(top_features["feature"], top_features["importance"], color="#457B9D", edgecolor="#1D3557", alpha=0.9)
    plt.xlabel("Importance (Gain)", fontsize=11, fontweight="bold")
    plt.title("XGBoost Baseline — Top Feature Importance (Gain)", fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved feature importance plot to: {output_png}")
    return df_imp


# ==============================================================================
# Human-Readable Feature Explanation & Predictions Export
# ==============================================================================

def generate_feature_explanation(row: pd.Series) -> List[str]:
    """
    Rule-based feature explanation function for an individual incident graph.
    Clearly labeled as 'Feature-based explanation'.
    """
    reasons: List[str] = []

    if row.get("num_edges", 0) >= 6:
        reasons.append("High transaction activity detected within the 72-hour incident window.")
    if row.get("num_cash_out_edges", 0) >= 1:
        reasons.append(f"Multiple cash-out transactions ({int(row['num_cash_out_edges'])}) were observed.")
    if row.get("num_atm_nodes", 0) >= 1:
        reasons.append(f"The graph contains terminal ATM cash-out nodes ({int(row['num_atm_nodes'])}).")
    if row.get("total_transaction_value", 0.0) >= 100000.0:
        reasons.append(f"Cumulative transaction volume (₹{row['total_transaction_value']:,.2f}) is unusually elevated.")
    if row.get("max_hop", 0) >= 2:
        reasons.append(f"Multi-hop fund forwarding structure spans {int(row['max_hop'])} hops from the incident entity.")
    if row.get("in_degree_incident", 0) >= 2 and row.get("out_degree_incident", 0) >= 1:
        reasons.append("Incident entity exhibits rapid fund aggregation and downstream redirection.")

    if not reasons:
        reasons.append("Isolated or standard low-volume peer-to-peer activity.")

    return reasons


def generate_predictions_table(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    output_path: Path = PREDICTIONS_FILE
) -> pd.DataFrame:
    """
    Generates and saves model predictions on the test set with relevant metadata.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    df_preds = meta_test.copy()
    df_preds["actual_label"] = y_test.values
    df_preds["predicted_probability"] = np.round(y_prob, 4)
    df_preds["predicted_label"] = y_pred

    cols_order = [
        "complaint_id",
        "incident_entity_id",
        "actual_label",
        "predicted_probability",
        "predicted_label",
        "num_nodes",
        "num_edges",
        "total_transaction_value",
        "num_cash_out_edges"
    ]
    available_cols = [c for c in cols_order if c in df_preds.columns]
    df_preds = df_preds[available_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved test set predictions to: {output_path}")
    return df_preds


# ==============================================================================
# Secondary Chronological Temporal Evaluation
# ==============================================================================

def run_temporal_evaluation(
    df_summary: pd.DataFrame,
    feature_names: List[str],
    target_col: str = TARGET_COL,
    output_path: Path = TEMPORAL_EVAL_FILE
) -> pd.DataFrame:
    """
    Secondary robustness evaluation using a chronological train/test split:
    Earlier 80% incidents -> Training, Later 20% incidents -> Testing.
    """
    df_sorted = df_summary.sort_values("incident_time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_tr = train_df[feature_names]
    y_tr = train_df[target_col].astype(int)
    X_te = test_df[feature_names]
    y_te = test_df[target_col].astype(int)

    neg_c = int((y_tr == 0).sum())
    pos_c = int((y_tr == 1).sum())
    spw = float(neg_c / pos_c) if pos_c > 0 else 1.0

    model_temp = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )
    model_temp.fit(X_tr, y_tr)

    y_p = model_temp.predict_proba(X_te)[:, 1]
    y_pred = (y_p >= 0.5).astype(int)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_te, y_p)
    pr_auc = average_precision_score(y_te, y_p)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

    rec_temp = [{
        "split_type": "Chronological (Earliest 80% Train -> Latest 20% Test)",
        "train_start": str(train_df["incident_time"].min()),
        "train_end": str(train_df["incident_time"].max()),
        "test_start": str(test_df["incident_time"].min()),
        "test_end": str(test_df["incident_time"].max()),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }]

    df_temp = pd.DataFrame(rec_temp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_temp.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved chronological temporal evaluation to: {output_path}")
    return df_temp


# ==============================================================================
# Model Persistence & Automated Validation Checks
# ==============================================================================

def save_model(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    model_path: Path = MODEL_FILE,
    feature_path: Path = FEATURE_SCHEMA_FILE
) -> None:
    """Saves trained model and feature list."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"[SUCCESS] Saved trained model to: {model_path}")

    with open(feature_path, "w") as f:
        json.dump({"features": feature_names}, f, indent=2)
    print(f"[SUCCESS] Saved feature schema to: {feature_path}")


def run_pipeline_validations(
    df_summary: pd.DataFrame,
    feature_names: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    df_preds: pd.DataFrame,
    model: xgb.XGBClassifier
) -> None:
    """
    Automated validation suite verifying data integrity, lack of leakage, and reproducibility.
    """
    # 1. Dataset existence & uniqueness
    assert len(df_summary) == 1000, f"Expected 1000 records, got {len(df_summary)}"
    assert df_summary["complaint_id"].nunique() == 1000, "complaint_id values are not unique!"

    # 2. Target check
    assert TARGET_COL in df_summary.columns, f"Target {TARGET_COL} missing!"
    assert set(df_summary[TARGET_COL].unique()) == {0, 1}, "Target must contain both 0 and 1 classes!"

    # 3. Leakage audit
    for f in feature_names:
        assert f not in EXCLUDED_COLUMNS, f"Data leakage error: {f} in feature list!"
        assert "suspicious" not in f.lower(), f"Suspicious substring in feature name {f}!"
        assert "ring" not in f.lower(), f"Ring substring in feature name {f}!"

    # 4. NaNs check
    assert not X_test.isnull().any().any(), "NaN values detected in test set features!"

    # 5. Prediction integrity
    assert len(df_preds) == len(y_test), "Prediction rows mismatch test set size!"
    assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all(), "Probabilities outside [0, 1] range!"

    # 6. Model reload check
    reloaded_model = xgb.XGBClassifier()
    reloaded_model.load_model(str(MODEL_FILE))
    reloaded_prob = reloaded_model.predict_proba(X_test)[:, 1]
    np.testing.assert_allclose(y_prob, reloaded_prob, rtol=1e-5, err_msg="Reloaded model predictions mismatch!")

    print("All Stage 3A XGBoost baseline validations PASSED successfully!")


# ==============================================================================
# Main Pipeline Routine
# ==============================================================================

def main():
    print("=" * 60)
    print("       STAGE 3A — XGBOOST BASELINE CLASSIFIER")
    print("=" * 60)

    # 1. Load Dataset
    df_summary = load_graph_summary()

    # 2. Data Quality & Leakage Audit
    validate_data_quality(df_summary)

    # 3. Select Features & Exclude Labels
    X, y, feature_names, excluded_cols = select_features(df_summary)

    # 4. Stratified Train / Test Split (80% / 20%)
    meta_cols = [c for c in ["complaint_id", "incident_entity_id", "num_nodes", "num_edges", "total_transaction_value", "num_cash_out_edges"] if c in df_summary.columns]
    df_meta = df_summary[meta_cols]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df_meta,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 5. Train XGBoost Model with scale_pos_weight
    model, scale_pos_weight = train_xgboost_baseline(X_train, y_train, random_state=RANDOM_STATE)

    # 6. Evaluate on Untouched Test Set
    metrics = evaluate_model(model, X_test, y_test, default_threshold=0.5)

    # 7. Threshold Analysis
    df_thresholds = analyze_thresholds(y_test, metrics["y_prob"])

    # 8. Visualizations
    plot_precision_recall_curve(y_test, metrics["y_prob"], metrics["pr_auc"])
    plot_roc_curve(y_test, metrics["y_prob"], metrics["roc_auc"])
    df_importance = calculate_feature_importance(model, feature_names)

    # 9. Predictions Export
    df_preds = generate_predictions_table(model, X_test, y_test, meta_test)

    # 10. Secondary Chronological Temporal Evaluation
    df_temporal = run_temporal_evaluation(df_summary, feature_names)

    # 11. Save Model & Features
    save_model(model, feature_names)

    # 12. Run Pipeline Validations
    run_pipeline_validations(df_summary, feature_names, X_test, y_test, metrics["y_prob"], df_preds, model)

    # 13. Final Structured Report
    print("\n" + "=" * 55)
    print("           STAGE 3A — XGBOOST BASELINE REPORT")
    print("=" * 55)
    print(f"Dataset Total Graphs       : {len(df_summary)}")
    print(f"Training Samples           : {len(X_train)} (80%)")
    print(f"Testing Samples            : {len(X_test)} (20%)")
    print(f"Features Count             : {len(feature_names)}")
    print(f"Class Distribution (Total) : Normal={int((y==0).sum())}, Suspicious={int((y==1).sum())}")
    print(f"Class Distribution (Test)  : Normal={int((y_test==0).sum())}, Suspicious={int((y_test==1).sum())}")
    print("-" * 55)
    print("PERFORMANCE (Test Set @ Threshold = 0.50):")
    print(f"  Accuracy                 : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision                : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall                   : {metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score                 : {metrics['f1'] * 100:.2f}%")
    print(f"  ROC-AUC                  : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC                   : {metrics['pr_auc']:.4f}")
    print("-" * 55)
    print("CONFUSION MATRIX:")
    print(f"  True Negatives (TN)      : {metrics['tn']}")
    print(f"  False Positives (FP)     : {metrics['fp']}")
    print(f"  False Negatives (FN)     : {metrics['fn']}")
    print(f"  True Positives (TP)      : {metrics['tp']}")
    print("-" * 55)
    print("TOP 5 FEATURES BY GAIN IMPORTANCE:")
    for idx, row in df_importance.head(5).iterrows():
        print(f"  {idx + 1}. {row['feature']:<28} (Gain: {row['importance']:.4f})")
    print("-" * 55)

    # Best experimental F1 threshold
    best_thresh_row = df_thresholds.loc[df_thresholds["f1"].idxmax()]
    print(f"Experimental Optimal F1 Threshold: {best_thresh_row['threshold']:.2f} (F1: {best_thresh_row['f1'] * 100:.2f}%, Precision: {best_thresh_row['precision'] * 100:.2f}%, Recall: {best_thresh_row['recall'] * 100:.2f}%)")
    print("-" * 55)

    # Example Feature-based Explanations
    print("\nFeature-Based Explanations (Demonstration on Test Incidents):")
    sample_pos = df_preds[df_preds["actual_label"] == 1].iloc[0]
    sample_neg = df_preds[df_preds["actual_label"] == 0].iloc[0]

    for label_type, s_row in [("Suspicious Incident", sample_pos), ("Normal Incident", sample_neg)]:
        print(f"\n--- {label_type}: Complaint {s_row['complaint_id']} (Entity: {s_row['incident_entity_id']}) ---")
        print(f"Actual: {s_row['actual_label']} | Pred Prob: {s_row['predicted_probability']:.4f} | Pred Class: {s_row['predicted_label']}")
        # Look up feature row
        feat_row = df_summary[df_summary["complaint_id"] == s_row["complaint_id"]].iloc[0]
        reasons = generate_feature_explanation(feat_row)
        print("Feature-based explanation:")
        for r in reasons:
            print(f"  • {r}")

    print("\n" + "=" * 55)
    print("Stage 3A XGBoost baseline is complete and ready for comparison with GraphSAGE.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
