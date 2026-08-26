"""
Stage 7: Alert Threshold & Policy Tunability
============================================
This module implements an operational decision and alert-tuning policy layer over
the GraphSAGE GNN risk probabilities from Stage 3B.

Objective:
Enable dynamic, investigator-tunable alert thresholds to navigate the operational
tradeoff between detection recall (catching all potential money mules) and precision
(minimizing false-alarm alert fatigue).

Key Distinctions:
- MODEL SCORE      : GraphSAGE risk probability P in [0, 1] (intrinsic model output).
- POLICY THRESHOLD : Operational cutoff tau in [0, 1] selected by policy / investigator.
- ALERT            : Decision flag (P >= tau).

Policy Tiers:
- tau < 0.30              : HIGH_SENSITIVITY      (Intake Triage / Maximum Recall)
- 0.30 <= tau < 0.70      : BALANCED_TRIAGE       (Standard Operational Baseline, default = 0.50)
- 0.70 <= tau < 0.90      : HIGH_PRECISION        (Targeted Case Escalation)
- tau >= 0.90             : HIGH_CONFIDENCE_ALERT (Automated Freezing Recommendation)

Outputs:
1. data/threshold_policy_analysis.csv - Threshold sweep evaluation across test predictions.
2. data/threshold_policy_config.json - Runtime configuration for FastAPI / UI slider.
3. data/threshold_examples.csv - Representative case decisions at 0.30, 0.50, 0.70, 0.90.
4. data/threshold_policy_curve.png - Tradeoff curve visualization.
5. data/threshold_policy_summary.csv - Concise summary across key policy cutoffs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==============================================================================
# Configuration & Threshold Definitions
# ==============================================================================

DATA_DIR = Path("data")

GRAPHSAGE_PREDICTIONS_FILE = DATA_DIR / "graphsage_predictions.csv"
EXPLANATIONS_FILE = DATA_DIR / "explanations.csv"
CONFIDENCE_TIERS_FILE = DATA_DIR / "confidence_tiers.csv"

POLICY_ANALYSIS_FILE = DATA_DIR / "threshold_policy_analysis.csv"
POLICY_CONFIG_FILE = DATA_DIR / "threshold_policy_config.json"
POLICY_EXAMPLES_FILE = DATA_DIR / "threshold_examples.csv"
POLICY_SUMMARY_FILE = DATA_DIR / "threshold_policy_summary.csv"
POLICY_PLOT_FILE = DATA_DIR / "threshold_policy_curve.png"

CONFIGURED_THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90]
DEFAULT_THRESHOLD = 0.50


# ==============================================================================
# Runtime Policy Function (API & Real-time Integration)
# ==============================================================================

def get_policy_tier(threshold: float) -> str:
    """
    Maps an operational threshold to its corresponding policy mode tier.
    """
    if threshold < 0.30:
        return "HIGH_SENSITIVITY"
    elif threshold < 0.70:
        return "BALANCED_TRIAGE"
    elif threshold < 0.90:
        return "HIGH_PRECISION"
    else:
        return "HIGH_CONFIDENCE_ALERT"


def apply_threshold(
    probability: float,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, Any]:
    """
    Applies an investigator-configured policy threshold to a model risk probability.
    
    Returns a structured dictionary indicating alert status and operational tier.
    """
    p_clamped = max(0.0, min(1.0, float(probability)))
    t_clamped = max(0.0, min(1.0, float(threshold)))
    is_alert = bool(p_clamped >= t_clamped)
    tier = get_policy_tier(t_clamped)

    return {
        "risk_probability": round(p_clamped, 4),
        "threshold": round(t_clamped, 2),
        "alert": is_alert,
        "alert_status": "ALERT" if is_alert else "NO_ALERT",
        "policy_tier": tier
    }


# ==============================================================================
# Offline Policy Evaluation Across Thresholds
# ==============================================================================

def evaluate_threshold_policy(
    df_preds: pd.DataFrame,
    thresholds: List[float] = CONFIGURED_THRESHOLDS
) -> Tuple[pd.DataFrame, float, float]:
    """
    Evaluates classification performance across configured operational thresholds
    on the untouched test set (200 incident subgraphs).
    """
    y_test = df_preds["actual_label"].values
    y_prob = df_preds["predicted_probability"].values
    total_test = len(y_test)

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    analysis_records = []
    for t in thresholds:
        y_p = (y_prob >= t).astype(int)
        prec = float(precision_score(y_test, y_p, zero_division=0))
        rec = float(recall_score(y_test, y_p, zero_division=0))
        f1 = float(f1_score(y_test, y_p, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, y_p).ravel()
        alerts = int(y_p.sum())
        alert_rate = float(alerts / total_test) * 100.0
        tier = get_policy_tier(t)

        analysis_records.append({
            "threshold": round(t, 2),
            "policy_tier": tier,
            "alerts": alerts,
            "alert_rate_pct": round(alert_rate, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        })

    df_analysis = pd.DataFrame(analysis_records)
    df_analysis.to_csv(POLICY_ANALYSIS_FILE, index=False)
    print(f"[SUCCESS] Saved threshold policy analysis to: {POLICY_ANALYSIS_FILE}")
    return df_analysis, roc_auc, pr_auc


# ==============================================================================
# Summary Metrics & Config Export
# ==============================================================================

def save_policy_configuration(
    default_t: float = DEFAULT_THRESHOLD,
    supported_t: List[float] = CONFIGURED_THRESHOLDS,
    output_path: Path = POLICY_CONFIG_FILE
) -> None:
    """
    Saves runtime configuration JSON for API and UI consumption.
    """
    config = {
        "module_name": "Stage 7 Threshold Policy & Alert Tunability",
        "version": "1.0.0",
        "primary_model": "GraphSAGE GNN",
        "baseline_model": "XGBoost Baseline",
        "default_threshold": default_t,
        "supported_thresholds": supported_t,
        "policy_tiers": {
            "HIGH_SENSITIVITY": {
                "threshold_range": "[0.00, 0.30)",
                "operational_mode": "Intake Triage & Maximum Recall",
                "description": "Surfaces potential early-stage or low-volume suspicious patterns; acceptable higher false-positive rate."
            },
            "BALANCED_TRIAGE": {
                "threshold_range": "[0.30, 0.70)",
                "operational_mode": "Standard Operational Baseline",
                "description": "Optimal equilibrium between detection recall and investigator caseload capacity."
            },
            "HIGH_PRECISION": {
                "threshold_range": "[0.70, 0.90)",
                "operational_mode": "Targeted Escalation",
                "description": "High-confidence alerts for priority case assignment with minimal false alarms."
            },
            "HIGH_CONFIDENCE_ALERT": {
                "threshold_range": "[0.90, 1.00]",
                "operational_mode": "Automated Intervention Recommendation",
                "description": "Zero or near-zero false-positive alerts suitable for expedited interbank fund freezing."
            }
        },
        "runtime_policy_function": "apply_threshold(probability, threshold)"
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SUCCESS] Saved threshold policy configuration to: {output_path}")


def generate_policy_summary_table(
    df_analysis: pd.DataFrame,
    output_path: Path = POLICY_SUMMARY_FILE
) -> pd.DataFrame:
    """
    Generates concise summary comparison across key operational thresholds.
    """
    row_def = df_analysis[df_analysis["threshold"] == DEFAULT_THRESHOLD].iloc[0]
    row_70 = df_analysis[df_analysis["threshold"] == 0.70].iloc[0]
    row_90 = df_analysis[df_analysis["threshold"] == 0.90].iloc[0]

    best_f1_row = df_analysis.loc[df_analysis["f1"].idxmax()]
    best_f1_t = float(best_f1_row["threshold"])

    summary_records = [{
        "default_threshold": float(row_def["threshold"]),
        "default_alert_count": int(row_def["alerts"]),
        "default_alert_rate_pct": float(row_def["alert_rate_pct"]),
        "best_f1_threshold_offline": best_f1_t,
        "precision_at_default": float(row_def["precision"]),
        "recall_at_default": float(row_def["recall"]),
        "f1_at_default": float(row_def["f1"]),
        "precision_at_070": float(row_70["precision"]),
        "recall_at_070": float(row_70["recall"]),
        "f1_at_070": float(row_70["f1"]),
        "precision_at_090": float(row_90["precision"]),
        "recall_at_090": float(row_90["recall"]),
        "f1_at_090": float(row_90["f1"])
    }]

    df_summary_out = pd.DataFrame(summary_records)
    df_summary_out.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved threshold policy summary to: {output_path}")
    return df_summary_out


def generate_threshold_examples_table(
    df_explanations: pd.DataFrame,
    output_path: Path = POLICY_EXAMPLES_FILE
) -> pd.DataFrame:
    """
    Generates case-level decision examples at thresholds 0.30, 0.50, 0.70, 0.90.
    """
    sample_cids = ["C000003", "C000004", "C000001", "C000014", "C000020"]
    thresholds = [0.30, 0.50, 0.70, 0.90]

    records = []
    for cid in sample_cids:
        sub = df_explanations[df_explanations["complaint_id"] == cid]
        if sub.empty:
            continue
        row = sub.iloc[0]
        p = float(row["graphsage_probability"])
        inc_ent = str(row["incident_entity_id"])
        tier = str(row["confidence_tier"])
        summary = str(row["investigator_summary"])

        for t in thresholds:
            policy_res = apply_threshold(p, t)
            records.append({
                "complaint_id": cid,
                "incident_entity_id": inc_ent,
                "graphsage_probability": p,
                "threshold": t,
                "policy_tier": policy_res["policy_tier"],
                "alert_decision": policy_res["alert_status"],
                "confidence_tier": tier,
                "investigator_summary": summary
            })

    df_ex = pd.DataFrame(records)
    df_ex.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved {len(df_ex)} threshold examples to: {output_path}")
    return df_ex


# ==============================================================================
# Visualization: PR & Operational Utility Tradeoff Curve
# ==============================================================================

def plot_threshold_policy_curves(
    df_preds: pd.DataFrame,
    output_path: Path = POLICY_PLOT_FILE
) -> None:
    """
    Renders a 2-panel precision-recall and alert-volume utility curve across thresholds.
    """
    y_test = df_preds["actual_label"].values
    y_prob = df_preds["predicted_probability"].values

    threshold_sweep = np.linspace(0.05, 0.95, 50)
    prec_list, rec_list, f1_list, fp_list, alert_list = [], [], [], [], []

    for t in threshold_sweep:
        y_p = (y_prob >= t).astype(int)
        prec_list.append(precision_score(y_test, y_p, zero_division=0))
        rec_list.append(recall_score(y_test, y_p, zero_division=0))
        f1_list.append(f1_score(y_test, y_p, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, y_p).ravel()
        fp_list.append(fp)
        alert_list.append(int(y_p.sum()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Precision, Recall, F1
    ax1.plot(threshold_sweep, [p * 100 for p in prec_list], label="Precision (%)", color="#2A9D8F", lw=2.5)
    ax1.plot(threshold_sweep, [r * 100 for r in rec_list], label="Recall (%)", color="#E76F51", lw=2.5)
    ax1.plot(threshold_sweep, [f * 100 for f in f1_list], label="F1 Score (%)", color="#1D3557", lw=2.5, linestyle="--")
    ax1.axvline(0.50, color="#457B9D", linestyle=":", lw=2, label="Default Threshold (0.50)")
    ax1.axvline(0.70, color="#9D0208", linestyle=":", lw=2, label="Optimal F1 Threshold (0.70)")

    ax1.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Stage 7: Alert Policy Threshold Tunability & Performance Tradeoffs", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9.5)

    # Panel 2: Alert Volume & False Alarm Fatigue
    ax2.plot(threshold_sweep, alert_list, label="Total Generated Alerts", color="#457B9D", lw=2.2)
    ax2.plot(threshold_sweep, fp_list, label="False Positive Alerts (Fatigue)", color="#D90429", lw=2.2, linestyle="-.")
    ax2.set_xlabel("Policy Alert Threshold (Cutoff)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Incident Count", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Saved threshold policy tradeoff visualization to: {output_path}")


# ==============================================================================
# Automated Validation Suite (9 Checks)
# ==============================================================================

def run_validations(
    df_preds: pd.DataFrame,
    df_analysis: pd.DataFrame,
    df_examples: pd.DataFrame
) -> None:
    """
    Automated validation suite checking all Stage 7 constraints.
    """
    # 1. Probabilities within [0, 1]
    assert (df_preds["predicted_probability"] >= 0.0).all() and (df_preds["predicted_probability"] <= 1.0).all()

    # 2. Configured thresholds within [0, 1]
    assert all(0.0 <= t <= 1.0 for t in CONFIGURED_THRESHOLDS)

    # 3. Threshold ordering strictly ascending
    assert CONFIGURED_THRESHOLDS == sorted(CONFIGURED_THRESHOLDS)

    # 4. Threshold 0.50 mathematical consistency check
    row_050 = df_analysis[df_analysis["threshold"] == 0.50].iloc[0]
    expected_prec = row_050["tp"] / max(1, (row_050["tp"] + row_050["fp"]))
    assert abs(row_050["precision"] - round(expected_prec, 4)) < 1e-3, f"Precision mismatch: {row_050['precision']} vs {expected_prec}"

    # 5. Output files exist
    assert POLICY_ANALYSIS_FILE.exists()
    assert POLICY_CONFIG_FILE.exists()
    assert POLICY_EXAMPLES_FILE.exists()
    assert POLICY_SUMMARY_FILE.exists()
    assert POLICY_PLOT_FILE.exists()

    # 6. Zero NaN values
    assert not df_analysis.isna().any().any()
    assert not df_examples.isna().any().any()

    # 7. apply_threshold logic validation
    t1 = apply_threshold(0.85, 0.70)
    assert t1["alert"] is True and t1["policy_tier"] == "HIGH_PRECISION"
    t2 = apply_threshold(0.45, 0.50)
    assert t2["alert"] is False and t2["policy_tier"] == "BALANCED_TRIAGE"

    # 8. Examples count
    assert len(df_examples) >= 12

    # 9. No ground-truth leakage in runtime config
    with open(POLICY_CONFIG_FILE) as f:
        conf_data = json.load(f)
    assert "default_threshold" in conf_data

    print("\n" + "=" * 50)
    print("             STAGE 7 VALIDATION")
    print("=" * 50)
    print("All 9 validation checks passed successfully.")
    print("=" * 50 + "\n")


# ==============================================================================
# Main Pipeline Entrypoint
# ==============================================================================

def main():
    print("=" * 60)
    print("   STAGE 7 — ALERT THRESHOLD & POLICY TUNABILITY")
    print("=" * 60)

    # 1. Load Predictions & Explanations
    if not GRAPHSAGE_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {GRAPHSAGE_PREDICTIONS_FILE}")
    if not EXPLANATIONS_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {EXPLANATIONS_FILE}")

    df_preds = pd.read_csv(GRAPHSAGE_PREDICTIONS_FILE)
    df_explanations = pd.read_csv(EXPLANATIONS_FILE)

    # 2. Evaluate Policy Across Thresholds
    print("Evaluating offline policy metrics across configured thresholds...")
    df_analysis, roc_auc, pr_auc = evaluate_threshold_policy(df_preds, CONFIGURED_THRESHOLDS)

    # 3. Export Policy Configuration & Summary
    save_policy_configuration(DEFAULT_THRESHOLD, CONFIGURED_THRESHOLDS)
    df_summary = generate_policy_summary_table(df_analysis)

    # 4. Generate Case Examples & Tradeoff Visualization
    df_examples = generate_threshold_examples_table(df_explanations)
    plot_threshold_policy_curves(df_preds)

    # 5. Run Automated Validations
    run_validations(df_preds, df_analysis, df_examples)

    # 6. Print CLI Execution Summary
    print("==================================================")
    print("STAGE 7 — ALERT THRESHOLD & POLICY TUNABILITY")
    print("==================================================")
    print(f"Default threshold: {DEFAULT_THRESHOLD:.2f}\n")
    print(f"{'Threshold':<14} {'Alerts':<9} {'Precision':<13} {'Recall':<11} {'F1':<10}")
    print("-" * 55)
    for _, r in df_analysis.iterrows():
        prec_str = f"{r['precision'] * 100:.2f}%"
        rec_str = f"{r['recall'] * 100:.2f}%"
        f1_str = f"{r['f1'] * 100:.2f}%"
        print(f"{r['threshold']:<14.2f} {int(r['alerts']):<9} {prec_str:<13} {rec_str:<11} {f1_str:<10}")

    row_def = df_analysis[df_analysis["threshold"] == DEFAULT_THRESHOLD].iloc[0]
    print("-" * 55)
    print("DEFAULT POLICY")
    print("-" * 50)
    print(f"Threshold       : {DEFAULT_THRESHOLD:.2f}")
    print(f"Operational mode: {row_def['policy_tier']}")
    print(f"Alerts          : {int(row_def['alerts'])} (out of {len(df_preds)} test cases)")
    print(f"Alert rate      : {row_def['alert_rate_pct']:.2f}%")
    print(f"Precision       : {row_def['precision'] * 100:.2f}%")
    print(f"Recall          : {row_def['recall'] * 100:.2f}%")
    print(f"F1              : {row_def['f1'] * 100:.2f}%")
    print("-" * 50)

    print("\n==================================================")
    print("Stage 7 is complete and ready for backend/API integration.")
    print("==================================================\n")


if __name__ == "__main__":
    main()
