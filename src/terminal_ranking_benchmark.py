"""
Stage 4 / Item 5: Multi-Dataset Terminal Ranking Benchmark
==========================================================
Evaluates Terminal Node / Exit Ranking on:
1. Dataset A (Synthetic): ATM Terminal cash-out exit ranking (with GPS coordinates).
2. Dataset B (IBM AML): Account-level exit ranking (out_degree ~ 0, max outflow, net retained).

GUARDRAIL NOTICE (Guardrails #1, #2, #5, #6):
- Geo-coordinates do NOT exist in IBM AML; marked explicitly unavailable.
- Performance is reported with exact sample size N, class balance, and hit rates.
- Evaluates whether ranking methodology generalizes to bank-to-bank ledger transfers.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from src.adapters.synthetic_adapter import SyntheticAdapter
from src.adapters.ibm_adapter import IBMAMLAdapter

DATA_DIR = Path("data")
OUTPUT_CSV = DATA_DIR / "terminal_ranking_multi_dataset_comparison.csv"


def main():
    print("=" * 70)
    print("   STAGE 4 / ITEM 5 ? TERMINAL RANKING BENCHMARK (DATASET A & B)")
    print("=" * 70)

    # -------------------------------------------------------------
    print("")
    print("[1/2] Dataset A (Synthetic Incident Subgraphs):")
    df_eval_a = pd.read_csv("data/terminal_prediction_evaluation.csv")
    top1_a = float(df_eval_a["top_1_hit_rate"].iloc[0]) / 100.0 if df_eval_a["top_1_hit_rate"].iloc[0] > 1.0 else float(df_eval_a["top_1_hit_rate"].iloc[0])
    top3_a = float(df_eval_a["top_3_hit_rate"].iloc[0]) / 100.0 if df_eval_a["top_3_hit_rate"].iloc[0] > 1.0 else float(df_eval_a["top_3_hit_rate"].iloc[0])
    mrr_a = float(df_eval_a["mean_reciprocal_rank_mrr"].iloc[0])
    n_incidents_a = int(df_eval_a["evaluable_cashout_incidents"].iloc[0])
    avg_cands_a = float(df_eval_a["avg_candidates_per_incident"].iloc[0])

    print(f"  ? Evaluable Laundering Incidents (N) : {n_incidents_a}")
    print(f"  ? Average Candidates per Incident    : {avg_cands_a:.2f}")
    print(f"  ? Top-1 Hit Rate                     : {top1_a*100:.2f}%")
    print(f"  ? Top-3 Hit Rate                     : {top3_a*100:.2f}%")
    print(f"  ? Mean Reciprocal Rank (MRR)         : {mrr_a:.4f}")

    # -------------------------------------------------------------
    # 2. DATASET B (IBM AML Bank-to-Bank Transactions)
    # -------------------------------------------------------------
    print("")
    print("[2/2] Dataset B (IBM AML Bank Ledger Transactions):")
    ibm = IBMAMLAdapter()
    print("  Loading IBM AML transactions...")
    df_ibm_raw = ibm.load_raw_transactions(nrows=1000000)
    print(f"  Loaded {len(df_ibm_raw):,} raw transactions.")

    df_ibm_tx = pd.DataFrame()
    df_ibm_tx["from_acc"] = "B" + df_ibm_raw["From Bank"].astype(str) + "_" + df_ibm_raw["Account"].astype(str)
    df_ibm_tx["to_acc"] = "B" + df_ibm_raw["To Bank"].astype(str) + "_" + df_ibm_raw["Account.1"].astype(str)
    df_ibm_tx["amount"] = df_ibm_raw["Amount Paid"].astype(float)
    df_ibm_tx["is_laundering"] = df_ibm_raw["Is Laundering"].astype(int)

    laundering_tx = df_ibm_tx[df_ibm_tx["is_laundering"] == 1]
    n_laundering_tx = len(laundering_tx)
    laundering_accounts = set(laundering_tx["from_acc"]).union(set(laundering_tx["to_acc"]))

    all_out = df_ibm_tx.groupby("from_acc").agg(
        out_degree=("to_acc", "count"),
        total_outflow=("amount", "sum"),
        max_outflow=("amount", "max")
    ).reset_index().rename(columns={"from_acc": "account"})

    all_in = df_ibm_tx.groupby("to_acc").agg(
        in_degree=("from_acc", "count"),
        total_inflow=("amount", "sum"),
        max_inflow=("amount", "max")
    ).reset_index().rename(columns={"to_acc": "account"})

    flow_df = pd.merge(all_in, all_out, on="account", how="outer").fillna(0)
    cand_df = flow_df[flow_df["account"].isin(laundering_accounts)].copy()

    true_exits = set(laundering_tx["to_acc"]).intersection(set(cand_df[cand_df["out_degree"] == 0]["account"]))
    cand_df["is_true_exit"] = cand_df["account"].isin(true_exits).astype(int)
    n_true_exits = len(true_exits)
    n_total_candidates = len(cand_df)

    # Ranking formulation
    s_out = 1.0 / (1.0 + cand_df["out_degree"])
    net_retained = np.maximum(0.0, cand_df["total_inflow"] - cand_df["total_outflow"])
    s_ret = net_retained / (cand_df["total_inflow"] + 1e-5)
    s_dom = cand_df["max_inflow"] / (cand_df["total_inflow"] + 1e-5)
    max_vol = np.log1p(cand_df["total_inflow"].max())
    s_vol = np.log1p(cand_df["total_inflow"]) / max(1.0, max_vol)

    cand_df["terminal_score"] = 0.40 * s_out + 0.30 * s_ret + 0.20 * s_dom + 0.10 * s_vol

    # Compute Local MRR per laundering source account
    reciprocal_ranks = []
    grouped_laundering = laundering_tx.groupby("from_acc")["to_acc"].apply(list).reset_index()

    for _, row in grouped_laundering.iterrows():
        dest_accounts = row["to_acc"]
        sub_cands = cand_df[cand_df["account"].isin(dest_accounts)].sort_values("terminal_score", ascending=False)
        ranks = np.where(sub_cands["is_true_exit"].values == 1)[0]
        if len(ranks) > 0:
            reciprocal_ranks.append(1.0 / (ranks[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    mrr_b = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    top1_b = float(np.mean([1.0 if rr == 1.0 else 0.0 for rr in reciprocal_ranks]))
    top3_b = float(np.mean([1.0 if rr >= 1.0/3.0 else 0.0 for rr in reciprocal_ranks]))

    print(f"  ? Evaluable Laundering Flow Clusters (N): {len(grouped_laundering):,}")
    print(f"  ? Candidate Accounts Evaluated          : {n_total_candidates:,} (Exit Sinks: {n_true_exits:,} | {n_true_exits/n_total_candidates*100:.2f}%)")
    print(f"  ? Local Top-1 Hit Rate                  : {top1_b*100:.2f}%")
    print(f"  ? Local Top-3 Hit Rate                  : {top3_b*100:.2f}%")
    print(f"  ? Mean Reciprocal Rank (MRR)            : {mrr_b:.4f}")

    df_multi_term = pd.DataFrame([
        {
            "dataset": "Dataset A (Synthetic)",
            "entity_level": "ATM Terminal Nodes (Physical Hardware)",
            "geo_coordinates": "AVAILABLE (Latitude/Longitude for 50 ATMs)",
            "n_evaluable_incidents": n_incidents_a,
            "n_candidates_evaluated": int(n_incidents_a * avg_cands_a),
            "top_1_hit_rate": round(top1_a, 4),
            "top_3_hit_rate": round(top3_a, 4),
            "mrr": round(mrr_a, 4),
            "notes": "Synthetic benchmark with clean ATM exit labels."
        },
        {
            "dataset": "Dataset B (IBM AML)",
            "entity_level": "Bank Accounts (out_degree=0 Exit Sinks)",
            "geo_coordinates": "UNAVAILABLE (No physical GPS recorded)",
            "n_evaluable_incidents": len(grouped_laundering),
            "n_candidates_evaluated": n_total_candidates,
            "top_1_hit_rate": round(top1_b, 4),
            "top_3_hit_rate": round(top3_b, 4),
            "mrr": round(mrr_b, 4),
            "notes": "Real bank-to-bank transfers without ATM nodes. Evaluates flow termination ranking."
        }
    ])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_multi_term.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Saved multi-dataset terminal ranking comparison to: {OUTPUT_CSV}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
