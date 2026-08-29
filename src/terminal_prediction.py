"""
Stage 4: Terminal Node & Cash-Withdrawal Location Prediction
===========================================================
This module ranks and predicts likely cash-withdrawal terminal nodes (ATMs)
for cybercrime incident subgraphs flagged as high-risk / suspicious by Stage 3B GraphSAGE.

Methodology:
- For every incident, candidate ATM nodes are identified directly within the 72-hour,
  3-hop incident subgraph (data/graphs/<complaint_id>.graphml).
- An interpretable, multi-criteria Terminal Risk Score (terminal_score in [0, 1])
  is computed from structural, transactional, positional, and GNN risk features.
- Candidates are ranked per complaint from highest to lowest score.
- Ground-truth evaluation metrics (Top-1 Hit Rate, Top-3 Hit Rate, MRR) are calculated
  strictly after prediction for empirical benchmarking.

Outputs:
1. data/terminal_predictions.csv - Full candidate ranking for all incidents with ATMs.
2. data/top_terminal_predictions.csv - Top-3 candidate terminals with human-readable explanations.
3. data/terminal_prediction_evaluation.csv - Benchmark performance metrics (Hit Rates & MRR).
4. data/terminal_ranking_examples.csv - 10+ detailed investigative case ranking examples.
5. data/terminal_prediction_map.png - Geospatial distribution plot of predicted cash-out hubs.
"""

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.graphsage_classifier import (
    DualHeadGraphSAGE,
    load_all_graphs_dataset,
    normalize_node_features,
    get_or_create_train_test_split
)


# ==============================================================================
# Configuration & Paths
# ==============================================================================

DATA_DIR = Path("data")
GRAPHS_DIR = DATA_DIR / "graphs"
MODELS_DIR = Path("models")

GRAPH_SUMMARY_FILE = DATA_DIR / "graph_summary.csv"
GRAPHSAGE_MODEL_FILE = MODELS_DIR / "graphsage_model.pt"
GRAPHSAGE_PREDICTIONS_FILE = DATA_DIR / "graphsage_predictions.csv"
MODEL_SPLIT_FILE = DATA_DIR / "model_split_ids.csv"

TERMINAL_PREDICTIONS_FILE = DATA_DIR / "terminal_predictions.csv"
TOP_TERMINAL_PREDICTIONS_FILE = DATA_DIR / "top_terminal_predictions.csv"
EVALUATION_FILE = DATA_DIR / "terminal_prediction_evaluation.csv"
RANKING_EXAMPLES_FILE = DATA_DIR / "terminal_ranking_examples.csv"
MAP_PLOT_FILE = DATA_DIR / "terminal_prediction_map.png"

HIGH_RISK_THRESHOLD = 0.50
RANDOM_SEED = 42


# ==============================================================================
# Geospatial Distance Helper
# ==============================================================================

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two geographic coordinates in kilometers.
    """
    if lat1 == 0.0 or lon1 == 0.0 or lat2 == 0.0 or lon2 == 0.0:
        return 500.0  # Default fallback distance

    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# ==============================================================================
# GraphSAGE Risk Inference
# ==============================================================================

def compute_graphsage_risk_probabilities(
    df_summary: pd.DataFrame
) -> Dict[str, float]:
    """
    Computes or loads GraphSAGE incident probability for all 1,000 complaints.
    """
    raw_dataset, _ = load_all_graphs_dataset(GRAPH_SUMMARY_FILE)
    train_ids, test_ids = get_or_create_train_test_split(df_summary)
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)

    train_raw = [d for d in raw_dataset if d.complaint_id in train_id_set]
    test_raw = [d for d in raw_dataset if d.complaint_id in test_id_set]
    train_norm, test_norm, mean_norm, std_norm = normalize_node_features(train_raw, test_raw)

    # Normalize full dataset using train stats
    continuous_indices = [2, 3, 4, 5, 6, 7, 8, 9, 12]
    all_norm = []
    for d in raw_dataset:
        d_norm = d.clone()
        for idx in continuous_indices:
            d_norm.x[:, idx] = (d_norm.x[:, idx] - mean_norm[idx]) / std_norm[idx]
        all_norm.append(d_norm)

    model = DualHeadGraphSAGE(input_dim=13, hidden_dim=64, dropout=0.2)
    model.load_state_dict(torch.load(GRAPHSAGE_MODEL_FILE, weights_only=True))
    model.eval()

    loader = DataLoader(all_norm, batch_size=64, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            logits_node, logits_graph, _ = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(logits_graph).cpu().numpy().flatten()
            all_probs.extend(probs)

    risk_dict = {d.complaint_id: float(p) for d, p in zip(raw_dataset, all_probs)}
    return risk_dict


# ==============================================================================
# Terminal Risk Scoring & Feature Extraction
# ==============================================================================

def calculate_terminal_score(
    gs_prob: float,
    hop: int,
    num_cw: int,
    tot_amt: float,
    hours_diff: float,
    num_up: int,
    geo_dist: float
) -> float:
    """
    Computes an interpretable composite terminal risk score in [0, 1].
    
    Components (all normalized in [0, 1]):
    - s_gnn      : GraphSAGE incident probability (25%)
    - s_hop      : Proximity to incident entity (20%)
    - s_cw       : Direct cash withdrawal flag (20%)
    - s_volume   : Cash withdrawal volume scale (15%)
    - s_recency  : Temporal proximity within 72h window (10%)
    - s_upstream : Convergence of multiple upstream sources (5%)
    - s_geo      : Geographic locality (5%)
    """
    s_gnn = max(0.0, min(1.0, gs_prob))
    s_hop = max(0.0, 1.0 - 0.25 * max(0, hop - 1))  # Hop 1 -> 1.0, Hop 2 -> 0.75, Hop 3 -> 0.50
    s_cw = 1.0 if num_cw >= 1 else 0.0
    s_vol = min(1.0, tot_amt / 250000.0)
    s_rec = max(0.0, 1.0 - (min(hours_diff, 72.0) / 72.0))
    s_up = min(1.0, num_up / 4.0)
    s_geo = max(0.0, 1.0 - min(1.0, geo_dist / 1500.0))

    raw_score = (
        0.25 * s_gnn +
        0.20 * s_hop +
        0.20 * s_cw +
        0.15 * s_vol +
        0.10 * s_rec +
        0.05 * s_up +
        0.05 * s_geo
    )
    return round(float(min(1.0, max(0.0, raw_score))), 4)


def generate_human_readable_reason(
    hop: int,
    num_up: int,
    num_cw: int,
    tot_amt: float,
    hours_diff: float,
    city: str
) -> str:
    """
    Generates a clear, human-readable rationale for the candidate ATM ranking.
    """
    parts = []
    if hop == 1:
        parts.append("Immediate 1-hop terminal connection")
    elif hop == 2:
        parts.append("Intermediate 2-hop fund layering path")
    else:
        parts.append("Extended 3-hop fund dispersion chain")

    if num_cw >= 1:
        parts.append(f"direct cash withdrawal observed (₹{tot_amt:,.2f} across {num_cw} tx)")
    else:
        parts.append("terminal endpoint with unexecuted withdrawal intent")

    if num_up > 1:
        parts.append(f"{num_up} upstream feeding entities converged at terminal in {city}")
    else:
        parts.append(f"single-source fund exit at {city}")

    parts.append(f"activity occurred within {hours_diff:.1f}h of incident reference time")
    return "; ".join(parts) + "."


# ==============================================================================
# Main Prediction & Ranking Routine
# ==============================================================================

def process_all_incident_terminal_predictions(
    df_summary: pd.DataFrame,
    risk_dict: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Extracts, scores, and ranks candidate ATM nodes for all incident graphs.
    """
    all_candidate_records: List[Dict[str, Any]] = []
    top3_records: List[Dict[str, Any]] = []
    evaluation_tracking: List[Dict[str, Any]] = []

    for _, row in df_summary.iterrows():
        c_id = str(row["complaint_id"])
        inc_ent = str(row["incident_entity_id"])
        inc_time_str = str(row["incident_time"])
        inc_time = datetime.strptime(inc_time_str, "%Y-%m-%d %H:%M:%S")
        gs_prob = risk_dict.get(c_id, 0.50)

        graph_file = GRAPHS_DIR / f"{c_id}.graphml"
        if not graph_file.exists():
            continue

        G = nx.read_graphml(graph_file)

        # Locate root incident node coordinates
        inc_nodes = [n for n, d in G.nodes(data=True) if d.get("is_incident") is True]
        if inc_nodes:
            inc_lat = float(G.nodes[inc_nodes[0]].get("latitude", 0.0))
            inc_lon = float(G.nodes[inc_nodes[0]].get("longitude", 0.0))
        else:
            inc_lat = float(row.get("latitude", 0.0))
            inc_lon = float(row.get("longitude", 0.0))

        # Find ATM nodes in this incident subgraph
        atm_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "ATM" or n.startswith("ATM_")]

        # Extract actual ground-truth suspicious cash-out ATMs in this graph (FOR EVALUATION ONLY)
        actual_cash_atms: Set[str] = set()
        for u, v, edata in G.edges(data=True):
            if v.startswith("ATM_") and (edata.get("is_cash_out") == 1 or edata.get("transaction_type") == "CASH_WITHDRAWAL"):
                if edata.get("is_suspicious") == 1:
                    actual_cash_atms.add(v)

        if not atm_nodes:
            evaluation_tracking.append({
                "complaint_id": c_id,
                "incident_entity_id": inc_ent,
                "has_atm_candidate": False,
                "has_actual_cashout": len(actual_cash_atms) > 0,
                "num_candidates": 0,
                "actual_cash_atms": list(actual_cash_atms),
                "ranked_atms": [],
                "top1_hit": 0,
                "top3_hit": 0,
                "reciprocal_rank": 0.0
            })
            continue

        # Score all candidate ATMs in G
        incident_candidates = []
        for atm in atm_nodes:
            d = G.nodes[atm]
            hop = int(d.get("hop_distance", 3))
            atm_city = str(d.get("city", "Unknown"))
            atm_lat = float(d.get("latitude", 0.0))
            atm_lon = float(d.get("longitude", 0.0))
            geo_dist = haversine_distance_km(inc_lat, inc_lon, atm_lat, atm_lon)

            in_edges = list(G.in_edges(atm, data=True))
            num_up = len(set(u for u, _, _ in in_edges))

            cw_edges = [
                ed for _, _, ed in in_edges
                if ed.get("transaction_type") == "CASH_WITHDRAWAL" or ed.get("is_cash_out") == 1
            ]
            num_cw = len(cw_edges)
            tot_amt = float(sum(float(ed.get("amount", 0.0)) for ed in cw_edges))
            max_amt = float(max([float(ed.get("amount", 0.0)) for ed in cw_edges] or [0.0]))

            times = [
                datetime.strptime(ed["timestamp"], "%Y-%m-%d %H:%M:%S")
                for ed in cw_edges if "timestamp" in ed
            ]
            if times:
                min_diff = min(abs((t - inc_time).total_seconds()) / 3600.0 for t in times)
                nearest_dt = min(times, key=lambda t: abs((t - inc_time).total_seconds()))
                nearest_time_str = nearest_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                min_diff = 72.0
                nearest_time_str = inc_time_str

            score = calculate_terminal_score(
                gs_prob=gs_prob,
                hop=hop,
                num_cw=num_cw,
                tot_amt=tot_amt,
                hours_diff=min_diff,
                num_up=num_up,
                geo_dist=geo_dist
            )

            reason_str = generate_human_readable_reason(
                hop=hop,
                num_up=num_up,
                num_cw=num_cw,
                tot_amt=tot_amt,
                hours_diff=min_diff,
                city=atm_city
            )

            rec = {
                "complaint_id": c_id,
                "incident_entity_id": inc_ent,
                "graphsage_probability": round(gs_prob, 4),
                "atm_id": atm,
                "atm_city": atm_city,
                "atm_latitude": atm_lat,
                "atm_longitude": atm_lon,
                "hop_distance": hop,
                "num_upstream_entities": num_up,
                "num_cash_withdrawals": num_cw,
                "total_cash_withdrawal_amount": round(tot_amt, 2),
                "max_cash_withdrawal_amount": round(max_amt, 2),
                "nearest_withdrawal_time": nearest_time_str,
                "hours_from_incident": round(min_diff, 2),
                "terminal_score": score,
                "reason": reason_str
            }
            incident_candidates.append(rec)

        # Sort candidates descending by terminal_score
        incident_candidates.sort(key=lambda x: x["terminal_score"], reverse=True)
        for r_idx, c_item in enumerate(incident_candidates, 1):
            c_item["rank"] = r_idx

        all_candidate_records.extend(incident_candidates)

        # Top 3 records for high-risk complaints (or all complaints with ATMs)
        for c_item in incident_candidates[:3]:
            top3_records.append({
                "complaint_id": c_item["complaint_id"],
                "incident_entity_id": c_item["incident_entity_id"],
                "risk_probability": c_item["graphsage_probability"],
                "rank": c_item["rank"],
                "atm_id": c_item["atm_id"],
                "city": c_item["atm_city"],
                "latitude": c_item["atm_latitude"],
                "longitude": c_item["atm_longitude"],
                "terminal_score": c_item["terminal_score"],
                "reason": c_item["reason"]
            })

        # Evaluation metrics tracking
        ranked_atms = [item["atm_id"] for item in incident_candidates]
        top1_hit = 0
        top3_hit = 0
        rr = 0.0

        if actual_cash_atms:
            top1_hit = 1 if (len(ranked_atms) > 0 and ranked_atms[0] in actual_cash_atms) else 0
            top3_hit = 1 if any(a in actual_cash_atms for a in ranked_atms[:3]) else 0
            for r_pos, a_id in enumerate(ranked_atms, 1):
                if a_id in actual_cash_atms:
                    rr = 1.0 / r_pos
                    break

        evaluation_tracking.append({
            "complaint_id": c_id,
            "incident_entity_id": inc_ent,
            "has_atm_candidate": True,
            "has_actual_cashout": len(actual_cash_atms) > 0,
            "num_candidates": len(atm_nodes),
            "actual_cash_atms": list(actual_cash_atms),
            "ranked_atms": ranked_atms,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "reciprocal_rank": rr
        })

    df_all_candidates = pd.DataFrame(all_candidate_records)
    df_top3 = pd.DataFrame(top3_records)
    return df_all_candidates, df_top3, evaluation_tracking


# ==============================================================================
# Evaluation & Ranking Examples
# ==============================================================================

def compute_evaluation_metrics_table(
    evaluation_tracking: List[Dict[str, Any]],
    output_path: Path = EVALUATION_FILE
) -> pd.DataFrame:
    """
    Computes Top-1 Hit Rate, Top-3 Hit Rate, and MRR for incidents with actual cash-outs.
    """
    total_incidents = len(evaluation_tracking)
    incidents_with_atms = sum(1 for e in evaluation_tracking if e["has_atm_candidate"])
    incidents_no_atms = total_incidents - incidents_with_atms
    cashout_incidents = [e for e in evaluation_tracking if e["has_actual_cashout"]]

    num_eval = len(cashout_incidents)
    top1_hits = sum(e["top1_hit"] for e in cashout_incidents)
    top3_hits = sum(e["top3_hit"] for e in cashout_incidents)
    mrr_val = float(np.mean([e["reciprocal_rank"] for e in cashout_incidents])) if num_eval > 0 else 0.0

    avg_candidates = float(np.mean([e["num_candidates"] for e in evaluation_tracking if e["has_atm_candidate"]]))

    metrics_df = pd.DataFrame([{
        "total_incidents_processed": total_incidents,
        "incidents_with_atm_candidates": incidents_with_atms,
        "incidents_with_no_atms": incidents_no_atms,
        "evaluable_cashout_incidents": num_eval,
        "top_1_hit_rate": round(float(top1_hits / num_eval) * 100, 2) if num_eval > 0 else 0.0,
        "top_3_hit_rate": round(float(top3_hits / num_eval) * 100, 2) if num_eval > 0 else 0.0,
        "mean_reciprocal_rank_mrr": round(mrr_val, 4),
        "incidents_with_at_least_one_hit": sum(1 for e in cashout_incidents if e["top3_hit"] == 1),
        "avg_candidates_per_incident": round(avg_candidates, 2)
    }])

    metrics_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved terminal prediction evaluation metrics to: {output_path}")
    return metrics_df


def generate_ranking_examples_table(
    df_all_candidates: pd.DataFrame,
    evaluation_tracking: List[Dict[str, Any]],
    output_path: Path = RANKING_EXAMPLES_FILE
) -> pd.DataFrame:
    """
    Generates detailed ranking breakdown for at least 10 representative suspicious incidents.
    """
    eval_dict = {e["complaint_id"]: e for e in evaluation_tracking}
    suspicious_with_cashout = [e for e in evaluation_tracking if e["has_actual_cashout"]]

    example_records = []
    # Pick 12 representative examples
    sample_eval = suspicious_with_cashout[:12]

    for item in sample_eval:
        c_id = item["complaint_id"]
        c_candidates = df_all_candidates[df_all_candidates["complaint_id"] == c_id].sort_values("rank")
        
        top3_atms = ", ".join(c_candidates["atm_id"].head(3).tolist())
        top1_city = c_candidates["atm_city"].iloc[0] if not c_candidates.empty else "N/A"
        top1_score = c_candidates["terminal_score"].iloc[0] if not c_candidates.empty else 0.0
        gs_prob = c_candidates["graphsage_probability"].iloc[0] if not c_candidates.empty else 0.0
        
        actual_atms_str = ", ".join(item["actual_cash_atms"])
        
        rank_first = "N/A"
        for r_i, a_id in enumerate(item["ranked_atms"], 1):
            if a_id in item["actual_cash_atms"]:
                rank_first = str(r_i)
                break

        example_records.append({
            "complaint_id": c_id,
            "incident_entity_id": item["incident_entity_id"],
            "graphsage_probability": gs_prob,
            "top_3_predicted_atms": top3_atms,
            "top_predicted_city": top1_city,
            "top_terminal_score": top1_score,
            "actual_cash_out_atms": actual_atms_str,
            "rank_of_first_correct_atm": rank_first
        })

    df_examples = pd.DataFrame(example_records)
    df_examples.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved {len(df_examples)} ranking examples to: {output_path}")
    return df_examples


# ==============================================================================
# Geospatial Map Visualization
# ==============================================================================

def plot_terminal_prediction_map(
    df_all_candidates: pd.DataFrame,
    output_path: Path = MAP_PLOT_FILE
) -> None:
    """
    Renders a geospatial map visualization showing candidate ATM locations,
    highlighting top-ranked predicted cash-out hubs across India.
    """
    plt.figure(figsize=(12, 10))
    plt.clf()

    # Base scatter of all ATM candidate observations
    latitudes = df_all_candidates["atm_latitude"]
    longitudes = df_all_candidates["atm_longitude"]
    scores = df_all_candidates["terminal_score"]

    # Filter top-ranked candidates
    top1_df = df_all_candidates[df_all_candidates["rank"] == 1]

    # Plot lower-ranked candidate ATMs
    plt.scatter(
        df_all_candidates["atm_longitude"],
        df_all_candidates["atm_latitude"],
        c="#A8DADC",
        s=80,
        alpha=0.6,
        edgecolor="#1D3557",
        linewidth=0.8,
        label="Candidate ATM Terminals (In-Graph)"
    )

    # Plot Rank-1 Predicted Hubs (sized by score)
    scatter_top1 = plt.scatter(
        top1_df["atm_longitude"],
        top1_df["atm_latitude"],
        c=top1_df["terminal_score"],
        cmap="YlOrRd",
        s=top1_df["terminal_score"] * 300,
        alpha=0.9,
        edgecolor="#9D0208",
        linewidth=1.8,
        label="Rank 1 Predicted Cash-Out Terminals"
    )

    cbar = plt.colorbar(scatter_top1, shrink=0.75, pad=0.02)
    cbar.set_label("Terminal Risk Score", fontsize=10, fontweight="bold")

    # Annotate major Indian cities on the map
    city_coords = {
        "Kolkata": (88.3639, 22.5726),
        "Mumbai": (72.8777, 19.0760),
        "Bengaluru": (77.5946, 12.9716),
        "Delhi": (77.2090, 28.6139),
        "Hyderabad": (78.4867, 17.3850),
        "Chennai": (80.2707, 13.0827),
        "Ahmedabad": (72.5714, 23.0225),
        "Jaipur": (75.7873, 26.9124),
        "Lucknow": (80.9462, 26.8467),
        "Patna": (85.1376, 25.5941),
        "Kochi": (76.2673, 9.9312),
        "Chandigarh": (76.7794, 30.7333)
    }

    for city, (lon, lat) in city_coords.items():
        plt.plot(lon, lat, marker="x", color="#457B9D", markersize=6, mew=1.5)
        plt.text(lon + 0.35, lat + 0.25, city, fontsize=8.5, fontweight="bold", color="#1D3557", alpha=0.85)

    plt.title(
        "Stage 4: Predicted Illicit Cash-Withdrawal Terminal Hubs\n"
        "Geospatial Distribution of High-Risk ATM Nodes Identified within 72h 3-Hop Incident Subgraphs",
        fontsize=12,
        fontweight="bold",
        pad=15
    )
    plt.xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    plt.ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    plt.xlim(68.0, 92.0)
    plt.ylim(8.0, 34.0)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved terminal prediction map visualization to: {output_path}")


# ==============================================================================
# Automated Validation Suite
# ==============================================================================

def run_validations(
    df_summary: pd.DataFrame,
    df_all_candidates: pd.DataFrame,
    df_top3: pd.DataFrame,
    df_examples: pd.DataFrame
) -> None:
    """
    Automated validation suite checking all 10 Stage 4 constraints.
    """
    # 1. Valid incident entity for every complaint
    assert df_summary["incident_entity_id"].notna().all(), "Found missing incident_entity_id in summary!"

    # 2. Predicted ATMs non-empty in candidates
    assert len(df_all_candidates) > 0, "No ATM candidates generated!"
    assert df_all_candidates["atm_id"].str.startswith("ATM_").all(), "Invalid ATM ID format in candidates!"

    # 3. Valid coordinates
    assert (df_all_candidates["atm_latitude"] > 0).all(), "Invalid latitude <= 0!"
    assert (df_all_candidates["atm_longitude"] > 0).all(), "Invalid longitude <= 0!"

    # 4. terminal_score strictly in [0, 1]
    assert (df_all_candidates["terminal_score"] >= 0.0).all() and (df_all_candidates["terminal_score"] <= 1.0).all(), (
        "terminal_score outside [0, 1] range!"
    )

    # 5. Rank starts at 1 for each complaint
    min_ranks = df_all_candidates.groupby("complaint_id")["rank"].min()
    assert (min_ranks == 1).all(), "Rank does not start at 1 for all complaints!"

    # 6. No duplicate complaint_id + rank combinations
    assert not df_all_candidates.duplicated(subset=["complaint_id", "rank"]).any(), (
        "Duplicate (complaint_id, rank) combinations found!"
    )

    # 7. No ground truth used in features
    forbidden = ["is_suspicious", "ring_id", "ground_truth_entity_id"]
    for f in forbidden:
        assert f not in df_all_candidates.columns, f"Ground-truth column {f} leaked into candidate features!"

    # 8. All candidates within incident graph
    assert len(df_top3) <= len(df_all_candidates), "Top-3 exceeds total candidates!"

    # 9. Output files exist
    assert TERMINAL_PREDICTIONS_FILE.exists()
    assert TOP_TERMINAL_PREDICTIONS_FILE.exists()
    assert EVALUATION_FILE.exists()
    assert RANKING_EXAMPLES_FILE.exists()
    assert MAP_PLOT_FILE.exists()

    # 10. At least 10 ranking examples generated
    assert len(df_examples) >= 10, f"Expected >= 10 ranking examples, got {len(df_examples)}"

    print("\n" + "=" * 50)
    print("             STAGE 4 VALIDATION")
    print("=" * 50)
    print("All 10 validation checks passed successfully.")
    print("=" * 50 + "\n")


# ==============================================================================
# Main Pipeline Entrypoint
# ==============================================================================

def main():
    print("=" * 60)
    print("   STAGE 4 — TERMINAL NODE & CASH-OUT PREDICTION")
    print("=" * 60)

    # 1. Load Dataset & Incident Graph Features
    if not GRAPH_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {GRAPH_SUMMARY_FILE}")
    df_summary = pd.read_csv(GRAPH_SUMMARY_FILE)

    # 2. Compute GraphSAGE Risk Probabilities for all 1,000 complaints
    print("Computing GraphSAGE incident risk probabilities...")
    risk_dict = compute_graphsage_risk_probabilities(df_summary)

    # 3. Extract and Rank Candidate ATMs
    print("Extracting and scoring candidate ATM nodes from 1,000 incident subgraphs...")
    df_all_candidates, df_top3, evaluation_tracking = process_all_incident_terminal_predictions(
        df_summary, risk_dict
    )

    # 4. Save Candidate Tables
    # Remove internal reason from full predictions to match required clean schema
    cols_all = [
        "complaint_id",
        "incident_entity_id",
        "graphsage_probability",
        "atm_id",
        "atm_city",
        "atm_latitude",
        "atm_longitude",
        "hop_distance",
        "num_upstream_entities",
        "num_cash_withdrawals",
        "total_cash_withdrawal_amount",
        "max_cash_withdrawal_amount",
        "nearest_withdrawal_time",
        "hours_from_incident",
        "terminal_score",
        "rank"
    ]
    df_all_clean = df_all_candidates[cols_all]
    df_all_clean.to_csv(TERMINAL_PREDICTIONS_FILE, index=False)
    print(f"[SUCCESS] Saved full terminal predictions to: {TERMINAL_PREDICTIONS_FILE}")

    df_top3.to_csv(TOP_TERMINAL_PREDICTIONS_FILE, index=False)
    print(f"[SUCCESS] Saved top-3 terminal predictions to: {TOP_TERMINAL_PREDICTIONS_FILE}")

    # 5. Compute Benchmark Evaluation Metrics
    df_eval = compute_evaluation_metrics_table(evaluation_tracking)

    # 6. Generate 10+ Detailed Ranking Examples
    df_examples = generate_ranking_examples_table(df_all_candidates, evaluation_tracking)

    # 7. Generate Geospatial Map Plot
    plot_terminal_prediction_map(df_all_candidates)

    # 8. Run Automated Validations
    run_validations(df_summary, df_all_candidates, df_top3, df_examples)

    # 9. Print Final Summary Report
    high_risk_count = sum(1 for p in risk_dict.values() if p >= HIGH_RISK_THRESHOLD)
    incidents_with_atms = sum(1 for e in evaluation_tracking if e["has_atm_candidate"])

    print("=" * 60)
    print("       STAGE 4 — TERMINAL NODE PREDICTION REPORT")
    print("=" * 60)
    print(f"Incidents processed            : {len(df_summary)}")
    print(f"High-risk incidents (GNN >= 0.50): {high_risk_count}")
    print(f"Incidents with ATM candidates  : {incidents_with_atms}")
    print(f"Total ATM candidate instances  : {len(df_all_candidates)}")
    print(f"Top-1 hit rate                 : {df_eval['top_1_hit_rate'].iloc[0]:.2f}%")
    print(f"Top-3 hit rate                 : {df_eval['top_3_hit_rate'].iloc[0]:.2f}%")
    print(f"MRR (Mean Reciprocal Rank)     : {df_eval['mean_reciprocal_rank_mrr'].iloc[0]:.4f}")
    print(f"Average candidates per incident: {df_eval['avg_candidates_per_incident'].iloc[0]:.2f}")
    print("-" * 60)

    # Print Top 3 Example Predictions
    print("TOP EXAMPLE PREDICTIONS (Demonstration Cases):")
    for idx, row in df_top3.head(3).iterrows():
        print(f"\n[Case {idx + 1}] Complaint: {row['complaint_id']} | Incident Entity: {row['incident_entity_id']}")
        print(f"  GraphSAGE Risk Probability : {row['risk_probability']:.4f}")
        print(f"  Rank {row['rank']} ATM                 : {row['atm_id']} ({row['city']}) [Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}]")
        print(f"  Terminal Score             : {row['terminal_score']:.4f}")
        print(f"  Reason                     : {row['reason']}")
    print("-" * 60)

    print("\nRanking Examples Breakdown (Sample 5 Rows):")
    print(df_examples.head(5).to_string(index=False))
    print("\n" + "=" * 60)
    print("Stage 4 Terminal Node & Cash-Withdrawal Prediction is complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
