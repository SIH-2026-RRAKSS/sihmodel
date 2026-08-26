"""
Stage 2 (IBM AML): Incident Subgraph Construction for Bank Ledger Data
======================================================================
Constructs 1,000 directed transaction subgraphs (200 Laundering / 800 Normal)
from the IBM AML dataset (HI-Small_Trans.csv) matching the scale and evaluation
protocol of Dataset A.

Schema & Guardrail Compliance:
- Subgraphs: 72-hour temporal window, <= 3 hops around seed accounts.
- Ground truth: is_laundering = 1 if subgraph contains any laundering transaction.
- Output: data/ibm_graphs/<subgraph_id>.graphml and data/ibm_graph_summary.csv.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adapters.ibm_adapter import IBMAMLAdapter

DATA_DIR = Path("data")
IBM_GRAPHS_DIR = DATA_DIR / "ibm_graphs"
IBM_SUMMARY_FILE = DATA_DIR / "ibm_graph_summary.csv"

def build_ibm_subgraphs(n_pos=200, n_neg=800, seed=42):
    print("=" * 70)
    print("   STAGE 2 ? IBM AML SUBGRAPH EXTRACTION (N=1,000)")
    print("=" * 70)
    
    np.random.seed(seed)
    IBM_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    ibm = IBMAMLAdapter()
    print("Loading raw IBM transactions...")
    df_raw = ibm.load_raw_transactions(nrows=1000000)
    
    df_tx = pd.DataFrame()
    df_tx["from_acc"] = "B" + df_raw["From Bank"].astype(str) + "_" + df_raw["Account"].astype(str)
    df_tx["to_acc"] = "B" + df_raw["To Bank"].astype(str) + "_" + df_raw["Account.1"].astype(str)
    df_tx["amount"] = df_raw["Amount Paid"].astype(float)
    df_tx["format"] = df_raw["Payment Format"].astype(str)
    df_tx["timestamp"] = pd.to_datetime(df_raw["Timestamp"])
    df_tx["is_laundering"] = df_raw["Is Laundering"].astype(int)
    
    # Positive seeds (accounts with laundering transactions)
    laundering_tx = df_tx[df_tx["is_laundering"] == 1]
    pos_seed_candidates = laundering_tx["from_acc"].unique()
    np.random.shuffle(pos_seed_candidates)
    selected_pos_seeds = pos_seed_candidates[:n_pos]
    
    # Negative seeds (accounts with zero laundering transactions in their neighborhood)
    all_laundering_accounts = set(laundering_tx["from_acc"]).union(set(laundering_tx["to_acc"]))
    clean_tx = df_tx[~df_tx["from_acc"].isin(all_laundering_accounts) & ~df_tx["to_acc"].isin(all_laundering_accounts)]
    neg_seed_candidates = clean_tx["from_acc"].unique()
    np.random.shuffle(neg_seed_candidates)
    selected_neg_seeds = neg_seed_candidates[:n_neg]
    
    print(f"Selected {len(selected_pos_seeds)} positive seeds and {len(selected_neg_seeds)} negative seeds.")
    
    # Build fast adjacency lookups
    # Group transactions by sender and receiver
    print("Indexing transaction ledger for temporal k-hop extraction...")
    tx_by_sender = df_tx.groupby("from_acc")
    tx_by_receiver = df_tx.groupby("to_acc")
    
    all_records = []
    
    # Extraction helper
    def extract_subgraph_for_seed(seed_acc, is_pos_target, sub_id):
        # Find anchor timestamp
        if is_pos_target:
            seed_laundering = df_tx[(df_tx["from_acc"] == seed_acc) & (df_tx["is_laundering"] == 1)]
            if not seed_laundering.empty:
                t0 = seed_laundering["timestamp"].iloc[0]
            else:
                t0 = df_tx[df_tx["from_acc"] == seed_acc]["timestamp"].iloc[0]
        else:
            t0 = df_tx[df_tx["from_acc"] == seed_acc]["timestamp"].iloc[0]
            
        t_start = t0 - timedelta(hours=36)
        t_end = t0 + timedelta(hours=36)
        
        # BFS up to 3 hops within [t_start, t_end]
        visited_nodes = {seed_acc}
        frontier = {seed_acc}
        subgraph_edges = []
        
        for hop in range(1, 4):
            next_frontier = set()
            for u in frontier:
                # Outgoing
                if u in tx_by_sender.groups:
                    out_df = tx_by_sender.get_group(u)
                    out_window = out_df[(out_df["timestamp"] >= t_start) & (out_df["timestamp"] <= t_end)]
                    for _, r in out_window.iterrows():
                        v = r["to_acc"]
                        subgraph_edges.append((u, v, r))
                        if v not in visited_nodes:
                            next_frontier.add(v)
                # Incoming
                if u in tx_by_receiver.groups:
                    in_df = tx_by_receiver.get_group(u)
                    in_window = in_df[(in_df["timestamp"] >= t_start) & (in_df["timestamp"] <= t_end)]
                    for _, r in in_window.iterrows():
                        v = r["from_acc"]
                        subgraph_edges.append((v, u, r))
                        if v not in visited_nodes:
                            next_frontier.add(v)
            visited_nodes.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
                
        # Build NetworkX MultiDiGraph
        G = nx.MultiDiGraph(subgraph_id=sub_id, seed_account=seed_acc)
        total_vol = 0.0
        max_vol = 0.0
        is_launder_flag = 0
        
        for u, v, r in subgraph_edges:
            amt = float(r["amount"])
            total_vol += amt
            max_vol = max(max_vol, amt)
            if int(r["is_laundering"]) == 1:
                is_launder_flag = 1
            G.add_edge(
                u, v,
                amount=amt,
                payment_format=str(r["format"]),
                timestamp=str(r["timestamp"]),
                is_laundering=int(r["is_laundering"])
            )
            
        # If no edges, add single isolate seed node
        if G.number_of_nodes() == 0:
            G.add_node(seed_acc)
            
        # Node features
        for node in G.nodes():
            in_d = G.in_degree(node)
            out_d = G.out_degree(node)
            G.nodes[node]["in_degree"] = in_d
            G.nodes[node]["out_degree"] = out_d
            G.nodes[node]["is_seed"] = int(node == seed_acc)
            G.nodes[node]["is_terminal_sink"] = int(out_d == 0 and in_d > 0)
            
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        in_deg_seed = G.in_degree(seed_acc) if seed_acc in G else 0
        out_deg_seed = G.out_degree(seed_acc) if seed_acc in G else 0
        fan_out = float(out_deg_seed / max(1, in_deg_seed))
        density = float(n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
        avg_deg = float(2 * n_edges / n_nodes) if n_nodes > 0 else 0.0
        num_sinks = sum(1 for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0)
        
        # Velocity
        velocity_tph = float(n_edges / 72.0)
        velocity_vph = float(total_vol / 72.0)
        
        # Save GraphML
        graph_path = IBM_GRAPHS_DIR / f"{sub_id}.graphml"
        nx.write_graphml(G, str(graph_path))
        
        return {
            "subgraph_id": sub_id,
            "seed_account": seed_acc,
            "contains_laundering": is_launder_flag,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "in_degree_seed": in_deg_seed,
            "out_degree_seed": out_deg_seed,
            "fan_out_ratio": round(fan_out, 4),
            "density": round(density, 4),
            "average_degree": round(avg_deg, 4),
            "num_terminal_sinks": num_sinks,
            "total_transaction_value": round(total_vol, 2),
            "max_transaction_value": round(max_vol, 2),
            "velocity_tph": round(velocity_tph, 4),
            "velocity_vph": round(velocity_vph, 2),
            "graphml_path": str(graph_path)
        }

    print("Extracting positive (laundering) subgraphs...")
    for idx, s in enumerate(selected_pos_seeds):
        sub_id = f"IBM_POS_{idx+1:04d}"
        rec = extract_subgraph_for_seed(s, is_pos_target=True, sub_id=sub_id)
        all_records.append(rec)
        
    print("Extracting negative (normal) subgraphs...")
    for idx, s in enumerate(selected_neg_seeds):
        sub_id = f"IBM_NEG_{idx+1:04d}"
        rec = extract_subgraph_for_seed(s, is_pos_target=False, sub_id=sub_id)
        all_records.append(rec)
        
    df_summary = pd.DataFrame(all_records)
    df_summary.to_csv(IBM_SUMMARY_FILE, index=False)
    
    print(f"\n[SUCCESS] Extracted {len(df_summary)} IBM subgraphs.")
    print(f"Summary saved to: {IBM_SUMMARY_FILE}")
    print(f"GraphML files stored in: {IBM_GRAPHS_DIR}")
    print("\nDataset Class Balance:")
    print(df_summary["contains_laundering"].value_counts())
    print("\nGraph Metrics Summary:")
    print(df_summary[["num_nodes", "num_edges", "total_transaction_value", "velocity_tph"]].describe().round(2))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    build_ibm_subgraphs()
