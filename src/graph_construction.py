"""
Stage 2: Incident Subgraph Extraction & Graph Construction
=========================================================
This module extracts 72-hour 3-hop directed incident subgraphs (networkx.MultiDiGraph)
around resolved cybercrime complaint entities for downstream GNN (GraphSAGE) and
XGBoost baseline classification.

Architecture & Pipeline:
complaint (complaints.csv)
   ↓
Stage 0 entity resolution (resolved_entities.csv)
   ↓
resolved incident entity (predicted_entity_id)
   ↓
72-hour transaction window (transactions.csv)
   ↓
3-hop neighborhood extraction (<= 3 hops from incident node)
   ↓
MultiDiGraph construction (ACCOUNT nodes, ATM terminal nodes, transaction edges)
   ↓
Storage in data/graphs/<complaint_id>.graphml + summary in data/graph_summary.csv

Important Principles:
- Ground-truth evaluation labels (is_suspicious, ring_id) are preserved as metadata
  for evaluation/debugging, but are NEVER used to construct or filter the graph.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt


# ==============================================================================
# Configuration & Paths
# ==============================================================================

DATA_DIR = Path("data")
COMPLAINTS_FILE = DATA_DIR / "complaints.csv"
RESOLVED_ENTITIES_FILE = DATA_DIR / "resolved_entities.csv"
ENTITY_MASTER_FILE = DATA_DIR / "entity_master.csv"
ENTITY_LOCATIONS_FILE = DATA_DIR / "entity_locations.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"

GRAPHS_DIR = DATA_DIR / "graphs"
GRAPH_SUMMARY_FILE = DATA_DIR / "graph_summary.csv"
DEMO_GRAPH_IMAGE_FILE = GRAPHS_DIR / "demo_graph.png"

DEFAULT_WINDOW_HOURS = 72
DEFAULT_MAX_HOPS = 3


# ==============================================================================
# Data Loading Routines
# ==============================================================================

def load_complaints(complaints_path: Path = COMPLAINTS_FILE) -> pd.DataFrame:
    """Loads input complaints dataset."""
    if not complaints_path.exists():
        raise FileNotFoundError(f"Missing complaints file: {complaints_path}")
    return pd.read_csv(complaints_path)


def load_resolved_entities(resolved_path: Path = RESOLVED_ENTITIES_FILE) -> pd.DataFrame:
    """Loads Stage 0 resolved entities mapping table."""
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing resolved entities file: {resolved_path}")
    return pd.read_csv(resolved_path)


def load_entity_master(master_path: Path = ENTITY_MASTER_FILE) -> pd.DataFrame:
    """Loads 700 master entity reference table."""
    if not master_path.exists():
        raise FileNotFoundError(f"Missing entity master file: {master_path}")
    return pd.read_csv(master_path)


def load_entity_locations(locations_path: Path = ENTITY_LOCATIONS_FILE) -> pd.DataFrame:
    """Loads geographic coordinate mapping for entities."""
    if not locations_path.exists():
        raise FileNotFoundError(f"Missing entity locations file: {locations_path}")
    return pd.read_csv(locations_path)


def load_transactions(transactions_path: Path = TRANSACTIONS_FILE) -> pd.DataFrame:
    """Loads 15,000 synthetic transaction dataset."""
    if not transactions_path.exists():
        raise FileNotFoundError(f"Missing transactions file: {transactions_path}")
    df_tx = pd.read_csv(transactions_path)
    df_tx["parsed_dt"] = pd.to_datetime(df_tx["timestamp"])
    return df_tx


def build_location_and_entity_lookups(
    df_master: pd.DataFrame,
    df_locations: pd.DataFrame,
    df_transactions: pd.DataFrame
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]], Set[str], Set[str]]:
    """
    Builds fast lookup tables for entity canonical names, locations, and valid IDs.
    """
    # Entity master name lookup
    entity_name_lookup = df_master.set_index("entity_id")["canonical_name"].to_dict()

    # Location lookup for ACCOUNT nodes
    location_lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in df_locations.iterrows():
        location_lookup[row["entity_id"]] = {
            "state": str(row["state"]),
            "city": str(row["city"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"])
        }

    # ATM Node Location Lookup from transaction receiver coordinates
    atm_transactions = df_transactions[df_transactions["receiver_entity_id"].str.startswith("ATM_")]
    atm_states = atm_transactions.groupby("receiver_entity_id")["receiver_state"].first().to_dict()

    # Known city coordinates for ATM states
    state_to_city_coords = {
        "West Bengal": ("Kolkata", 22.5726, 88.3639),
        "Maharashtra": ("Mumbai", 19.0760, 72.8777),
        "Karnataka": ("Bengaluru", 12.9716, 77.5946),
        "Delhi": ("Delhi", 28.6139, 77.2090),
        "Tamil Nadu": ("Chennai", 13.0827, 80.2707),
        "Telangana": ("Hyderabad", 17.3850, 78.4867),
        "Gujarat": ("Ahmedabad", 23.0225, 72.5714),
        "Rajasthan": ("Jaipur", 26.9124, 75.7873),
        "Uttar Pradesh": ("Lucknow", 26.8467, 80.9462),
        "Bihar": ("Patna", 25.5941, 85.1376),
        "Odisha": ("Bhubaneswar", 20.2961, 85.8245),
        "Kerala": ("Kochi", 9.9312, 76.2673),
        "Madhya Pradesh": ("Bhopal", 23.2599, 77.4126),
        "Punjab": ("Chandigarh", 30.7333, 76.7794)
    }

    valid_atms: Set[str] = set()
    for atm_id, state in atm_states.items():
        valid_atms.add(atm_id)
        city, lat, lon = state_to_city_coords.get(state, ("Unknown", 20.5937, 78.9629))
        location_lookup[atm_id] = {
            "state": state,
            "city": city,
            "latitude": lat,
            "longitude": lon
        }

    valid_entities = set(df_master["entity_id"])
    return entity_name_lookup, location_lookup, valid_entities, valid_atms


# ==============================================================================
# Subgraph Extraction & Graph Processing
# ==============================================================================

def filter_transactions_by_time(
    df_transactions: pd.DataFrame,
    window_start: datetime,
    window_end: datetime
) -> pd.DataFrame:
    """Filters transactions within [window_start, window_end] inclusive."""
    mask = (df_transactions["parsed_dt"] >= window_start) & (df_transactions["parsed_dt"] <= window_end)
    return df_transactions[mask]


def extract_incident_subgraph(
    df_window_tx: pd.DataFrame,
    incident_entity: str,
    incident_time: datetime,
    entity_name_lookup: Dict[str, str],
    location_lookup: Dict[str, Dict[str, Any]],
    max_hops: int = DEFAULT_MAX_HOPS
) -> nx.MultiDiGraph:
    """
    Constructs a directed MultiDiGraph for the 72-hour window and extracts the
    3-hop neighborhood around the incident entity.
    """
    subgraph = nx.MultiDiGraph()

    if df_window_tx.empty:
        # Isolated incident node graph
        _add_single_node(subgraph, incident_entity, True, 0, entity_name_lookup, location_lookup)
        return subgraph

    # 1. Build temporary MultiDiGraph for all transactions in the time window
    G_window = nx.MultiDiGraph()
    for _, tx in df_window_tx.iterrows():
        u = tx["sender_entity_id"]
        v = tx["receiver_entity_id"]
        tx_id = str(tx["transaction_id"])
        G_window.add_edge(
            u, v,
            key=tx_id,
            transaction_id=tx_id,
            amount=float(tx["amount"]),
            timestamp=str(tx["timestamp"]),
            transaction_type=str(tx["transaction_type"]),
            channel=str(tx["channel"]),
            is_cash_out=int(tx["is_cash_out"]),
            is_suspicious=int(tx["is_suspicious"]),
            ring_id=str(tx["ring_id"])
        )

    # 2. If incident entity did not transact in the window, return isolated node
    if incident_entity not in G_window:
        _add_single_node(subgraph, incident_entity, True, 0, entity_name_lookup, location_lookup)
        return subgraph

    # 3. Use Temporal BFS
    subgraph = extract_temporal_subgraph(
        global_graph=G_window,
        root_node=incident_entity,
        incident_timestamp=incident_time,
        max_hops=max_hops,
        max_degree_per_hop=15,
        window_hours=DEFAULT_WINDOW_HOURS
    )
    
    # 5. Populate comprehensive node attributes for extracted nodes
    for node in list(subgraph.nodes()):
        is_inc = (node == incident_entity)
        hop_dist = subgraph.nodes[node].get("hop_distance", 0)
        _set_node_attributes(subgraph, node, is_inc, hop_dist, entity_name_lookup, location_lookup)

    return subgraph


def extract_temporal_subgraph(
    global_graph: nx.MultiDiGraph,
    root_node: str,
    incident_timestamp: Any,
    max_hops: int = 3,
    max_degree_per_hop: int = 15,
    window_hours: int = 72
) -> nx.MultiDiGraph:
    """
    Extracts a temporal, downstream subgraph using BFS.
    Supports both datetime objects (synthetic data) and unix timestamps (IBM data).
    """
    from collections import deque
    
    # Calculate max_time based on type
    if isinstance(incident_timestamp, int) or isinstance(incident_timestamp, float):
        max_time = incident_timestamp + (window_hours * 3600)
    else:
        max_time = incident_timestamp + timedelta(hours=window_hours)
        
    queue = deque([(root_node, incident_timestamp, 0)])
    visited_nodes = {root_node: 0}
    
    while queue:
        curr_node, curr_time, curr_hop = queue.popleft()
        
        if curr_hop >= max_hops:
            continue
            
        out_edges = []
        for _, v, key, data in global_graph.out_edges(curr_node, data=True, keys=True):
            edge_time_raw = data.get("timestamp")
            if edge_time_raw is None:
                continue
                
            # Parse edge time dynamically
            if isinstance(edge_time_raw, str):
                edge_time = datetime.strptime(edge_time_raw, "%Y-%m-%d %H:%M:%S")
            else:
                edge_time = edge_time_raw
                
            if curr_time <= edge_time <= max_time:
                out_edges.append((v, key, data, edge_time))
                
        # Cap Fan-out by amount (if missing amount, defaults to 0)
        out_edges.sort(key=lambda e: float(e[2].get("amount", 0.0)), reverse=True)
        out_edges = out_edges[:max_degree_per_hop]
        
        for v, key, data, edge_time in out_edges:
            if v not in visited_nodes or curr_hop + 1 < visited_nodes[v]:
                visited_nodes[v] = curr_hop + 1
                queue.append((v, edge_time, curr_hop + 1))
                
    # Build the resulting subgraph
    nodes_in_neighborhood = set(visited_nodes.keys())
    subgraph = nx.MultiDiGraph()
    
    for node in nodes_in_neighborhood:
        hop_dist = visited_nodes[node]
        subgraph.add_node(node, hop_distance=hop_dist)
        
    for u in nodes_in_neighborhood:
        for _, v, key, data in global_graph.out_edges(u, data=True, keys=True):
            if v in nodes_in_neighborhood:
                subgraph.add_edge(u, v, key=key, **data)

    return subgraph


def _add_single_node(
    G: nx.MultiDiGraph,
    node_id: str,
    is_incident: bool,
    hop_dist: int,
    entity_name_lookup: Dict[str, str],
    location_lookup: Dict[str, Dict[str, Any]]
) -> None:
    """Helper to add an isolated node with complete attributes."""
    G.add_node(node_id)
    _set_node_attributes(G, node_id, is_incident, hop_dist, entity_name_lookup, location_lookup)


def _set_node_attributes(
    G: nx.MultiDiGraph,
    node: str,
    is_incident: bool,
    hop_dist: int,
    entity_name_lookup: Dict[str, str],
    location_lookup: Dict[str, Dict[str, Any]]
) -> None:
    """Sets standard typed attributes for an ACCOUNT or ATM node."""
    loc = location_lookup.get(node, {"state": "Unknown", "city": "Unknown", "latitude": 0.0, "longitude": 0.0})
    if node.startswith("ATM_"):
        node_type = "ATM"
        is_terminal = True
        canonical_name = "ATM Cash Terminal"
    else:
        node_type = "ACCOUNT"
        is_terminal = False
        canonical_name = entity_name_lookup.get(node, "Unknown Entity")

    G.nodes[node]["entity_id"] = str(node)
    G.nodes[node]["node_type"] = str(node_type)
    G.nodes[node]["canonical_name"] = str(canonical_name)
    G.nodes[node]["state"] = str(loc["state"])
    G.nodes[node]["city"] = str(loc["city"])
    G.nodes[node]["latitude"] = float(loc["latitude"])
    G.nodes[node]["longitude"] = float(loc["longitude"])
    G.nodes[node]["is_incident"] = bool(is_incident)
    G.nodes[node]["is_terminal"] = bool(is_terminal)
    G.nodes[node]["hop_distance"] = int(hop_dist)


# ==============================================================================
# Graph Statistics & Evaluation Calculation
# ==============================================================================

def calculate_graph_statistics(
    subgraph: nx.MultiDiGraph,
    complaint_id: str,
    incident_entity: str,
    incident_time: datetime,
    window_start: datetime,
    window_end: datetime
) -> Dict[str, Any]:
    """
    Computes comprehensive structural and financial statistics for an incident subgraph.
    """
    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()

    # Node counts by type
    num_account_nodes = sum(1 for _, d in subgraph.nodes(data=True) if d.get("node_type") == "ACCOUNT")
    num_atm_nodes = sum(1 for _, d in subgraph.nodes(data=True) if d.get("node_type") == "ATM")
    num_terminal_nodes = sum(1 for _, d in subgraph.nodes(data=True) if d.get("is_terminal") is True)

    # Maximum hop distance
    max_hop = max([d.get("hop_distance", 0) for _, d in subgraph.nodes(data=True)], default=0)

    # Edge and financial amounts
    amounts = [d.get("amount", 0.0) for _, _, d in subgraph.edges(data=True)]
    total_val = float(sum(amounts))
    max_val = float(max(amounts)) if amounts else 0.0
    avg_val = float(total_val / len(amounts)) if amounts else 0.0

    # Cash-out edges
    num_cash_out = sum(1 for _, _, d in subgraph.edges(data=True) if d.get("is_cash_out") == 1)

    # Incident node degree
    if incident_entity in subgraph:
        in_deg_inc = subgraph.in_degree(incident_entity)
        out_deg_inc = subgraph.out_degree(incident_entity)
    else:
        in_deg_inc = 0
        out_deg_inc = 0

    # Graph connectivity and density
    # Density for directed multigraph: num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1
    density = round(num_edges / (num_nodes * (num_nodes - 1)), 6) if num_nodes > 1 else 0.0
    num_components = nx.number_weakly_connected_components(subgraph) if num_nodes > 0 else 0
    avg_degree = round(float(2 * num_edges / num_nodes), 4) if num_nodes > 0 else 0.0

    # Ground-truth evaluation attributes (offline evaluation only)
    suspicious_edges = [d for _, _, d in subgraph.edges(data=True) if d.get("is_suspicious") == 1]
    contains_suspicious = 1 if len(suspicious_edges) > 0 else 0
    rings = set(d.get("ring_id") for d in suspicious_edges if d.get("ring_id") and d.get("ring_id") != "NORMAL")
    suspicious_ring_count = len(rings)

    return {
        "complaint_id": complaint_id,
        "incident_entity_id": incident_entity,
        "incident_time": incident_time.strftime("%Y-%m-%d %H:%M:%S"),
        "window_start": window_start.strftime("%Y-%m-%d %H:%M:%S"),
        "window_end": window_end.strftime("%Y-%m-%d %H:%M:%S"),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_account_nodes": num_account_nodes,
        "num_atm_nodes": num_atm_nodes,
        "num_terminal_nodes": num_terminal_nodes,
        "max_hop": max_hop,
        "total_transaction_value": round(total_val, 2),
        "max_transaction_value": round(max_val, 2),
        "avg_transaction_value": round(avg_val, 2),
        "num_cash_out_edges": num_cash_out,
        "in_degree_incident": in_deg_inc,
        "out_degree_incident": out_deg_inc,
        "density": density,
        "number_of_connected_components": num_components,
        "average_degree": avg_degree,
        "contains_suspicious_activity": contains_suspicious,
        "suspicious_ring_count": suspicious_ring_count
    }


# ==============================================================================
# Validation Routine
# ==============================================================================

def validate_incident_graphs(
    df_complaints: pd.DataFrame,
    df_summary: pd.DataFrame,
    valid_entities: Set[str],
    valid_atms: Set[str],
    sample_subgraph: nx.MultiDiGraph,
    sample_graph_path: Path
) -> None:
    """
    Runs automated validation assertions across extracted graphs and summaries.
    """
    # 1. Every complaint has a predicted_entity_id
    assert len(df_summary) == len(df_complaints), (
        f"Summary count {len(df_summary)} does not match complaint count {len(df_complaints)}!"
    )
    assert df_summary["incident_entity_id"].notna().all(), "Found missing incident_entity_id!"

    # 2. Every predicted_entity_id exists in entity_master.csv
    assert set(df_summary["incident_entity_id"]).issubset(valid_entities), (
        "Found incident_entity_id not present in entity_master.csv!"
    )

    # 3. Max hop distance in any graph is <= 3
    assert (df_summary["max_hop"] <= 3).all(), "Found node with hop distance > 3!"

    # 4. ATM node consistency
    assert (df_summary["num_atm_nodes"] == df_summary["num_terminal_nodes"]).all(), (
        "Mismatch between num_atm_nodes and num_terminal_nodes!"
    )

    # 5. Roundtrip validation of GraphML format
    reloaded_G = nx.read_graphml(sample_graph_path)
    assert reloaded_G.number_of_nodes() == sample_subgraph.number_of_nodes(), (
        "GraphML node count mismatch upon reload!"
    )
    assert reloaded_G.number_of_edges() == sample_subgraph.number_of_edges(), (
        "GraphML edge count mismatch upon reload!"
    )

    print("All Stage 2 graph construction validations PASSED successfully!")


# ==============================================================================
# Demonstration Visualization Routine
# ==============================================================================

def create_demo_visualization(
    subgraph: nx.MultiDiGraph,
    demo_meta: Dict[str, Any],
    output_path: Path = DEMO_GRAPH_IMAGE_FILE
) -> None:
    """
    Renders and saves a clear NetworkX + Matplotlib visual representation
    for a multi-hop incident subgraph showing Account nodes, ATM terminals, and directed flows.
    """
    plt.figure(figsize=(13, 9))
    plt.clf()

    incident_node = demo_meta["incident_entity_id"]
    nodes = list(subgraph.nodes())

    # Build node color and size maps
    node_colors = []
    node_sizes = []
    node_labels = {}

    for n in nodes:
        d = subgraph.nodes[n]
        node_labels[n] = n
        if n == incident_node:
            node_colors.append("#E63946")  # Crimson Red for Incident
            node_sizes.append(1800)
        elif d.get("node_type") == "ATM":
            node_colors.append("#2A9D8F")  # Teal / Emerald for ATM
            node_sizes.append(1400)
        else:
            node_colors.append("#457B9D")  # Slate Blue for Account
            node_sizes.append(1200)

    # Compute a clean layout
    # Use spring layout with fixed seed
    pos = nx.spring_layout(subgraph, seed=42, k=1.8 / np.sqrt(max(len(nodes), 1)))

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        nodelist=nodes,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        edgecolors="#1D3557",
        linewidths=2.0
    )

    # Draw node labels
    nx.draw_networkx_labels(
        subgraph, pos,
        labels=node_labels,
        font_size=8.5,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#FFFFFF"
    )

    # Distinguish suspicious vs normal edges for visualization clarity
    susp_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get("is_suspicious") == 1]
    norm_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get("is_suspicious") == 0]

    if norm_edges:
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=norm_edges,
            edge_color="#A8DADC",
            width=1.8,
            alpha=0.7,
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1"
        )

    if susp_edges:
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=susp_edges,
            edge_color="#E76F51",
            width=2.6,
            alpha=0.95,
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.12"
        )

    # Legend and Title
    plt.title(
        f"Stage 2 Incident Subgraph — Complaint {demo_meta['complaint_id']} (Incident Node: {incident_node})\n"
        f"72-Hour Window: {demo_meta['window_start']} to {demo_meta['window_end']} | Nodes: {demo_meta['num_nodes']} | Edges: {demo_meta['num_edges']}",
        fontsize=12, fontweight="bold", pad=15
    )

    # Legend handles
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Incident Node (Root)', markerfacecolor='#E63946', markersize=14),
        plt.Line2D([0], [0], marker='o', color='w', label='Bank Account Node', markerfacecolor='#457B9D', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='ATM Terminal (Cash-Out)', markerfacecolor='#2A9D8F', markersize=12),
        plt.Line2D([0], [0], color='#E76F51', lw=2.5, label='Mule Ring Transaction (Ground Truth)'),
        plt.Line2D([0], [0], color='#A8DADC', lw=2, label='Normal Transaction Flow')
    ]
    plt.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=9)
    plt.axis("off")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved demonstration graph visualization to: {output_path}")


# ==============================================================================
# Main Generation Pipeline
# ==============================================================================

def main():
    print("=" * 60)
    print("   STAGE 2 — INCIDENT SUBGRAPH EXTRACTION ENGINE")
    print("=" * 60)

    # 1. Load All Prerequisites
    df_complaints = load_complaints()
    df_resolved = load_resolved_entities()
    df_master = load_entity_master()
    df_locations = load_entity_locations()
    df_transactions = load_transactions()

    complaint_to_entity = df_resolved.set_index("complaint_id")["predicted_entity_id"].to_dict()

    entity_name_lookup, location_lookup, valid_entities, valid_atms = build_location_and_entity_lookups(
        df_master, df_locations, df_transactions
    )

    print(f"Loaded {len(df_complaints)} complaints and {len(df_transactions)} transactions.")
    print(f"Entity universe: {len(valid_entities)} financial accounts, {len(valid_atms)} ATM terminal nodes.")

    # 2. Process All 1,000 Complaints
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []

    print(f"\nExtracting 72-hour 3-hop incident subgraphs for {len(df_complaints)} complaints...")

    # Keep a reference sample graph for validation
    sample_subgraph: Optional[nx.MultiDiGraph] = None
    sample_graph_path: Optional[Path] = None

    # Track demonstration candidate
    demo_candidate_id = "C000014"  # Multi-hop ring with cash-out terminal
    demo_subgraph: Optional[nx.MultiDiGraph] = None
    demo_meta: Optional[Dict[str, Any]] = None

    for idx, c_row in df_complaints.iterrows():
        c_id = str(c_row["complaint_id"])
        inc_entity = complaint_to_entity[c_id]
        c_date_str = str(c_row["complaint_date"]).strip()

        # 72-hour window anchored at complaint_date 00:00:00
        incident_time = datetime.strptime(c_date_str, "%Y-%m-%d")
        window_start = incident_time - timedelta(hours=DEFAULT_WINDOW_HOURS)
        window_end = incident_time + timedelta(hours=DEFAULT_WINDOW_HOURS)

        # Filter window transactions
        df_window_tx = filter_transactions_by_time(df_transactions, window_start, window_end)

        # Extract 3-hop MultiDiGraph
        subgraph = extract_incident_subgraph(
            df_window_tx=df_window_tx,
            incident_entity=inc_entity,
            incident_time=incident_time,
            entity_name_lookup=entity_name_lookup,
            location_lookup=location_lookup,
            max_hops=DEFAULT_MAX_HOPS
        )

        # Compute graph metrics
        graph_stats = calculate_graph_statistics(
            subgraph=subgraph,
            complaint_id=c_id,
            incident_entity=inc_entity,
            incident_time=incident_time,
            window_start=window_start,
            window_end=window_end
        )
        summaries.append(graph_stats)

        # Save individual GraphML file
        graph_path = GRAPHS_DIR / f"{c_id}.graphml"
        nx.write_graphml(subgraph, graph_path)

        if sample_subgraph is None and subgraph.number_of_nodes() > 1:
            sample_subgraph = subgraph
            sample_graph_path = graph_path

        if c_id == demo_candidate_id:
            demo_subgraph = subgraph
            demo_meta = graph_stats

    # Fallback demo candidate if not set
    if demo_subgraph is None:
        demo_candidate_id = df_complaints.iloc[0]["complaint_id"]
        demo_subgraph = sample_subgraph
        demo_meta = summaries[0]

    # 3. Save Summary CSV
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(GRAPH_SUMMARY_FILE, index=False)
    print(f"[SUCCESS] Saved graph summary table to: {GRAPH_SUMMARY_FILE}")
    print(f"[SUCCESS] Saved {len(df_complaints)} incident subgraphs to: {GRAPHS_DIR}/")

    # 4. Run Validations
    validate_incident_graphs(
        df_complaints=df_complaints,
        df_summary=df_summary,
        valid_entities=valid_entities,
        valid_atms=valid_atms,
        sample_subgraph=sample_subgraph,
        sample_graph_path=sample_graph_path
    )

    # 5. Generate Demonstration Visualization
    create_demo_visualization(demo_subgraph, demo_meta, DEMO_GRAPH_IMAGE_FILE)

    # 6. Print Overall Statistics
    print("\n" + "=" * 55)
    print("      STAGE 2 INCIDENT GRAPH EXTRACTION SUMMARY")
    print("=" * 55)
    print(f"Total Complaints Processed : {len(df_summary)}")
    print(f"Total Incident Graphs Saved: {len(df_summary)}")
    print(f"Average Nodes per Graph    : {df_summary['num_nodes'].mean():.2f}")
    print(f"Average Edges per Graph    : {df_summary['num_edges'].mean():.2f}")
    print(f"Maximum Nodes in a Graph   : {df_summary['num_nodes'].max()}")
    print(f"Maximum Edges in a Graph   : {df_summary['num_edges'].max()}")
    print("-" * 55)
    susp_graphs = (df_summary["contains_suspicious_activity"] == 1).sum()
    cash_graphs = (df_summary["num_cash_out_edges"] > 0).sum()
    print(f"Graphs with Suspicious Activity: {susp_graphs} ({susp_graphs / len(df_summary) * 100:.1f}%)")
    print(f"Graphs with ATM Cash-Out Edges : {cash_graphs} ({cash_graphs / len(df_summary) * 100:.1f}%)")
    print("=" * 55 + "\n")

    # 7. Print Demonstration Report
    print("=" * 55)
    print("        GRAPH CONSTRUCTION DEMONSTRATION")
    print("=" * 55)
    print(f"Complaint ID   : {demo_meta['complaint_id']}")
    print(f"Incident Entity: {demo_meta['incident_entity_id']}")
    print("-" * 55)
    print("Time Window:")
    print(f"  From : {demo_meta['window_start']}")
    print(f"  To   : {demo_meta['window_end']}")
    print("-" * 55)
    print(f"Nodes          : {demo_meta['num_nodes']}")
    print(f"Edges          : {demo_meta['num_edges']}")
    print(f"Account Nodes  : {demo_meta['num_account_nodes']}")
    print(f"ATM Nodes      : {demo_meta['num_atm_nodes']}")
    print(f"Maximum Hop    : {demo_meta['max_hop']}")
    print(f"Cash-out Edges : {demo_meta['num_cash_out_edges']}")
    print(f"Total Value    : ₹{demo_meta['total_transaction_value']:,.2f}")
    print("-" * 55)
    has_susp_str = "YES" if demo_meta["contains_suspicious_activity"] == 1 else "NO"
    print(f"Suspicious activity present: {has_susp_str}")
    rings_observed = set(d.get("ring_id") for _, _, d in demo_subgraph.edges(data=True) if d.get("ring_id") != "NORMAL")
    print(f"Rings observed             : {', '.join(rings_observed) if rings_observed else 'None'}")
    print("=" * 55 + "\n")

    print(f"Example Transaction Paths in Incident Subgraph ({demo_meta['complaint_id']}):")
    print("-" * 90)
    for u, v, k, d in list(demo_subgraph.edges(data=True, keys=True))[:8]:
        flow_tag = "[SUSPICIOUS MULE FLOW]" if d.get("is_suspicious") == 1 else "[NORMAL FLOW]"
        print(f"  {u:<11} ──(₹{d['amount']:>10.2f} via {d['transaction_type']:<15})──> {v:<11} {flow_tag}")
    print("-" * 90 + "\n")

    print("Sample 5 Rows from data/graph_summary.csv:")
    print(df_summary.head(5).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
