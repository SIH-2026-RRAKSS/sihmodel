"""
Real-Time Transaction Stream Ingestion & Streaming Graph Inference Engine
=========================================================================
Implements an event-driven, sliding-window temporal graph accumulator and
dynamic incident scoring engine.

Key Capabilities:
1. TemporalTransactionGraph: Sliding-window memory index maintaining transactions
   within a configurable horizon (e.g. 72 hours).
2. On-the-Fly Incident Extraction: Sub-millisecond k-hop BFS around any complaint entity.
3. Live GraphSAGE Scoring: Inductive PyG feature construction and PyTorch model evaluation.
4. Micro-Batch Stream Simulator: Simulates a live transaction stream and benchmarks throughput.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.graphsage_classifier import DualHeadGraphSAGE

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


def normalize_single_graph_features(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    """Normalizes continuous feature dimensions using trained Z-score parameters."""
    x_norm = x.clone()
    if mean is not None and std is not None:
        continuous_indices = [2, 3, 4, 5, 6, 7, 8, 9, 12]
        for idx in continuous_indices:
            x_norm[:, idx] = (x_norm[:, idx] - mean[idx]) / std[idx]
    return x_norm


class DynamicAnomalyTrigger:
    """
    O(1) lightweight statistical behavioral anomaly gate (Stage 1).
    Applies Z-score and velocity thresholding to trigger proactive GNN triage.
    """
    def __init__(self):
        # account_id -> { count, mean, M2, daily_sums: {date_str: sum} }
        self.profiles = defaultdict(lambda: {
            "count": 0,
            "mean": 0.0,
            "M2": 0.0,
            "daily_sums": defaultdict(float)
        })

    def evaluate_transaction(self, tx: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        src = str(tx.get("sender_entity_id") or tx.get("source_entity_id"))
        amount = float(tx.get("amount", 0.0))
        ts_val = tx.get("timestamp")
        
        if isinstance(ts_val, str):
            ts = pd.to_datetime(ts_val)
        elif isinstance(ts_val, datetime):
            ts = ts_val
        else:
            ts = datetime.utcnow()
            
        date_str = ts.strftime("%Y-%m-%d")
        
        profile = self.profiles[src]
        count = profile["count"]
        mean = profile["mean"]
        
        # Rule 3: Cold-Start Safeguard
        if count == 0:
            self._update_profile(src, amount, date_str)
            if amount >= 200000:
                return True, f"COLD_START_SPIKE (Amt: {amount})"
            return False, None
            
        # Calculate standard deviation
        variance = profile["M2"] / count if count > 1 else 0.0
        std = np.sqrt(variance) if variance > 0 else 1.0  # avoid div by zero
        
        # Rule 1: Single Transaction Outlier
        z_score = (amount - mean) / std
        if z_score >= 3.5 and amount >= 25000:
            self._update_profile(src, amount, date_str)
            return True, f"SINGLE_TX_OUTLIER (Z-Score: {z_score:.2f}, Amt: {amount})"
            
        # Rule 2: Daily Velocity Spike
        daily_sums = profile["daily_sums"]
        past_days = [d for d in daily_sums.keys() if d != date_str]
        
        if past_days:
            daily_avg = sum(daily_sums[d] for d in past_days) / len(past_days)
            current_day_sum = daily_sums[date_str] + amount
            if current_day_sum >= max(2.5 * daily_avg, 50000):
                self._update_profile(src, amount, date_str)
                return True, f"DAILY_VELOCITY_SPIKE (Day Sum: {current_day_sum}, Avg: {daily_avg:.2f})"
        
        self._update_profile(src, amount, date_str)
        return False, None

    def _update_profile(self, src: str, amount: float, date_str: str):
        profile = self.profiles[src]
        profile["count"] += 1
        delta = amount - profile["mean"]
        profile["mean"] += delta / profile["count"]
        delta2 = amount - profile["mean"]
        profile["M2"] += delta * delta2
        profile["daily_sums"][date_str] += amount


class TemporalTransactionGraph:
    """
    In-memory sliding-window temporal transaction graph engine.
    Maintains accounts, ATM terminals, and directed payment transactions.
    """

    def __init__(self, window_hours: int = 72, max_hops: int = 3):
        self.window_hours = window_hours
        self.max_hops = max_hops
        self.graph = nx.MultiDiGraph()
        self.events = deque()  # stores (timestamp, u, v, key)
        self.entity_locations: Dict[str, Tuple[float, float]] = {}
        self.entity_cities: Dict[str, str] = {}
        self.model: Optional[DualHeadGraphSAGE] = None
        self.model_config: Dict[str, Any] = {}
        self.mean_norm: Optional[torch.Tensor] = None
        self.std_norm: Optional[torch.Tensor] = None
        self._load_metadata_and_models()
        self.anomaly_trigger = DynamicAnomalyTrigger()
        self.proactive_alerts = []
        self.trigger_count = 0

    def _load_metadata_and_models(self):
        """Loads node metadata (coordinates/cities) and GraphSAGE model."""
        loc_file = DATA_DIR / "entity_locations.csv"
        if loc_file.exists():
            df_loc = pd.read_csv(loc_file)
            for _, row in df_loc.iterrows():
                eid = row["entity_id"]
                lat = float(row.get("latitude", 0.0)) if pd.notna(row.get("latitude")) else 0.0
                lon = float(row.get("longitude", 0.0)) if pd.notna(row.get("longitude")) else 0.0
                self.entity_locations[eid] = (lat, lon)
                self.entity_cities[eid] = str(row.get("city", "UNKNOWN"))

        # Load PyTorch GraphSAGE Model
        model_path = MODELS_DIR / "graphsage_model.pt"
        cfg_path = MODELS_DIR / "graphsage_config.json"
        if model_path.exists() and cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.model_config = json.load(f)
                in_dim = self.model_config.get("input_dim", 13)
                hidden_dim = self.model_config.get("hidden_dim", 64)
                self.model = DualHeadGraphSAGE(input_dim=in_dim, hidden_dim=hidden_dim)
                try:
                    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
                except Exception:
                    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
                self.model.load_state_dict(state_dict)
                self.model.eval()
                
                # Load normalization tensors
                mean_path = MODELS_DIR / "synthetic_mean.pt"
                std_path = MODELS_DIR / "synthetic_std.pt"
                if mean_path.exists() and std_path.exists():
                    self.mean_norm = torch.load(mean_path, map_location=torch.device("cpu"), weights_only=True)
                    self.std_norm = torch.load(std_path, map_location=torch.device("cpu"), weights_only=True)
            except Exception as e:
                print(f"[Streaming] Notice: Model checkpoint loading fallback: {e}")

    def ingest_transaction(self, tx: Dict[str, Any], purge_expired: bool = True) -> str:
        """
        Ingests a single transaction event into the sliding window.
        Purges events older than (current_timestamp - window_hours).
        """
        tx_id = str(tx.get("transaction_id", f"TX_{int(time.time()*1000)}"))
        src = str(tx.get("sender_entity_id") or tx.get("source_entity_id"))
        dst = str(tx.get("receiver_entity_id") or tx.get("destination_entity_id"))
        amount = float(tx.get("amount", 0.0))
        ts_val = tx.get("timestamp")
        if isinstance(ts_val, str):
            ts = pd.to_datetime(ts_val)
        elif isinstance(ts_val, datetime):
            ts = ts_val
        else:
            ts = datetime.utcnow()

        # Add / update source node
        if not self.graph.has_node(src):
            src_type = "ATM" if src.startswith("ATM_") else "ACCOUNT"
            self.graph.add_node(
                src,
                node_type=src_type,
                city=self.entity_cities.get(src, "UNKNOWN"),
                latitude=self.entity_locations.get(src, (0.0, 0.0))[0],
                longitude=self.entity_locations.get(src, (0.0, 0.0))[1],
                is_terminal=bool(src_type == "ATM")
            )

        # Add / update destination node
        if not self.graph.has_node(dst):
            dst_type = "ATM" if dst.startswith("ATM_") else "ACCOUNT"
            self.graph.add_node(
                dst,
                node_type=dst_type,
                city=self.entity_cities.get(dst, "UNKNOWN"),
                latitude=self.entity_locations.get(dst, (0.0, 0.0))[0],
                longitude=self.entity_locations.get(dst, (0.0, 0.0))[1],
                is_terminal=bool(dst_type == "ATM")
            )

        # Add edge
        key = self.graph.add_edge(
            src,
            dst,
            key=tx_id,
            transaction_id=tx_id,
            amount=amount,
            timestamp=str(ts),
            dt=ts,
            channel=tx.get("channel", "TRANSFER"),
            is_cash_out=bool(tx.get("is_cash_out", 0) or dst.startswith("ATM_"))
        )

        self.events.append((ts, src, dst, key))

        if purge_expired and self.events:
            cutoff = ts - timedelta(hours=self.window_hours)
            while self.events and self.events[0][0] < cutoff:
                old_ts, u, v, old_key = self.events.popleft()
                if self.graph.has_edge(u, v, key=old_key):
                    self.graph.remove_edge(u, v, key=old_key)

        # Stage 1: Fast O(1) Anomaly Trigger
        triggered, reason = self.anomaly_trigger.evaluate_transaction(tx)
        if triggered:
            self.trigger_count += 1
            # Stage 2: Event-Driven Graph Triage
            subgraph = self.extract_subgraph_around_entity(src, as_of_time=ts)
            res = self.score_subgraph_live(subgraph, seed_entity_id=src)
            if res["risk_probability"] >= 0.70:
                alert = {
                    "timestamp": str(ts),
                    "transaction_id": tx_id,
                    "seed_entity": src,
                    "trigger_reason": reason,
                    "gnn_risk": res["risk_probability"],
                    "confidence": res["confidence_tier"],
                    "mules": res.get("mule_probabilities", {})
                }
                self.proactive_alerts.append(alert)

        return tx_id

    def ingest_batch(self, transactions: List[Dict[str, Any]]) -> int:
        """High-throughput ingestion of transaction batch."""
        count = 0
        for tx in transactions:
            self.ingest_transaction(tx, purge_expired=False)
            count += 1
        return count

    def extract_subgraph_around_entity(
        self,
        seed_entity_id: str,
        max_hops: Optional[int] = None,
        as_of_time: Optional[datetime] = None
    ) -> nx.MultiDiGraph:
        """
        Extracts a k-hop directed MultiDiGraph centered around the seed entity.
        Bounded by max_hops and the 72-hour observation window.
        """
        hops = max_hops if max_hops is not None else self.max_hops
        subgraph = nx.MultiDiGraph()
        
        if not self.graph.has_node(seed_entity_id):
            raise KeyError(f"Entity {seed_entity_id} not found in active streaming graph. It may not have transacted in the current time window, or the pipeline is not correctly ingesting it.")
            
        # Optional: Start BFS queue from earliest transaction of seed entity to enforce downstream tracking
        start_time = as_of_time
        if start_time is None:
            # Fallback to earliest transaction time
            earliest = min((e[2].get("dt", datetime.min) for e in self.graph.out_edges(seed_entity_id, data=True)), default=datetime.min)
            start_time = earliest if earliest != datetime.min else datetime.min

        queue = deque([(seed_entity_id, start_time, 0)])
        visited_nodes = {seed_entity_id: 0} # map node -> shortest hop_distance
        
        while queue:
            curr_node, curr_time, curr_hop = queue.popleft()
            
            if curr_hop >= hops:
                continue
                
            max_time = start_time + timedelta(hours=self.window_hours)
            
            # Get all outbound edges
            out_edges = []
            for _, v, key, data in self.graph.out_edges(curr_node, data=True, keys=True):
                edge_time = data.get("dt", datetime.min)
                if curr_time <= edge_time <= max_time: # Temporal and window constraints
                    out_edges.append((v, key, data, edge_time))
                    
            # Cap Fan-out to Top 15 by amount
            out_edges.sort(key=lambda e: float(e[2].get("amount", 0.0)), reverse=True)
            out_edges = out_edges[:15]
            
            for v, key, data, edge_time in out_edges:
                if v not in visited_nodes or curr_hop + 1 < visited_nodes[v]:
                    visited_nodes[v] = curr_hop + 1
                    queue.append((v, edge_time, curr_hop + 1))
                    
        # Add all visited nodes to subgraph
        for node, dist in visited_nodes.items():
            ndata = self.graph.nodes[node].copy()
            ndata["hop_distance"] = dist
            ndata["is_incident"] = (node == seed_entity_id)
            subgraph.add_node(node, **ndata)
            
        # Add internal edges
        for u in visited_nodes:
            for _, v, key, data in self.graph.out_edges(u, data=True, keys=True):
                if v in visited_nodes:
                    subgraph.add_edge(u, v, key=key, **data)
                    
        return subgraph

    def score_subgraph_live(self, subgraph: nx.MultiDiGraph, seed_entity_id: str) -> Dict[str, Any]:
        """
        Runs live GraphSAGE inference and terminal cash-out prediction on the extracted subgraph.
        """
        node_list = list(subgraph.nodes())
        num_nodes = len(node_list)

        if self.model is None or num_nodes == 0:
            return {
                "risk_probability": 0.0,
                "confidence_tier": "NORMAL",
                "is_suspicious": False,
                "num_nodes": max(1, num_nodes),
                "num_edges": subgraph.number_of_edges(),
                "terminals": []
            }

        node_to_idx = {n: i for i, n in enumerate(node_list)}

        features = []
        for n in node_list:
            nd = subgraph.nodes[n]
            in_edges = list(subgraph.in_edges(n, data=True))
            out_edges = list(subgraph.out_edges(n, data=True))

            in_deg = len(in_edges)
            out_deg = len(out_edges)
            in_amt = sum(e[2].get("amount", 0.0) for e in in_edges)
            out_amt = sum(e[2].get("amount", 0.0) for e in out_edges)
            avg_in = in_amt / max(in_deg, 1)
            avg_out = out_amt / max(out_deg, 1)

            feat = [
                1.0 if nd.get("node_type") == "ACCOUNT" else 0.0,
                1.0 if nd.get("node_type") == "ATM" or str(n).startswith("ATM_") else 0.0,
                float(nd.get("hop_distance", 0)),
                float(in_deg),
                float(out_deg),
                float(in_amt),
                float(out_amt),
                float(avg_in),
                float(avg_out),
                float(in_deg + out_deg),
                1.0 if nd.get("is_incident", False) or n == seed_entity_id else 0.0,
                1.0 if nd.get("is_terminal", False) or str(n).startswith("ATM_") else 0.0,
                15.0  # city code placeholder (default to 15 matching CITY_MAP unknown)
            ]
            features.append(feat)

        x_raw = torch.tensor(features, dtype=torch.float32)
        x_norm = normalize_single_graph_features(x_raw, self.mean_norm, self.std_norm)

        edge_list = []
        for u, v in subgraph.edges():
            edge_list.append([node_to_idx[u], node_to_idx[v]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        batch = torch.zeros(len(node_list), dtype=torch.long)
        pyg_data = Data(x=x_norm, edge_index=edge_index, batch=batch)

        with torch.no_grad():
            out_node, out_graph, emb = self.model(pyg_data.x, pyg_data.edge_index, pyg_data.batch)
            prob = float(torch.sigmoid(out_graph).item())
            node_probs = torch.sigmoid(out_node).cpu().numpy()
            
        mule_probabilities = {str(node_list[i]): float(node_probs[i]) for i in range(len(node_list))}

        # Determine confidence tier
        has_atm = any(str(n).startswith("ATM_") or subgraph.nodes[n].get("is_terminal") for n in node_list)

        if prob >= 0.70 and (has_atm or num_nodes >= 3):
            tier = "HIGH_CONFIDENCE"
        elif prob >= 0.50:
            tier = "MEDIUM_CONFIDENCE"
        else:
            tier = "NORMAL"

        # Terminal candidates ranking
        terminal_candidates = []
        for n in node_list:
            if str(n).startswith("ATM_") or subgraph.nodes[n].get("is_terminal"):
                in_amt = sum(e[2].get("amount", 0.0) for e in subgraph.in_edges(n, data=True))
                score = round(0.25 * prob + 0.35 * (1.0 / (subgraph.nodes[n].get("hop_distance", 1) + 1)) + 0.40 * min(in_amt / 50000.0, 1.0), 4)
                terminal_candidates.append({
                    "terminal_id": n,
                    "city": self.entity_cities.get(n, "UNKNOWN"),
                    "terminal_score": score,
                    "coordinates": self.entity_locations.get(n, (0.0, 0.0)),
                    "hop_distance": subgraph.nodes[n].get("hop_distance", 1)
                })

        terminal_candidates.sort(key=lambda x: x["terminal_score"], reverse=True)

        return {
            "risk_probability": round(prob, 4),
            "confidence_tier": tier,
            "is_suspicious": bool(prob >= 0.50),
            "num_nodes": len(node_list),
            "num_edges": subgraph.number_of_edges(),
            "mule_probabilities": mule_probabilities,
            "terminals": terminal_candidates
        }


# ==============================================================================
# Streaming Simulator & Benchmark Suite
# ==============================================================================

def run_streaming_benchmark(num_transactions: int = 5000) -> Dict[str, Any]:
    """
    Simulates streaming transaction ingestion and benchmarks latency per incident.
    """
    print("=" * 70)
    print("[BENCHMARK] RUNNING STREAMING TRANSACTION INGESTION & INFERENCE BENCHMARK")
    print(f"Target transaction volume: {num_transactions}")
    print("=" * 70)

    engine = TemporalTransactionGraph(window_hours=72, max_hops=3)

    tx_file = DATA_DIR / "transactions.csv"
    if not tx_file.exists():
        print("[Streaming] transactions.csv not found, skipping benchmark.")
        return {}

    df_tx = pd.read_csv(tx_file).head(num_transactions)
    tx_records = df_tx.to_dict(orient="records")

    # Measure ingestion throughput
    t0 = time.time()
    for tx in tx_records:
        engine.ingest_transaction(tx, purge_expired=False)
    t_ingest = time.time() - t0
    tx_per_sec = len(tx_records) / max(t_ingest, 0.0001)

    print(f"[STREAM] Ingested {len(tx_records)} transactions in {t_ingest:.3f}s ({tx_per_sec:,.0f} Tx/sec)")
    print(f"         Graph state: {engine.graph.number_of_nodes()} nodes, {engine.graph.number_of_edges()} edges")

    # Benchmark dynamic incident extraction & GraphSAGE inference
    complaints_file = DATA_DIR / "complaints.csv"
    resolved_file = DATA_DIR / "resolved_entities.csv"
    sample_entities = []

    if resolved_file.exists():
        df_res = pd.read_csv(resolved_file)
        sample_entities = df_res["predicted_entity_id"].dropna().unique()[:100].tolist()
    else:
        sample_entities = list(engine.graph.nodes())[:100]

    latencies_ms = []
    scored_results = []

    for eid in sample_entities:
        t_start = time.perf_counter()
        subgraph = engine.extract_subgraph_around_entity(eid, max_hops=3)
        res = engine.score_subgraph_live(subgraph, seed_entity_id=eid)
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        latencies_ms.append(lat_ms)
        scored_results.append(res)

    lat_arr = np.array(latencies_ms)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_lat = float(np.mean(lat_arr))

    print(f"\n[LATENCY] DYNAMIC GRAPH INFERENCE LATENCY (N = {len(sample_entities)} queries):")
    print(f"   * Mean Latency:    {mean_lat:.2f} ms")
    print(f"   * Median (p50):    {p50:.2f} ms")
    print(f"   * 95th Percentile: {p95:.2f} ms")
    print(f"   * 99th Percentile: {p99:.2f} ms")
    print(f"   * Target Sub-50ms SLA: {'PASSED [OK]' if p95 < 50.0 else 'WARNING'}")

    benchmark_summary = {
        "transactions_ingested": len(tx_records),
        "ingestion_duration_sec": round(t_ingest, 4),
        "ingestion_rate_tx_per_sec": round(tx_per_sec, 1),
        "stage1_trigger_count": engine.trigger_count,
        "stage1_filter_efficiency": round(100.0 * (1.0 - (engine.trigger_count / max(len(tx_records), 1))), 2),
        "proactive_alerts_generated": len(engine.proactive_alerts),
        "graph_node_count": engine.graph.number_of_nodes(),
        "graph_edge_count": engine.graph.number_of_edges(),
        "num_incident_queries": len(sample_entities),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "sub_50ms_sla_passed": bool(p95 < 50.0)
    }

    # Save benchmark report
    out_file = DATA_DIR / "streaming_benchmark_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"[Streaming] Benchmark summary saved to: {out_file}")

    return benchmark_summary


if __name__ == "__main__":
    run_streaming_benchmark(5000)
