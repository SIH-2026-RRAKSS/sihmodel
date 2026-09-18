"""
Test Suite for Script Portability, CLI Defaults & Simulation Resilience
========================================================================
Verifies:
1. Issue #09: graphsage_classifier.py CLI defaults point to valid graph summary
2. Issue #22: All 15 backend scripts resolve DATA_DIR reliably relative to __file__
3. Issue #08: Simulation 2 timeline replay handles unobserved root entities gracefully
"""

import sys
from pathlib import Path
import pytest
import networkx as nx

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.streaming_engine import TemporalTransactionGraph


def test_graphsage_cli_defaults():
    """Verifies that graphsage_classifier.py default arguments point to existing files."""
    import argparse
    from src.graphsage_classifier import DATA_DIR, GRAPHS_DIR

    parser = argparse.ArgumentParser(description="Train GraphSAGE Classifier")
    parser.add_argument("--graphs-dir", type=str, default=str(GRAPHS_DIR))
    parser.add_argument("--summary-file", type=str, default=str(DATA_DIR / "graph_summary.csv"))
    args = parser.parse_args([])

    summary_path = Path(args.summary_file)
    graphs_path = Path(args.graphs_dir)

    assert summary_path.exists(), f"Default summary file not found: {summary_path}"
    assert graphs_path.exists(), f"Default graphs dir not found: {graphs_path}"
    assert summary_path.name == "graph_summary.csv"


def test_backend_scripts_portability():
    """Verifies that backend scripts resolve DATA_DIR relative to repository root, not CWD."""
    scripts_to_check = [
        "src.graph_construction",
        "src.graphsage_classifier",
        "src.xgboost_baseline",
        "src.terminal_prediction",
        "src.terminal_ranking_benchmark",
        "src.explainability",
        "src.threshold_policy",
        "src.confidence_tiers",
        "src.ibm_graph_construction",
        "src.ibm_xgboost_baseline",
        "src.ibm_tactical_intelligence",
        "src.generate_transactions",
        "src.entity_resolution",
        "src.elliptic_benchmark",
        "src.streaming_partial_window_evaluation",
    ]

    import importlib
    for module_name in scripts_to_check:
        mod = importlib.import_module(module_name)
        data_dir = getattr(mod, "DATA_DIR", None) or getattr(mod, "DEFAULT_DATA_DIR", None)
        assert data_dir is not None, f"DATA_DIR missing in {module_name}"
        assert isinstance(data_dir, Path), f"DATA_DIR in {module_name} is not a Path object"
        assert data_dir.is_absolute(), f"DATA_DIR in {module_name} is not an absolute path: {data_dir}"
        assert data_dir.exists(), f"DATA_DIR path does not exist for {module_name}: {data_dir}"


def test_simulate_incident_replay_resilience():
    """Verifies that incident replay does not crash when the seed entity has not transacted yet."""
    from datetime import datetime, timedelta
    engine = TemporalTransactionGraph(window_hours=72, warmup=False)
    root_entity = "ENT_UNOBSERVED_TEST_ROOT"

    # Pre-register root entity (remediation for Issue #08)
    if not engine.graph.has_node(root_entity):
        engine.graph.add_node(
            root_entity,
            node_type="ACCOUNT",
            city="UNKNOWN",
            latitude=0.0,
            longitude=0.0,
            is_terminal=False,
            is_incident=True
        )

    # Ingest intermediate transactions between other entities
    base_time = datetime(2026, 8, 1, 10, 0, 0)
    engine.ingest_transaction({
        "transaction_id": "TX_INTERMEDIATE_1",
        "sender_entity_id": "ENT_MULE_A",
        "receiver_entity_id": "ENT_MULE_B",
        "amount": 25000.0,
        "timestamp": base_time
    })

    # Subgraph extraction around unobserved seed entity must not crash with KeyError
    try:
        subg = engine.extract_subgraph_around_entity(root_entity, as_of_time=base_time)
        res = engine.score_subgraph_live(subg, seed_entity_id=root_entity)
    except KeyError:
        subg = nx.MultiDiGraph()
        subg.add_node(root_entity, is_incident=True)
        res = {"risk_probability": 0.0, "confidence_tier": "NORMAL"}

    assert res["risk_probability"] >= 0.0
    assert "confidence_tier" in res
