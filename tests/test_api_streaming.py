"""
Automated Test Suite for Stage 8 Backend API, Streaming Engine & DB Persistence
================================================================================
Executes end-to-end integration and unit tests for:
1. SQLite / SQLAlchemy Database Persistence Layer
2. Temporal Sliding-Window Streaming Transaction Ingestion Engine
3. FastAPI Backend REST API Endpoints & Case Dossier Generation
4. Policy Threshold Calibration Simulator
"""

import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database import get_db_session, Complaint, EntityMaster, TransactionRecord, IncidentPrediction
from src.streaming_engine import TemporalTransactionGraph
from src.api import app


client = TestClient(app)


# ==============================================================================
# 1. Database Persistence Tests
# ==============================================================================

from src.database import DEFAULT_DB_PATH
import pytest
import os

@pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="Database not seeded")
def test_database_entities_and_complaints():
    """Verifies that database is properly seeded and indexed."""
    session = get_db_session()
    try:
        entity_count = session.query(EntityMaster).count()
        complaint_count = session.query(Complaint).count()
        tx_count = session.query(TransactionRecord).count()
        pred_count = session.query(IncidentPrediction).count()

        assert entity_count >= 700, f"Expected >= 700 entities, got {entity_count}"
        assert complaint_count >= 1000, f"Expected >= 1000 complaints, got {complaint_count}"
        assert tx_count >= 10000, f"Expected >= 10000 transactions, got {tx_count}"
        assert pred_count >= 1000, f"Expected >= 1000 predictions, got {pred_count}"

        # Test entity resolution relationship
        sample_comp = session.query(Complaint).filter(Complaint.complaint_id == "C000001").first()
        assert sample_comp is not None
        assert sample_comp.predicted_entity_id is not None
    finally:
        session.close()


# ==============================================================================
# 2. Streaming Engine Ingestion & Subgraph Extraction Tests
# ==============================================================================

def test_streaming_engine_ingestion_and_subgraph():
    """Tests temporal sliding-window graph ingestion and k-hop BFS."""
    engine = TemporalTransactionGraph(window_hours=72, max_hops=3)

    sample_txs = [
        {"transaction_id": "TX_TEST_01", "sender_entity_id": "ENT_000001", "receiver_entity_id": "ENT_000002", "amount": 50000.0, "timestamp": "2026-08-25 10:00:00"},
        {"transaction_id": "TX_TEST_02", "sender_entity_id": "ENT_000002", "receiver_entity_id": "ATM_001", "amount": 48000.0, "timestamp": "2026-08-25 11:30:00", "is_cash_out": 1},
    ]

    for tx in sample_txs:
        tx_id = engine.ingest_transaction(tx)
        assert tx_id is not None

    assert engine.graph.number_of_nodes() >= 3
    assert engine.graph.number_of_edges() >= 2

    # Extract subgraph around seed
    sub = engine.extract_subgraph_around_entity("ENT_000001", max_hops=2)
    assert sub.has_node("ENT_000001")
    assert sub.has_node("ENT_000002")
    assert sub.has_node("ATM_001")
    assert sub.nodes["ENT_000001"]["is_incident"] is True


# ==============================================================================
# 3. FastAPI REST API Endpoint Tests
# ==============================================================================

def test_api_health():
    """Tests /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "timestamp" in data
    assert data["database_connected"] is True


def test_api_stats():
    """Tests /api/stats endpoint."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_incidents_monitored"] >= 0
    assert "HIGH_CONFIDENCE" in data["tier_breakdown"]
    assert len(data["model_comparison"]) > 0
    assert "model" in data["model_comparison"][0]


def test_api_list_incidents():
    """Tests /api/incidents endpoint with pagination and filtering."""
    response = client.get("/api/incidents?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] >= 1000
    assert len(data["items"]) == 10
    assert "complaint_id" in data["items"][0]

    # Test filtering by tier
    response_high = client.get("/api/incidents?tier=HIGH_CONFIDENCE&page=1&page_size=5")
    assert response_high.status_code == 200
    data_high = response_high.json()
    for item in data_high["items"]:
        assert item["confidence_tier"] == "HIGH_CONFIDENCE"


def test_api_incident_detail():
    """Tests /api/incidents/{incident_id} endpoint."""
    response = client.get("/api/incidents/C000003")
    assert response.status_code == 200
    data = response.json()
    assert data["complaint"]["complaint_id"] == "C000003"
    assert "resolved_canonical_entity" in data
    assert "model_prediction" in data
    assert len(data.get("investigative_evidence_bullets", [])) >= 3

    # Test case beyond 50 (Issue #04 regression test)
    res_beyond = client.get("/api/incidents/C000100")
    assert res_beyond.status_code == 200
    data_beyond = res_beyond.json()
    assert len(data_beyond.get("investigative_evidence_bullets", [])) >= 3
    assert len(data_beyond["model_prediction"].get("executive_summary", "")) > 0
    # Issue #14: Head 2 Node Mule Probability
    assert "node_mule_probability_head2" in data_beyond["model_prediction"]
    assert data_beyond["model_prediction"]["node_mule_probability_head2"] is not None


def test_api_incident_graph():
    """Tests /api/incidents/{incident_id}/graph structure generation."""
    response = client.get("/api/incidents/C000003/graph")
    assert response.status_code == 200
    data = response.json()
    assert data["incident_id"] == "C000003"
    assert len(data["nodes"]) > 0
    assert "id" in data["nodes"][0]
    assert "color" in data["nodes"][0]
    # Issue #14: Head 2 Node Mule Score attached to graph nodes
    assert "node_mule_score" in data["nodes"][0]
    assert data["nodes"][0]["node_mule_score"] >= 0.0
    assert data["is_dormant"] is False

    # Issue #16: Test dormant incident handling (C000001 has 0 transfers in 72h window)
    res_dormant = client.get("/api/incidents/C000001/graph")
    assert res_dormant.status_code == 200
    data_dormant = res_dormant.json()
    assert data_dormant["is_dormant"] is True
    assert data_dormant["num_edges"] == 0
    assert data_dormant["lifetime_tx_count"] > 0
    assert "surveillance window" in data_dormant["dormant_reason"].lower()
    assert data_dormant["nodes"][0]["is_dormant"] is True

    # Issue #16: Test expanded historical graph for C000001
    res_expanded = client.get("/api/incidents/C000001/graph?expand_historical=true")
    assert res_expanded.status_code == 200
    data_expanded = res_expanded.json()
    assert data_expanded["is_historical_expanded"] is True
    assert data_expanded["num_nodes"] > 0
    assert data_expanded["num_edges"] > 0


def test_api_live_prediction():
    """Tests /api/predict/subgraph live inference."""
    payload = {"seed_entity_id": "ENT_000040", "max_hops": 3}
    response = client.post("/api/predict/subgraph", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["seed_entity_id"] == "ENT_000040"
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert "confidence_tier" in data
    assert data["confidence_tier"] in ["NORMAL", "MEDIUM_CONFIDENCE", "HIGH_CONFIDENCE"]


def test_api_policy_tuning():
    """Tests /api/policy/tune endpoint with synthetic and IBM data."""
    payload = {"threshold": 0.70, "dataset": "synthetic"}
    response = client.post("/api/policy/tune", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["threshold"] == 0.70
    assert data["precision_percent"] > 85.0
    assert "alerts_generated" in data


def test_api_dossier_export():
    """Tests /api/dossier/{incident_id}/export endpoint in Markdown, HTML, and JSON."""
    # Markdown
    res_md = client.get("/api/dossier/C000003/export?format=markdown")
    assert res_md.status_code == 200
    assert "FINANCIAL CYBERCRIME INVESTIGATIVE DOSSIER" in res_md.text

    # HTML
    res_html = client.get("/api/dossier/C000003/export?format=html")
    assert res_html.status_code == 200
    assert "<html>" in res_html.text

    # JSON
    res_json = client.get("/api/dossier/C000003/export?format=json")
    assert res_json.status_code == 200
    assert res_json.json()["complaint"]["complaint_id"] == "C000003"


def test_api_three_way_benchmark():
    """Tests /api/benchmarks/three_way endpoint."""
    response = client.get("/api/benchmarks/three_way")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_transaction_ingest():
    """Tests POST /api/ingest/transaction endpoint for Stage 1 gate & Stage 2 triage."""
    # 1. Normal transaction (Stage 1 not flagged)
    res_norm = client.post("/api/ingest/transaction", json={
        "source_entity": "ENT_000001",
        "destination_entity": "ENT_000002",
        "amount": 2000.0
    })
    assert res_norm.status_code == 200
    data_norm = res_norm.json()
    assert "transaction_id" in data_norm
    assert data_norm["stage_1_flagged"] is False

    # 2. Anomaly transaction (high volume triggering Stage 1 & Stage 2)
    res_high = client.post("/api/ingest/transaction", json={
        "source_entity": "ENT_000001",
        "destination_entity": "ATM_001",
        "amount": 500000.0
    })
    assert res_high.status_code == 200
    data_high = res_high.json()
    assert data_high["stage_1_flagged"] is True
    assert data_high["stage_2_risk_probability"] is not None


def test_out_of_order_streaming_eviction():
    """Tests out-of-order event ingestion & min-heap sliding-window eviction."""
    from datetime import datetime, timedelta
    engine = TemporalTransactionGraph(window_hours=72, warmup=False)
    base_time = datetime(2026, 8, 1, 12, 0, 0)

    # Ingest event at T0
    engine.ingest_transaction({
        "transaction_id": "TX_1",
        "sender_entity_id": "ENT_000001",
        "receiver_entity_id": "ENT_000002",
        "amount": 1000.0,
        "timestamp": base_time
    })

    # Ingest out-of-order event at T0 - 10h
    engine.ingest_transaction({
        "transaction_id": "TX_0",
        "sender_entity_id": "ENT_000001",
        "receiver_entity_id": "ENT_000003",
        "amount": 500.0,
        "timestamp": base_time - timedelta(hours=10)
    })
    assert engine.graph.number_of_edges() == 2

    # Ingest event at T0 + 70h (purges TX_0 which is 80h old, retains TX_1 and TX_2)
    engine.ingest_transaction({
        "transaction_id": "TX_2",
        "sender_entity_id": "ENT_000002",
        "receiver_entity_id": "ENT_000004",
        "amount": 2000.0,
        "timestamp": base_time + timedelta(hours=70)
    })
    assert not engine.graph.has_edge("ENT_000001", "ENT_000003", key="TX_0")
    assert engine.graph.has_edge("ENT_000001", "ENT_000002", key="TX_1")
    assert engine.graph.has_edge("ENT_000002", "ENT_000004", key="TX_2")

def test_api_entities_locations():
    response = client.get("/api/entities/locations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Could be empty depending on the test data, but it shouldn't 500
