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
    assert "total_incidents_monitored" in data
    assert "tier_breakdown" in data
    assert "model_comparison" in data


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


def test_api_incident_graph():
    """Tests /api/incidents/{incident_id}/graph structure generation."""
    response = client.get("/api/incidents/C000003/graph")
    assert response.status_code == 200
    data = response.json()
    assert data["incident_id"] == "C000003"
    assert len(data["nodes"]) > 0
    assert "id" in data["nodes"][0]
    assert "color" in data["nodes"][0]


def test_api_live_prediction():
    """Tests /api/predict/subgraph live inference."""
    payload = {"seed_entity_id": "ENT_000040", "max_hops": 3}
    response = client.post("/api/predict/subgraph", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["seed_entity_id"] == "ENT_000040"
    assert "risk_probability" in data
    assert "confidence_tier" in data


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
