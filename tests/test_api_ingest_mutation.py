"""
Test Suite for Dynamic Ingestion and Mutation Endpoints (Issue #07)
===================================================================
Verifies:
1. POST /api/complaints (FIR / Citizen Complaint Registration & Live Triage)
2. POST /api/upload/transactions (Bank CSV Ledger Upload & Sliding Window Ingestion)
3. POST /api/ingest/transactions/batch (JSON Batch Transaction Ingestion)
4. POST /api/entities (Target Account and ATM Terminal Provisioning)
5. Audit Log Tracing across all mutations
"""

import io
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api import app, STREAMING_ENGINE
from src.database import get_db_session, Complaint, EntityMaster, TransactionRecord, IncidentPrediction, AuditLog

client = TestClient(app)


def test_register_complaint_and_verify_pipeline():
    """Tests POST /api/complaints endpoint and downstream verification."""
    unique_cid = f"TEST_FIR_{int(time.time()) % 100000}"
    test_acc = f"999888{int(time.time()) % 10000}"

    payload = {
        "complaint_id": unique_cid,
        "complaint_date": "2026-09-19",
        "complainant_name": "Test Inspector",
        "police_station_id": "PS_CYBER_CRIME",
        "district": "New Delhi",
        "state": "Delhi",
        "reported_account_number": test_acc,
        "reported_ifsc": "HDFC0001234",
        "reported_amount": 150000.0,
        "scam_category": "TASK_FRAUD",
        "description": "Victim was lured into fake investment group."
    }

    try:
        # 1. Register complaint
        response = client.post("/api/complaints", json=payload)
        assert response.status_code == 201, response.text
        data = response.json()
        assert data["complaint_id"] == unique_cid
        assert data["incident_id"] == f"INC_{unique_cid}"
        assert data["predicted_entity_id"] is not None
        assert "graphsage_risk_probability" in data
        assert "confidence_tier" in data

        # 2. Verify in SQLite database
        session = get_db_session()
        try:
            comp = session.query(Complaint).filter(Complaint.complaint_id == unique_cid).first()
            assert comp is not None
            assert comp.reported_account_number == test_acc
            assert comp.reported_amount == 150000.0

            # Verify auto-provisioned entity
            entity = session.query(EntityMaster).filter(EntityMaster.entity_id == comp.predicted_entity_id).first()
            assert entity is not None
            assert entity.canonical_account_number == test_acc

            # Verify incident prediction
            pred = session.query(IncidentPrediction).filter(IncidentPrediction.complaint_id == unique_cid).first()
            assert pred is not None
            assert pred.incident_id == f"INC_{unique_cid}"

            # Verify audit log
            audit = session.query(AuditLog).filter(
                AuditLog.action == "COMPLAINT_REGISTERED",
                AuditLog.target_id == unique_cid
            ).first()
            assert audit is not None
        finally:
            session.close()

        # 3. Verify queryable via /api/incidents
        inc_res = client.get(f"/api/incidents?search={unique_cid}")
        assert inc_res.status_code == 200
        inc_items = inc_res.json()["items"]
        assert any(item["complaint_id"] == unique_cid for item in inc_items)

        # 4. Verify queryable via /api/incidents/{incident_id}
        detail_res = client.get(f"/api/incidents/{unique_cid}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["complaint"]["complaint_id"] == unique_cid
        assert detail_data["complaint"]["reported_amount"] == 150000.0

        # 5. Verify duplicate complaint ID returns 409
        dup_res = client.post("/api/complaints", json=payload)
        assert dup_res.status_code == 409

    finally:
        # Cleanup
        cleanup_session = get_db_session()
        try:
            cleanup_session.query(IncidentPrediction).filter(IncidentPrediction.complaint_id == unique_cid).delete()
            comp = cleanup_session.query(Complaint).filter(Complaint.complaint_id == unique_cid).first()
            if comp and comp.predicted_entity_id:
                eid = comp.predicted_entity_id
                cleanup_session.query(Complaint).filter(Complaint.complaint_id == unique_cid).delete()
                cleanup_session.query(EntityMaster).filter(EntityMaster.entity_id == eid).delete()
            else:
                cleanup_session.query(Complaint).filter(Complaint.complaint_id == unique_cid).delete()
            cleanup_session.query(AuditLog).filter(AuditLog.target_id == unique_cid).delete()
            cleanup_session.commit()
        finally:
            cleanup_session.close()


def test_upload_transactions_csv():
    """Tests POST /api/upload/transactions endpoint with a CSV stream."""
    unique_tx1 = f"TX_CSV_A_{int(time.time()) % 10000}"
    unique_tx2 = f"TX_CSV_B_{int(time.time()) % 10000}"

    csv_data = f"""transaction_id,sender_entity_id,receiver_entity_id,amount,timestamp,channel
{unique_tx1},ENT_CSV_SRC_01,ENT_CSV_DST_01,35000.50,2026-09-19 01:00:00,IMPS
{unique_tx2},ENT_CSV_SRC_02,ENT_CSV_DST_02,85000.00,2026-09-19 01:15:00,RTGS
"""

    files = {
        "file": ("bank_feed.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")
    }

    try:
        response = client.post("/api/upload/transactions", files=files)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["total_ingested"] == 2
        assert data["total_amount"] == pytest.approx(120000.50, 0.01)
        assert "alerts_triggered" in data

        # Verify transactions in SQLite
        session = get_db_session()
        try:
            tx1 = session.query(TransactionRecord).filter(TransactionRecord.transaction_id == unique_tx1).first()
            assert tx1 is not None
            assert tx1.sender_entity_id == "ENT_CSV_SRC_01"
            assert tx1.amount == 35000.50

            # Verify audit log
            audit = session.query(AuditLog).filter(AuditLog.action == "TRANSACTION_BATCH_UPLOAD").first()
            assert audit is not None
        finally:
            session.close()

        # Verify sliding window graph has edges
        assert STREAMING_ENGINE.graph.has_edge("ENT_CSV_SRC_01", "ENT_CSV_DST_01", key=unique_tx1)
        assert STREAMING_ENGINE.graph.has_edge("ENT_CSV_SRC_02", "ENT_CSV_DST_02", key=unique_tx2)

    finally:
        # Cleanup
        cleanup_session = get_db_session()
        try:
            cleanup_session.query(TransactionRecord).filter(
                TransactionRecord.transaction_id.in_([unique_tx1, unique_tx2])
            ).delete(synchronize_session=False)
            cleanup_session.commit()
        finally:
            cleanup_session.close()


def test_upload_transactions_invalid_file():
    """Tests that non-CSV file uploads are rejected with 400 Bad Request."""
    bad_files = {
        "file": ("malware.exe", io.BytesIO(b"binary data"), "application/octet-stream")
    }
    res = client.post("/api/upload/transactions", files=bad_files)
    assert res.status_code == 400


def test_ingest_transactions_batch_json():
    """Tests POST /api/ingest/transactions/batch endpoint."""
    unique_tx = f"TX_JSON_{int(time.time()) % 10000}"
    payload = {
        "transactions": [
            {
                "transaction_id": unique_tx,
                "source_entity": "ENT_BATCH_01",
                "destination_entity": "ENT_BATCH_02",
                "amount": 42000.0,
                "timestamp": time.time()
            }
        ]
    }

    try:
        response = client.post("/api/ingest/transactions/batch", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["total_ingested"] == 1
        assert data["total_amount"] == 42000.0

        # Verify DB persistence
        session = get_db_session()
        try:
            tx = session.query(TransactionRecord).filter(TransactionRecord.transaction_id == unique_tx).first()
            assert tx is not None
            assert tx.amount == 42000.0
        finally:
            session.close()

    finally:
        cleanup_session = get_db_session()
        try:
            cleanup_session.query(TransactionRecord).filter(
                TransactionRecord.transaction_id == unique_tx
            ).delete(synchronize_session=False)
            cleanup_session.commit()
        finally:
            cleanup_session.close()


def test_create_entity():
    """Tests POST /api/entities endpoint for account and ATM terminal provisioning."""
    unique_ent_id = f"ENT_NEW_{int(time.time()) % 10000}"
    unique_atm_id = f"ATM_NEW_{int(time.time()) % 10000}"

    try:
        # 1. Create account entity
        account_payload = {
            "entity_id": unique_ent_id,
            "canonical_account_number": "987654321012",
            "canonical_ifsc": "SBIN0000001",
            "canonical_holder_name": "Rohan Sharma",
            "bank_name": "State Bank of India",
            "branch_name": "Connaught Place",
            "state": "Delhi",
            "district": "New Delhi",
            "entity_type": "ACCOUNT"
        }
        res1 = client.post("/api/entities", json=account_payload)
        assert res1.status_code == 201, res1.text
        d1 = res1.json()
        assert d1["entity_id"] == unique_ent_id
        assert d1["entity_type"] == "ACCOUNT"

        # 2. Create ATM terminal entity
        atm_payload = {
            "entity_id": unique_atm_id,
            "canonical_holder_name": "SBI ATM - CP Block B",
            "bank_name": "State Bank of India",
            "state": "Delhi",
            "district": "New Delhi",
            "latitude": 28.6328,
            "longitude": 77.2197,
            "entity_type": "ATM"
        }
        res2 = client.post("/api/entities", json=atm_payload)
        assert res2.status_code == 201, res2.text
        d2 = res2.json()
        assert d2["entity_id"] == unique_atm_id
        assert d2["entity_type"] == "ATM"

        # 3. Verify in SQLite
        session = get_db_session()
        try:
            e1 = session.query(EntityMaster).filter(EntityMaster.entity_id == unique_ent_id).first()
            assert e1 is not None
            assert e1.canonical_holder_name == "Rohan Sharma"

            e2 = session.query(EntityMaster).filter(EntityMaster.entity_id == unique_atm_id).first()
            assert e2 is not None
            assert e2.entity_type == "ATM"
            assert e2.latitude == pytest.approx(28.6328, 0.0001)
        finally:
            session.close()

        # 4. Verify duplicate rejection
        dup_res = client.post("/api/entities", json=account_payload)
        assert dup_res.status_code == 409

    finally:
        cleanup_session = get_db_session()
        try:
            cleanup_session.query(EntityMaster).filter(
                EntityMaster.entity_id.in_([unique_ent_id, unique_atm_id])
            ).delete(synchronize_session=False)
            cleanup_session.query(AuditLog).filter(
                AuditLog.target_id.in_([unique_ent_id, unique_atm_id])
            ).delete(synchronize_session=False)
            cleanup_session.commit()
        finally:
            cleanup_session.close()
