"""
Database Persistence & Relational Schema Layer
==============================================
Provides SQLite / SQLAlchemy ORM models, migration, indexing, and seed utilities
for the Cybercrime Predictive Analytics platform.

Tables:
1. complaints: Registered cybercrime complaints & entity mappings.
2. entity_master: Master bank accounts and ATM terminal hardware profiles.
3. transactions: 15,000+ transactional payment records with timestamps and amounts.
4. incident_predictions: Stored GraphSAGE risk probabilities, confidence tiers, and terminal locations.
5. audit_logs: Timestamped operational and triage actions.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "cybercrime_aml.db"

Base = declarative_base()


# ==============================================================================
# SQLAlchemy ORM Models
# ==============================================================================

class EntityMaster(Base):
    """Canonical financial entity (Account or ATM Terminal)."""
    __tablename__ = "entity_master"

    entity_id = Column(String(50), primary_key=True, index=True)
    canonical_account_number = Column(String(50), index=True)
    canonical_ifsc = Column(String(20), index=True)
    canonical_holder_name = Column(String(100))
    bank_name = Column(String(100))
    branch_name = Column(String(100))
    state = Column(String(50))
    district = Column(String(50))
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    entity_type = Column(String(20), default="ACCOUNT")  # ACCOUNT or ATM

    # Relationships
    complaints = relationship("Complaint", back_populates="resolved_entity")


class Complaint(Base):
    """Cybercrime complaint filed across police stations."""
    __tablename__ = "complaints"

    complaint_id = Column(String(50), primary_key=True, index=True)
    complaint_date = Column(String(50))
    complainant_name = Column(String(100), nullable=True)
    police_station_id = Column(String(50), index=True, nullable=True)
    district = Column(String(50))
    state = Column(String(50))
    reported_account_number = Column(String(50), index=True)
    reported_ifsc = Column(String(20), index=True)
    reported_amount = Column(Float)
    scam_category = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    predicted_entity_id = Column(String(50), ForeignKey("entity_master.entity_id"), nullable=True, index=True)

    # Relationships
    resolved_entity = relationship("EntityMaster", back_populates="complaints")
    predictions = relationship("IncidentPrediction", back_populates="complaint")


class TransactionRecord(Base):
    """Financial transaction between entities or cash-out to ATM."""
    __tablename__ = "transactions"

    transaction_id = Column(String(50), primary_key=True, index=True)
    sender_entity_id = Column(String(50), index=True)
    receiver_entity_id = Column(String(50), index=True)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    transaction_type = Column(String(30))  # IMPS, NEFT, RTGS, UPI, ATM_WITHDRAWAL
    channel = Column(String(30))
    is_cash_out = Column(Boolean, default=False)
    is_suspicious_ground_truth = Column(Boolean, default=False)
    ring_id_ground_truth = Column(String(50), nullable=True)


class IncidentPrediction(Base):
    """Stored model inference result for an incident subgraph."""
    __tablename__ = "incident_predictions"

    incident_id = Column(String(50), primary_key=True, index=True)
    complaint_id = Column(String(50), ForeignKey("complaints.complaint_id"), nullable=True, index=True)
    graphsage_risk_probability = Column(Float)
    confidence_tier = Column(String(50))  # HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, NORMAL, FIRST_TIME_RING_CANDIDATE
    top_terminal_id = Column(String(50), nullable=True)
    top_terminal_score = Column(Float, nullable=True)
    top_terminal_city = Column(String(50), nullable=True)
    num_nodes = Column(Integer)
    num_edges = Column(Integer)
    executive_summary = Column(Text, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    complaint = relationship("Complaint", back_populates="predictions")


class AuditLog(Base):
    """Operational audit log for actions taken by investigators."""
    __tablename__ = "audit_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(String(50), default="ANALYST_01")
    action = Column(String(100))
    target_id = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)


# Indexes for multi-hop graph neighbor queries
Index("idx_tx_sender_time", TransactionRecord.sender_entity_id, TransactionRecord.timestamp)
Index("idx_tx_receiver_time", TransactionRecord.receiver_entity_id, TransactionRecord.timestamp)


# ==============================================================================
# Database Helper Functions
# ==============================================================================

def get_engine(db_path: Path = DEFAULT_DB_PATH):
    """Creates a SQLite engine with multi-threading support."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


def get_db_session(db_path: Path = DEFAULT_DB_PATH) -> Session:
    """Returns a new database session."""
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Creates all database tables and indexes."""
    engine = get_engine(db_path)
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Database initialized successfully at: {db_path}")


def seed_database_from_csv(data_dir: Path = DATA_DIR, db_path: Path = DEFAULT_DB_PATH) -> Dict[str, int]:
    """Loads existing CSV datasets into SQLite database."""
    # Ensure fresh schema
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    init_db(db_path)
    session = get_db_session(db_path)
    counts = {}

    try:
        # 1. Entity Master
        master_file = data_dir / "entity_master.csv"
        locations_file = data_dir / "entity_locations.csv"
        if master_file.exists():
            df_master = pd.read_csv(master_file)
            loc_dict = {}
            if locations_file.exists():
                df_loc = pd.read_csv(locations_file)
                for r in df_loc.to_dict(orient="records"):
                    loc_dict[r["entity_id"]] = {
                        "lat": r.get("latitude"),
                        "lon": r.get("longitude"),
                        "state": r.get("state", ""),
                        "city": r.get("city", "")
                    }

            entities_to_add = []
            for r in df_master.to_dict(orient="records"):
                eid = r["entity_id"]
                loc_data = loc_dict.get(eid, {})
                lat = loc_data.get("lat")
                lon = loc_data.get("lon")
                l_state = loc_data.get("state", "")
                l_city = loc_data.get("city", "")
                etype = "ATM" if str(eid).startswith("ATM_") else "ACCOUNT"
                entities_to_add.append(EntityMaster(
                    entity_id=eid,
                    canonical_account_number=str(r.get("account_number", "")),
                    canonical_ifsc=str(r.get("ifsc", "")),
                    canonical_holder_name=str(r.get("canonical_name", "")),
                    bank_name=str(r.get("bank_name", "")),
                    branch_name=str(r.get("branch_name", "")),
                    state=str(r.get("state", "")) or l_state,
                    district=str(r.get("district", "")) or l_city,
                    latitude=float(lat) if pd.notna(lat) and lat is not None else None,
                    longitude=float(lon) if pd.notna(lon) and lon is not None else None,
                    entity_type=etype
                ))
            session.bulk_save_objects(entities_to_add)
            session.commit()
            counts["entities"] = len(entities_to_add)

        # 2. Complaints
        complaints_file = data_dir / "complaints.csv"
        resolved_file = data_dir / "resolved_entities.csv"
        if complaints_file.exists():
            df_comp = pd.read_csv(complaints_file)
            res_dict = {}
            if resolved_file.exists():
                df_res = pd.read_csv(resolved_file)
                res_dict = dict(zip(df_res["complaint_id"], df_res["predicted_entity_id"]))

            complaints_to_add = []
            for r in df_comp.to_dict(orient="records"):
                cid = r["complaint_id"]
                holder_name = r.get("account_holder_name") or r.get("complainant_name", "")
                acc_num = r.get("account_number") or r.get("reported_account_number", "")
                ifsc_code = r.get("ifsc") or r.get("reported_ifsc", "")
                scam = r.get("complaint_type") or r.get("scam_category", "")
                complaints_to_add.append(Complaint(
                    complaint_id=cid,
                    complaint_date=str(r.get("complaint_date", "")),
                    complainant_name=str(holder_name),
                    police_station_id=str(r.get("police_station_id", "PS_DEFAULT")),
                    district=str(r.get("district", "")),
                    state=str(r.get("state", "")),
                    reported_account_number=str(acc_num),
                    reported_ifsc=str(ifsc_code),
                    reported_amount=float(r.get("reported_amount", 0.0)),
                    scam_category=str(scam),
                    description=str(r.get("description", "")),
                    predicted_entity_id=res_dict.get(cid)
                ))
            session.bulk_save_objects(complaints_to_add)
            session.commit()
            counts["complaints"] = len(complaints_to_add)

        # 3. Transactions
        tx_file = data_dir / "transactions.csv"
        if tx_file.exists():
            df_tx = pd.read_csv(tx_file)
            txs_to_add = []
            for r in df_tx.to_dict(orient="records"):
                dt = pd.to_datetime(r["timestamp"])
                src = r.get("sender_entity_id") or r.get("source_entity_id")
                dst = r.get("receiver_entity_id") or r.get("destination_entity_id")
                txs_to_add.append(TransactionRecord(
                    transaction_id=str(r["transaction_id"]),
                    sender_entity_id=str(src),
                    receiver_entity_id=str(dst),
                    amount=float(r["amount"]),
                    timestamp=dt,
                    transaction_type=str(r.get("transaction_type", "NEFT")),
                    channel=str(r.get("channel", "INTERNET_BANKING")),
                    is_cash_out=bool(r.get("is_cash_out", 0)),
                    is_suspicious_ground_truth=bool(r.get("is_suspicious", False)),
                    ring_id_ground_truth=str(r.get("ring_id")) if pd.notna(r.get("ring_id")) else None
                ))
            session.bulk_save_objects(txs_to_add)
            session.commit()
            counts["transactions"] = len(txs_to_add)

        # 4. Predictions & Confidence Tiers
        tiers_file = data_dir / "confidence_tiers.csv"
        graph_summary_file = data_dir / "graph_summary.csv"
        exp_file = data_dir / "explanations.csv"

        if tiers_file.exists():
            df_tiers = pd.read_csv(tiers_file)
            graph_dict = {}
            if graph_summary_file.exists():
                df_gs = pd.read_csv(graph_summary_file)
                for r in df_gs.to_dict(orient="records"):
                    graph_dict[r["complaint_id"]] = (r.get("num_nodes", 0), r.get("num_edges", 0))

            exp_dict = {}
            if exp_file.exists():
                df_exp = pd.read_csv(exp_file)
                for r in df_exp.to_dict(orient="records"):
                    exp_dict[r["complaint_id"]] = r.get("investigator_summary", "")

            preds_to_add = []
            for r in df_tiers.to_dict(orient="records"):
                cid = r["complaint_id"]
                prob = float(r.get("graphsage_probability") or r.get("graphsage_risk_probability") or 0.0)
                tier = str(r.get("confidence_tier", "NORMAL"))
                t_id = str(r.get("top_terminal", "NONE"))
                t_city = str(r.get("top_terminal_city", "NONE"))
                t_score = float(r.get("terminal_score", 0.0))
                n_nodes, n_edges = graph_dict.get(cid, (1, 0))
                summary = exp_dict.get(cid, "")

                preds_to_add.append(IncidentPrediction(
                    incident_id=cid,
                    complaint_id=cid,
                    graphsage_risk_probability=prob,
                    confidence_tier=tier,
                    top_terminal_id=t_id if t_id != "NONE" else None,
                    top_terminal_score=t_score if t_score > 0 else None,
                    top_terminal_city=t_city if t_city != "NONE" else None,
                    num_nodes=n_nodes,
                    num_edges=n_edges,
                    executive_summary=summary
                ))
            session.bulk_save_objects(preds_to_add)
            session.commit()
            counts["predictions"] = len(preds_to_add)

        # 5. Initial Audit Log
        session.add(AuditLog(
            action="DATABASE_SEED",
            details=f"Seeded {counts.get('entities', 0)} entities, {counts.get('complaints', 0)} complaints, {counts.get('transactions', 0)} txs."
        ))
        session.commit()

        print(f"[DB] Database seeding complete: {counts}")
        return counts

    finally:
        session.close()


def log_action(action: str, target_id: Optional[str] = None, details: Optional[str] = None, user_id: str = "SYSTEM", db_path: Path = DEFAULT_DB_PATH):
    """Records an entry in the operational audit trail."""
    session = get_db_session(db_path)
    try:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            target_id=target_id,
            details=details
        )
        session.add(entry)
        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    print("=== Cybercrime AML Database Initialization ===")
    res = seed_database_from_csv()
    print("Completed with summary:", res)
