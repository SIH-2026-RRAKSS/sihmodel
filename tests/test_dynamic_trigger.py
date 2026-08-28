import pytest
import datetime
from typing import Any, Dict
from src.streaming_engine import TemporalTransactionGraph, DynamicAnomalyTrigger

def create_tx(src: str, amount: float, ts_str: str) -> Dict[str, Any]:
    return {
        "transaction_id": f"TX_{int(datetime.datetime.now().timestamp() * 1000)}",
        "sender_entity_id": src,
        "receiver_entity_id": "DST_1",
        "amount": amount,
        "timestamp": ts_str,
        "channel": "TRANSFER",
        "is_cash_out": 0
    }

def test_normal_flow_negative_gate():
    trigger = DynamicAnomalyTrigger()
    
    # Feed 10 transactions around 2000
    for i in range(10):
        tx = create_tx("ACC_NORMAL", 2000.0 + (i * 10), "2024-01-01 10:00:00")
        is_triggered, reason = trigger.evaluate_transaction(tx)
        assert is_triggered is False, "Normal transaction should not trigger Stage 1"
        
def test_anomaly_spike_positive_gate():
    trigger = DynamicAnomalyTrigger()
    
    # Profile account with mean ~2000, std ~500
    amounts = [1500, 2000, 2500, 1800, 2200, 2000, 2100, 1900, 1700, 2300]
    for amt in amounts:
        tx = create_tx("ACC_SPIKE", amt, "2024-01-01 10:00:00")
        trigger.evaluate_transaction(tx)
        
    # Inject outlier of 75,000
    tx_outlier = create_tx("ACC_SPIKE", 75000.0, "2024-01-02 10:00:00")
    is_triggered, reason = trigger.evaluate_transaction(tx_outlier)
    assert is_triggered is True, "Outlier should trigger Stage 1"
    assert "SINGLE_TX_OUTLIER" in reason, f"Unexpected reason: {reason}"
    assert "Z-Score" in reason, f"Reason doesn't contain Z-Score: {reason}"

def test_proactive_gnn_execution(monkeypatch):
    engine = TemporalTransactionGraph(window_hours=72, max_hops=3)
    
    # Mock the extract and score methods to avoid loading ML models during unit tests
    def mock_extract(*args, **kwargs):
        return "mock_subgraph"
        
    def mock_score(*args, **kwargs):
        return {
            "risk_probability": 0.85, # Over 0.70 threshold
            "confidence_tier": "HIGH_CONFIDENCE",
            "is_suspicious": True,
            "num_nodes": 5,
            "num_edges": 10,
            "mule_probabilities": {"ACC_MOCK": 0.95},
            "terminals": []
        }
        
    monkeypatch.setattr(engine, "extract_subgraph_around_entity", mock_extract)
    monkeypatch.setattr(engine, "score_subgraph_live", mock_score)
    
    # Warm up account with 1 transaction
    tx_warmup = create_tx("ACC_TEST_GNN", 1000.0, "2024-01-01 10:00:00")
    engine.ingest_transaction(tx_warmup)
    
    # Send outlier
    tx_outlier = create_tx("ACC_TEST_GNN", 100000.0, "2024-01-01 11:00:00")
    engine.ingest_transaction(tx_outlier)
    
    assert engine.trigger_count >= 1, "Engine should have registered a trigger"
    assert len(engine.proactive_alerts) == 1, "One proactive alert should be generated"
    
    alert = engine.proactive_alerts[0]
    assert alert["seed_entity"] == "ACC_TEST_GNN"
    assert alert["gnn_risk"] == 0.85
    assert "ACC_MOCK" in alert["mules"]
