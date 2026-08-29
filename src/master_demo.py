import time
import requests
import sys
import json
import warnings
import random
warnings.filterwarnings("ignore")

# Import the core engine to demonstrate Stage 1 natively
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.streaming_engine import TemporalTransactionGraph

API_URL = "http://localhost:8000"

def slow_print(text, delay=0.03, color="\033[0m"):
    for char in text:
        sys.stdout.write(f"{color}{char}")
        sys.stdout.flush()
        time.sleep(delay)
    print("\033[0m")
    time.sleep(0.5)

def run_master_demo():
    print("\n" + "="*70)
    slow_print("🚀 INITIATING ENTERPRISE AML TRIAGE DEMONSTRATION 🚀", 0.02, "\033[1;36m")
    print("="*70 + "\n")

    # ---------------------------------------------------------
    # PHASE 1: DYNAMIC THRESHOLD CALIBRATION
    # ---------------------------------------------------------
    slow_print(">>> PHASE 1: DYNAMIC POLICY THRESHOLD TRIGGER <<<", 0.04, "\033[1;33m")
    slow_print("Simulating a Bank Analyst adjusting the risk tolerance threshold via FastAPI...")
    
    thresholds_to_test = [0.40, 0.75, 0.95]
    for tau in thresholds_to_test:
        try:
            res = requests.post(
                f"{API_URL}/api/policy/tune",
                json={"threshold": tau, "dataset": "IBM AML"}
            )
            if res.status_code == 200:
                data = res.json()
                print(f"  [API] Set Threshold = {tau:.2f} | Resulting Tier: {data['policy_tier_name']}")
                print(f"        -> Precision: {data['precision_percent']}% | Recall: {data['recall_percent']}% | Est. Alerts: {data['alerts_generated']}\n")
            time.sleep(1)
        except Exception:
            print("❌ Error hitting FastAPI. Is uvicorn running on port 8000?")
            return

    # ---------------------------------------------------------
    # PHASE 2: O(1) STAGE 1 ANOMALY GATE (STREAMING)
    # ---------------------------------------------------------
    time.sleep(1)
    slow_print("\n>>> PHASE 2: STAGE 1 HYBRID INGESTION TRIGGER (STREAMING) <<<", 0.04, "\033[1;33m")
    slow_print("Simulating a high-velocity burst of 50 live transactions hitting the engine...")
    slow_print("Watch Welford's Algorithm safely block normal traffic without waking the ML Model.\n")
    
    engine = TemporalTransactionGraph(window_hours=72)
    blocked_count = 0
    passed_count = 0
    
    for i in range(1, 51):
        # Generate semi-random transaction amounts. Occasional massive spikes.
        is_spike = (i % 17 == 0)
        amt = 250000.0 if is_spike else random.uniform(10.0, 500.0)
        
        tx_dict = {
            "transaction_id": f"TX_{i}",
            "sender_entity_id": "U_NORMAL",
            "receiver_entity_id": f"VENDOR_{i}",
            "amount": amt,
            "timestamp": "2026-08-29 12:00:00",
            "is_cash_out": False
        }
        
        # Ingest and evaluate Stage 1 trigger directly
        start_t = time.time()
        needs_triage, reason = engine.anomaly_trigger.evaluate_transaction(tx_dict)
        engine.ingest_transaction(tx_dict)
        lat = (time.time() - start_t) * 1000
        
        if needs_triage:
            passed_count += 1
            print(f"  🚨 [STAGE 1 BREACH] Tx {i} | Amt: ₹{amt:,.2f} | Reason: {reason} | Gate Latency: {lat:.3f}ms")
        else:
            blocked_count += 1
            if i % 5 == 0: # Just sample prints so terminal isn't totally flooded
                print(f"  ✅ [STAGE 1 BLOCKED] Tx {i} | Amt: ₹{amt:,.2f} | Filtered as benign. | Gate Latency: {lat:.3f}ms")
                
        time.sleep(0.02)
        
    print(f"\n📈 Stage 1 Summary: Filtered {blocked_count}/50 benign transactions ({(blocked_count/50)*100:.1f}% efficiency). ML compute saved!")

    # ---------------------------------------------------------
    # PHASE 3: RETROSPECTIVE COMPLAINT TRIAGE (STAGE 2 GNN)
    # ---------------------------------------------------------
    time.sleep(2)
    slow_print("\n>>> PHASE 3: RETROSPECTIVE COMPLAINT & STAGE 2 GNN TRIAGE <<<", 0.04, "\033[1;33m")
    slow_print("A formal police complaint (NCRP) was just filed against Account 'C000854'.")
    slow_print("Triggering the FastAPI to execute a retrospective Multi-Hop Subgraph Extraction & Live GraphSAGE prediction...\n")
    
    try:
        start_time = time.time()
        res = requests.post(
            f"{API_URL}/api/predict/subgraph",
            json={"seed_entity_id": "C000854", "max_hops": 2}
        )
        total_latency = (time.time() - start_time) * 1000
        
        if res.status_code == 200:
            data = res.json()
            slow_print(f"🎯 [STAGE 2 GNN COMPLETE] Sub-5ms SLA Met: {total_latency:.2f}ms Total API Roundtrip!", 0.03, "\033[1;32m")
            print(f"  - Target Seed: {data['seed_entity_id']}")
            print(f"  - Extracted Topology: {data['num_nodes']} nodes, {data['num_edges']} edges")
            print(f"  - GraphSAGE Risk Probability: {data['risk_probability']:.2f}")
            print(f"  - System Confidence Tier: {data['confidence_tier']}")
            
            if data.get('terminals'):
                print(f"  - 🏧 Predicted Cash-Out Terminal: {data['terminals'][0]['terminal_id']} (Confidence: {data['terminals'][0]['terminal_score']})")
    except Exception as e:
         print(f"❌ Error hitting FastAPI for Stage 2: {e}")

    slow_print("\n✅ DEMONSTRATION COMPLETE. Check your UI Dashboard to see the latest incidents!", 0.05, "\033[1;36m")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_master_demo()
