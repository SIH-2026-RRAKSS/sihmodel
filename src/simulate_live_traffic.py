import time
import random
import requests

API_URL = "http://localhost:8000"

def simulate_traffic():
    print("🚀 Initiating Live Cybercrime Attack Simulation...")
    print(f"📡 Target API: {API_URL}")
    print("---")
    
    try:
        requests.get(f"{API_URL}/api/health")
    except Exception:
        print("❌ ERROR: FastAPI is not running! Please start it with: uvicorn src.api:app --port 8000")
        return

    seed_targets = [
        "C000003", "C000014", "C000854", "C000913", "C000022",
        "C000880", "C000971", "C000860", "C000937", "C000958"
    ]
    
    attack_count = 0
    try:
        while True:
            target = random.choice(seed_targets)
            start_time = time.time()
            try:
                res = requests.post(
                    f"{API_URL}/api/predict/subgraph", 
                    json={"seed_entity_id": target, "max_hops": 2},
                    timeout=2
                )
                latency_ms = (time.time() - start_time) * 1000
                
                if res.status_code == 200:
                    data = res.json()
                    is_suspicious = data.get("is_suspicious", False)
                    risk = data.get("risk_probability", 0.0)
                    tier = data.get("confidence_tier", "UNKNOWN")
                    
                    if is_suspicious:
                        print(f"🚨 [ALERT] Entity {target} | Risk: {risk:.2f} | Tier: {tier} | Latency: {latency_ms:.2f}ms")
                    else:
                        print(f"✅ [CLEAN] Entity {target} | Risk: {risk:.2f} | Latency: {latency_ms:.2f}ms")
            except Exception as e:
                print(f"⚠️ Request failed: {e}")
                
            attack_count += 1
            time.sleep(random.uniform(0.1, 1.5))
            
    except KeyboardInterrupt:
        print(f"\n🛑 Simulation Stopped. Total attacks simulated: {attack_count}")

if __name__ == "__main__":
    simulate_traffic()
