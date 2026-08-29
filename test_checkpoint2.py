import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.database import get_db_session, Complaint
from src.api import predict_live_subgraph, LivePredictRequest

def run_test():
    session = get_db_session()
    comp = session.query(Complaint).filter(Complaint.complaint_id == "C000014").first()
    if comp is None:
        print("Complaint not found!")
        return
    print(f"Complaint C000014 entity: {comp.predicted_entity_id}")
    
    req = LivePredictRequest(seed_entity_id=comp.predicted_entity_id, max_hops=3)
    try:
        res = predict_live_subgraph(req)
        print("Live prediction result:")
        print(f"Risk Probability: {res.risk_probability}")
        print(f"Confidence Tier: {res.confidence_tier}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_test()
