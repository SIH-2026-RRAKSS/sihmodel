import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.api import get_incident_detail, list_incidents

def run_test():
    try:
        # Checkpoint 3 API response
        res = get_incident_detail("C000014")
        print("Detail API Response:")
        print(f"Risk Probability: {res['model_prediction']['graphsage_risk_probability']}")
        print(f"Confidence Tier: {res['model_prediction']['confidence_tier']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_test()
