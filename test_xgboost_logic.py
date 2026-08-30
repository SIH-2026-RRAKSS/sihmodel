import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
import numpy as np

def main():
    model_path = Path("models/xgboost_baseline.json")
    if not model_path.exists():
        print("XGBoost model not found!")
        return

    xgb = XGBClassifier()
    xgb.load_model(model_path)

    feature_names = xgb.feature_names_in_
    # Note: xgb.feature_importances_ gives gain usually, let's explicitly get gain
    booster = xgb.get_booster()
    gain_importance = booster.get_score(importance_type='gain')
    weight_importance = booster.get_score(importance_type='weight')
    
    print("XGBoost Top Features by GAIN:")
    # sort by gain
    sorted_gain = sorted(gain_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, gain in sorted_gain:
        print(f"{feat:<25}: {gain:.4f}")

    print("\nXGBoost Top Features by WEIGHT:")
    sorted_weight = sorted(weight_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, weight in sorted_weight:
        print(f"{feat:<25}: {weight:.4f}")

if __name__ == "__main__":
    main()
