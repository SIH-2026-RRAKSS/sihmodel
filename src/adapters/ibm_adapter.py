"""
IBM AML Dataset Adapter (Dataset B)
===================================
Loads the IBM Transactions for Anti-Money Laundering dataset (HI-Small_Trans.csv).
Builds bank-to-bank transaction graphs with laundering labels.

LIMITATIONS AND UNAVAILABLE FIELDS:
- Geo-coordinates: UNAVAILABLE (No physical latitude/longitude recorded).
- ATM Terminals: UNAVAILABLE (Bank-to-bank / payment formats only; no ATM terminal IDs).
- Complaint IDs: UNAVAILABLE (Transactions are bank ledger records, not citizen complaints).
- IFSC Codes: UNAVAILABLE (Bank IDs are integers, e.g. 10, 3208).

DOWNSTREAM PIPELINE IMPACT:
- Stage 2 Geo-Cluster Estimation (k-NN) CANNOT RUN on IBM data (explicitly disabled).
- Stage 4 Terminal Prediction evaluates account-level cash exit/outflow dynamics (out-degree ~ 0,
  large single outflows, account age), NOT physical ATM coordinates.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import networkx as nx

class IBMAMLAdapter:
    def __init__(self, file_path: Optional[str] = None):
        if file_path is None:
            cache_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "ealtman2019" / "ibm-transactions-for-anti-money-laundering-aml" / "versions" / "8" / "HI-Small_Trans.csv"
            if cache_path.exists():
                self.file_path = cache_path
            else:
                self.file_path = Path("data/ibm_aml/HI-Small_Trans.csv")
        else:
            self.file_path = Path(file_path)

        self.has_geo_coordinates = False
        self.has_atm_terminals = False
        self.has_ifsc_codes = False
        self.has_complaint_ids = False
        self.is_temporal = True

    def load_raw_transactions(self, nrows: Optional[int] = None) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Missing IBM AML dataset file at: {self.file_path}")
        return pd.read_csv(self.file_path, nrows=nrows)

    def build_transaction_dataframe(self, nrows: Optional[int] = 500000) -> pd.DataFrame:
        df_raw = self.load_raw_transactions(nrows=nrows)
        
        df = pd.DataFrame()
        df["transaction_id"] = [f"IBM_TX_{i:07d}" for i in range(len(df_raw))]
        df["from_entity_id"] = "BANK_" + df_raw["From Bank"].astype(str) + "_ACC_" + df_raw["Account"].astype(str)
        df["to_entity_id"] = "BANK_" + df_raw["To Bank"].astype(str) + "_ACC_" + df_raw["Account.1"].astype(str)
        df["timestamp"] = pd.to_datetime(df_raw["Timestamp"])
        df["amount"] = df_raw["Amount Paid"].astype(float)
        df["payment_currency"] = df_raw["Payment Currency"]
        df["payment_format"] = df_raw["Payment Format"]
        df["is_laundering"] = df_raw["Is Laundering"].astype(int)
        
        df["target_atm_id"] = None
        df["latitude"] = np.nan
        df["longitude"] = np.nan
        df["complaint_id"] = None
        df["source_ifsc"] = None
        df["dest_ifsc"] = None
        
        return df

    def extract_terminal_ranking_features(self, df_tx: pd.DataFrame) -> pd.DataFrame:
        inflow = df_tx.groupby("to_entity_id").agg(
            in_degree=("transaction_id", "count"),
            total_inflow=("amount", "sum"),
            max_inflow=("amount", "max"),
            has_laundering_in=("is_laundering", "max")
        ).reset_index().rename(columns={"to_entity_id": "account_id"})

        outflow = df_tx.groupby("from_entity_id").agg(
            out_degree=("transaction_id", "count"),
            total_outflow=("amount", "sum"),
            max_outflow=("amount", "max"),
            has_laundering_out=("is_laundering", "max")
        ).reset_index().rename(columns={"from_entity_id": "account_id"})

        merged = pd.merge(inflow, outflow, on="account_id", how="outer").fillna(0)
        merged["net_retained"] = merged["total_inflow"] - merged["total_outflow"]
        merged["is_terminal_exit"] = (merged["out_degree"] == 0) & (merged["in_degree"] > 0)
        merged["is_laundering"] = ((merged["has_laundering_in"] > 0) | (merged["has_laundering_out"] > 0)).astype(int)
        
        return merged

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "dataset_name": "Dataset B (IBM AML Bank-to-Bank)",
            "has_geo_coordinates": self.has_geo_coordinates,
            "has_atm_terminals": self.has_atm_terminals,
            "has_ifsc_codes": self.has_ifsc_codes,
            "has_complaint_ids": self.has_complaint_ids,
            "is_temporal": self.is_temporal,
            "supported_stages": [
                "Stage 3A (XGBoost Baseline on Graph Metrics)",
                "Stage 4 (Terminal Account Exit Ranking)"
            ],
            "unsupported_stages": [
                "Stage 0 (Entity Resolution on Complaints)",
                "Stage 2 (Geo-Cluster Estimation)",
                "Stage 10 (Geo Heatmap Dashboard)"
            ],
            "unsupported_reason": "IBM dataset contains bank-to-bank ledger transfers without citizen complaints, IFSC codes, or physical ATM GPS coordinates."
        }
