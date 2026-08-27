"""
Synthetic Dataset Adapter (Dataset A)
=====================================
Wraps domestic-typology synthetic complaints, transactions, entity resolution,
and incident subgraphs.

Capabilities and Schema:
- Geo-coordinates: AVAILABLE (Lat/Lon for Indian cities and ATMs)
- ATM Terminals: AVAILABLE (ATM_001 - ATM_050)
- Complaint IDs: AVAILABLE (C000001 - C001000)
- IFSC Codes: AVAILABLE (Standard RBI bank IFSC formats)
- Graph Format: 72-hour 3-hop directed NetworkX MultiDiGraph incident subgraphs.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import pandas as pd

class SyntheticAdapter:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.complaints_file = self.data_dir / "complaints.csv"
        self.transactions_file = self.data_dir / "transactions.csv"
        self.entity_master_file = self.data_dir / "entity_master.csv"
        self.entity_locations_file = self.data_dir / "entity_locations.csv"
        self.graphs_dir = self.data_dir / "graphs"
        self.graph_summary_file = self.data_dir / "graph_summary.csv"

        self.has_geo_coordinates = True
        self.has_atm_terminals = True
        self.has_ifsc_codes = True
        self.has_complaint_ids = True
        self.is_temporal = True

    def load_complaints(self) -> pd.DataFrame:
        if not self.complaints_file.exists():
            raise FileNotFoundError(f"Missing complaints file: {self.complaints_file}")
        return pd.read_csv(self.complaints_file)

    def load_transactions(self) -> pd.DataFrame:
        if not self.transactions_file.exists():
            raise FileNotFoundError(f"Missing transactions file: {self.transactions_file}")
        return pd.read_csv(self.transactions_file)

    def load_entity_master(self) -> pd.DataFrame:
        if not self.entity_master_file.exists():
            raise FileNotFoundError(f"Missing entity master file: {self.entity_master_file}")
        return pd.read_csv(self.entity_master_file)

    def load_entity_locations(self) -> pd.DataFrame:
        if not self.entity_locations_file.exists():
            raise FileNotFoundError(f"Missing entity locations file: {self.entity_locations_file}")
        return pd.read_csv(self.entity_locations_file)

    def load_graph_summary(self) -> pd.DataFrame:
        if not self.graph_summary_file.exists():
            raise FileNotFoundError(f"Missing graph summary file: {self.graph_summary_file}")
        return pd.read_csv(self.graph_summary_file)

    def load_subgraph(self, complaint_id: str) -> nx.MultiDiGraph:
        graphml_path = self.graphs_dir / f"{complaint_id}.graphml"
        if not graphml_path.exists():
            raise FileNotFoundError(f"Missing GraphML file: {graphml_path}")
        return nx.read_graphml(graphml_path)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "dataset_name": "Dataset A (Synthetic Typology-Grounded)",
            "has_geo_coordinates": self.has_geo_coordinates,
            "has_atm_terminals": self.has_atm_terminals,
            "has_ifsc_codes": self.has_ifsc_codes,
            "has_complaint_ids": self.has_complaint_ids,
            "is_temporal": self.is_temporal,
            "supported_stages": [
                "Stage 0 (Entity Resolution)",
                "Stage 1 (Synthetic Transactions)",
                "Stage 2 (Incident Graph Construction)",
                "Stage 3A (XGBoost Baseline)",
                "Stage 3B (GraphSAGE GNN)",
                "Stage 4 (Terminal Node Prediction)",
                "Stage 5 (Confidence Tiers and Novelty)",
                "Stage 6 (Explainability Layer)",
                "Stage 7 (Alert Threshold Policy)",
                "Stage 10 (Interactive Dashboard)"
            ]
        }
