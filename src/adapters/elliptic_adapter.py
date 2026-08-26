"""
Elliptic Bitcoin Dataset Adapter (Dataset C)
============================================
Loads the Elliptic Bitcoin Transaction Graph via PyTorch Geometric.
Strictly isolated for GraphSAGE GNN architecture validation (node classification).

LIMITATIONS AND STRUCTURAL DIFFERENCES:
- Domain: Bitcoin transaction DAG (UTXO-based), NOT bank accounts or citizen complaints.
- Geo-coordinates: UNAVAILABLE.
- ATM Terminals: UNAVAILABLE.
- Bank IDs / IFSC: UNAVAILABLE.
- Task Structure: Node-level classification (Class 1 = Illicit, Class 0 = Licit, Class 2/Unknown = Unlabeled),
  whereas Dataset A is Graph-level / Subgraph-level classification.

GUARDRAIL NOTICE:
- Elliptic metrics are reported SEPARATELY from Dataset A and B.
- Never blended into a single accuracy/F1 claim.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import pandas as pd
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.data import Data

class EllipticAdapter:
    def __init__(self, root_dir: str = "data/elliptic"):
        self.root_dir = Path(root_dir)
        self.dataset = EllipticBitcoinDataset(root=str(self.root_dir))
        
        self.has_geo_coordinates = False
        self.has_atm_terminals = False
        self.has_ifsc_codes = False
        self.has_bank_accounts = False
        self.is_node_classification_only = True

    def load_data(self) -> Data:
        return self.dataset[0]

    def get_train_test_split(self, split_timestep: int = 34) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Splits labeled nodes temporally:
        Train: timesteps 1..split_timestep (standard benchmark: 1..34)
        Test:  timesteps (split_timestep+1)..49 (standard benchmark: 35..49)
        Target: 1 = Illicit (4,545 nodes), 0 = Licit (42,019 nodes)
        Unknowns (157,205 nodes) are excluded from loss evaluation.
        """
        data = self.dataset[0]
        
        # Load raw timesteps
        feat_csv = self.root_dir / "raw" / "elliptic_txs_features.csv"
        if feat_csv.exists():
            df_feat = pd.read_csv(feat_csv, header=None, usecols=[1])
            timesteps = torch.tensor(df_feat.iloc[:, 0].values, dtype=torch.long)
        else:
            # Fallback if raw csv is cleaned
            timesteps = torch.ones(data.num_nodes, dtype=torch.long)
            
        # PyG mapping: y=1 is Illicit, y=0 is Licit, y=2 is Unknown
        labeled_mask = (data.y == 0) | (data.y == 1)
        
        train_mask = labeled_mask & (timesteps <= split_timestep)
        test_mask = labeled_mask & (timesteps > split_timestep)
        
        data.binary_y = data.y.clone()
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.timesteps = timesteps
        
        return data, train_mask, test_mask

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "dataset_name": "Dataset C (Elliptic Bitcoin Transaction Graph)",
            "total_nodes": 203769,
            "total_edges": 234355,
            "feature_dim": 165,
            "labeled_nodes": 46564,
            "illicit_nodes": 4545,
            "licit_nodes": 42019,
            "unlabeled_nodes": 157205,
            "has_geo_coordinates": self.has_geo_coordinates,
            "has_atm_terminals": self.has_atm_terminals,
            "has_ifsc_codes": self.has_ifsc_codes,
            "has_bank_accounts": self.has_bank_accounts,
            "is_node_classification_only": self.is_node_classification_only,
            "supported_stages": [
                "Stage 3B Architecture Validation (Inductive GraphSAGE on Node Classification)"
            ],
            "unsupported_stages": [
                "Stage 0 (Entity Resolution)",
                "Stage 2 (Geo-Cluster Estimation)",
                "Stage 4 (ATM Terminal Prediction)",
                "Stage 10 (Heatmap Dashboard)"
            ],
            "unsupported_reason": "Bitcoin transaction graph structure; no fiat bank entities, complaints, or physical cash-out ATM locations."
        }
