import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, f1_score
from torch_geometric.nn import SAGEConv, GINConv
import torch_geometric.utils as pyg_utils

from src.adapters.elliptic_adapter import EllipticAdapter

def get_opt(y, prob):
    p, r, t = precision_recall_curve(y, prob)
    f1s = 2 * (p * r) / (p + r + 1e-9)
    best_idx = np.argmax(f1s)
    return t[best_idx] if best_idx < len(t) else 0.50

class BaseSAGE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, 64)
        self.clf = nn.Linear(64, 1)
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.clf(x)

class FatSAGE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 256)
        self.conv2 = SAGEConv(256, 128)
        self.clf = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.clf(x)

class GIN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, 128), nn.ReLU(), nn.Linear(128, 128)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64)))
        self.clf = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.clf(x)

def run_ablation():
    seeds = [42, 101, 2024, 7, 99]
    adapter = EllipticAdapter()
    data, _, _ = adapter.get_train_test_split(split_timestep=34)
    feat_csv = adapter.root_dir / "raw" / "elliptic_txs_features.csv"
    df_feat = pd.read_csv(feat_csv, header=None, usecols=[1])
    timesteps = torch.tensor(df_feat.iloc[:, 0].values, dtype=torch.long)
    
    labeled = (data.y == 0) | (data.y == 1)
    tm_all = labeled & (timesteps <= 29)
    vm = labeled & (timesteps > 29) & (timesteps <= 34)
    tem = labeled & (timesteps > 34)
    
    Y_C = data.binary_y.numpy()
    pw_t = torch.tensor([float(np.sum(Y_C[tm_all] == 0) / max(1, np.sum(Y_C[tm_all] == 1)))])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    
    # Pre-compute undirected edge index
    edge_index_undirected = pyg_utils.to_undirected(data.edge_index)
    
    results = {'BaseSAGE': [], 'FatSAGE': [], 'GIN': [], 'Undirected_BaseSAGE': [], 'Undirected_GIN': []}
    
    for s in seeds:
        print(f"\nSeed {s}...")
        torch.manual_seed(s)
        
        models = {
            'BaseSAGE': (BaseSAGE(data.x.size(1)), data.edge_index),
            'FatSAGE': (FatSAGE(data.x.size(1)), data.edge_index),
            'GIN': (GIN(data.x.size(1)), data.edge_index),
            'Undirected_BaseSAGE': (BaseSAGE(data.x.size(1)), edge_index_undirected),
            'Undirected_GIN': (GIN(data.x.size(1)), edge_index_undirected)
        }
        
        for name, (mdl, e_idx) in models.items():
            opt = torch.optim.Adam(mdl.parameters(), lr=0.003, weight_decay=1e-4)
            bf, bs = 0, None
            for _ in range(40):
                mdl.train()
                opt.zero_grad()
                out = mdl(data.x, e_idx).squeeze(-1)
                crit(out[tm_all], data.binary_y.float()[tm_all]).backward()
                opt.step()
                mdl.eval()
                with torch.no_grad():
                    val_out = mdl(data.x, e_idx).squeeze(-1)
                    val_prob = torch.sigmoid(val_out[vm]).numpy()
                vf1 = f1_score(Y_C[vm], val_prob>=0.5)
                if vf1 >= bf: bf, bs = vf1, copy.deepcopy(mdl.state_dict())
            
            mdl.load_state_dict(bs)
            mdl.eval()
            with torch.no_grad():
                full_prob = torch.sigmoid(mdl(data.x, e_idx).squeeze(-1)).numpy()
            
            opt_t = get_opt(Y_C[vm], full_prob[vm])
            test_f1 = f1_score(Y_C[tem], full_prob[tem]>=opt_t)
            results[name].append(test_f1)
            print(f"  {name}: {test_f1*100:.2f}%")
            
    print("\nFINAL RESULTS (5 seeds):")
    for name, arr in results.items():
        arr = np.array(arr) * 100
        print(f"{name:20s}: {np.mean(arr):.2f}±{np.std(arr):.2f}%")

if __name__ == "__main__":
    run_ablation()
