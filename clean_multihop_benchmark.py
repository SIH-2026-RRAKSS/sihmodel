import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from pathlib import Path

from src.graphsage_classifier import DualHeadGraphSAGE, load_all_graphs_dataset
from src.ibm_graphsage_classifier import load_or_create_ibm_pyg_dataset, TARGET_COL, IBMGraphSAGE

def eval_multihop_dataset_a(seeds=[42, 101, 2024, 7, 99]):
    import copy
    f1s = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        raw_dataset, df_summary = load_all_graphs_dataset(Path('data/graph_summary.csv'))
        
        train_val_ids, test_ids = train_test_split(
            df_summary['complaint_id'].tolist(), test_size=0.20, random_state=seed, stratify=df_summary['contains_suspicious_activity']
        )
        test_set = set(test_ids)
        test_raw = [d for d in raw_dataset if d.complaint_id in test_set]
        
        # Filter for multi-node
        test_raw_multi = [d for d in test_raw if d.num_nodes > 1]
        
        from clean_5seed_benchmark import normalize_node_features_3way
        df_train_val = df_summary[df_summary['complaint_id'].isin(set(train_val_ids))]
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=0.125, random_state=seed, stratify=df_train_val['contains_suspicious_activity']
        )
        val_set = set(val_ids)
        train_raw = [d for d in raw_dataset if d.complaint_id not in test_set and d.complaint_id not in val_set]
        val_raw = [d for d in raw_dataset if d.complaint_id in val_set]
        _, _, _, test_norm = normalize_node_features_3way(train_raw, val_raw, test_raw_multi)
        
        test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)
        
        model = DualHeadGraphSAGE(input_dim=13, hidden_dim=64, dropout=0.2)
        model.load_state_dict(torch.load(f'models/clean_seed_checkpoints_dataset_a/seed{seed}.pt'))
        model.eval()
        
        test_preds, test_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                _, logits, _ = model(batch.x, batch.edge_index, batch.batch)
                prob = torch.sigmoid(logits).cpu().numpy()
                test_preds.extend((prob >= 0.5).astype(int))
                test_targets.extend(batch.y.view(-1).cpu().numpy().astype(int))
                
        gnn_test_f1 = f1_score(test_targets, test_preds, zero_division=0)
        f1s.append(gnn_test_f1)
    
    print(f"Dataset A Multi-Node Clean F1: {np.mean(f1s)*100:.2f}% +/- {np.std(f1s)*100:.2f}%")

def eval_multihop_ibm(seeds=[42, 101, 2024, 7, 99]):
    import copy
    f1s = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        raw_dataset, df_summary = load_or_create_ibm_pyg_dataset()
        
        train_val_ids, test_ids = train_test_split(
            df_summary['subgraph_id'].tolist(), test_size=0.20, random_state=seed, stratify=df_summary[TARGET_COL]
        )
        test_set = set(test_ids)
        test_raw = [d for d in raw_dataset if d.subgraph_id in test_set]
        
        # Filter for multi-node
        test_raw_multi = [d for d in test_raw if d.num_nodes > 1]
        
        from clean_5seed_benchmark import normalize_node_features_3way
        df_train_val = df_summary[df_summary['subgraph_id'].isin(set(train_val_ids))]
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=0.125, random_state=seed, stratify=df_train_val[TARGET_COL]
        )
        val_set = set(val_ids)
        train_raw = [d for d in raw_dataset if d.subgraph_id not in test_set and d.subgraph_id not in val_set]
        val_raw = [d for d in raw_dataset if d.subgraph_id in val_set]
        _, _, _, test_norm = normalize_node_features_3way(train_raw, val_raw, test_raw_multi)
        
        test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)
        
        model = IBMGraphSAGE(input_dim=7, hidden_dim=64, dropout=0.2)
        model.load_state_dict(torch.load(f'models/clean_seed_checkpoints_ibm/seed{seed}.pt'))
        model.eval()
        
        test_preds, test_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                out, _ = model(batch.x, batch.edge_index, batch.batch)
                prob = torch.sigmoid(out).cpu().numpy()
                test_preds.extend((prob >= 0.5).astype(int).tolist())
                test_targets.extend(batch.y.view(-1).cpu().numpy().astype(int).tolist())
                
        gnn_test_f1 = f1_score(test_targets, test_preds, zero_division=0)
        f1s.append(gnn_test_f1)
    
    print(f"Dataset B (IBM) Multi-Node Clean F1: {np.mean(f1s)*100:.2f}% +/- {np.std(f1s)*100:.2f}%")

if __name__ == '__main__':
    eval_multihop_dataset_a()
    eval_multihop_ibm()
