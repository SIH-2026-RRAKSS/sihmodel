import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from pathlib import Path
from src.graphsage_classifier import DualHeadGraphSAGE, load_all_graphs_dataset
from src.ibm_graphsage_classifier import load_or_create_ibm_pyg_dataset, TARGET_COL, IBMGraphSAGE

def normalize_node_features_3way(train_raw, val_raw, test_raw):
    import copy
    train_dataset = copy.deepcopy(train_raw)
    val_dataset = copy.deepcopy(val_raw)
    test_dataset = copy.deepcopy(test_raw)
    
    all_x = torch.cat([d.x for d in train_dataset], dim=0)
    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0)
    std[std == 0] = 1.0
    
    for d in train_dataset: d.x = (d.x - mean) / std
    for d in val_dataset: d.x = (d.x - mean) / std
    for d in test_dataset: d.x = (d.x - mean) / std
    
    return None, train_dataset, val_dataset, test_dataset

def evaluate_clean_dataset_a(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    raw_dataset, df_summary = load_all_graphs_dataset(Path('data/graph_summary.csv'))
    
    train_val_ids, test_ids = train_test_split(
        df_summary['complaint_id'].tolist(), test_size=0.20, random_state=seed, stratify=df_summary['contains_suspicious_activity']
    )
    df_train_val = df_summary[df_summary['complaint_id'].isin(set(train_val_ids))]
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.125, random_state=seed, stratify=df_train_val['contains_suspicious_activity']
    )
    
    test_set = set(test_ids)
    val_set = set(val_ids)
    
    train_raw = [d for d in raw_dataset if d.complaint_id not in test_set and d.complaint_id not in val_set]
    val_raw = [d for d in raw_dataset if d.complaint_id in val_set]
    test_raw = [d for d in raw_dataset if d.complaint_id in test_set]
    
    _, train_norm, val_norm, test_norm = normalize_node_features_3way(train_raw, val_raw, test_raw)
    
    train_loader = DataLoader(train_norm, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_norm, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)
    
    model = DualHeadGraphSAGE(input_dim=13, hidden_dim=64, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.37]))
    
    best_val_f1 = -1
    best_weights = None
    
    for epoch in range(150):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            _, logits_graph, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits_graph, batch.y.view(-1).float())
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                _, logits, _ = model(batch.x, batch.edge_index, batch.batch)
                prob = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend((prob >= 0.5).astype(int))
                val_targets.extend(batch.y.view(-1).cpu().numpy().astype(int))
        
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_weights)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            _, logits, _ = model(batch.x, batch.edge_index, batch.batch)
            prob = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend((prob >= 0.5).astype(int))
            test_targets.extend(batch.y.view(-1).cpu().numpy().astype(int))
            
    gnn_test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    print(f'Dataset A - Seed {seed} | GNN Clean Test F1: {gnn_test_f1:.4f}')

def evaluate_clean_ibm(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    raw_dataset, df_summary = load_or_create_ibm_pyg_dataset()
    
    train_val_ids, test_ids = train_test_split(
        df_summary['subgraph_id'].tolist(), test_size=0.20, random_state=seed, stratify=df_summary[TARGET_COL]
    )
    df_train_val = df_summary[df_summary['subgraph_id'].isin(set(train_val_ids))]
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.125, random_state=seed, stratify=df_train_val[TARGET_COL]
    )
    
    test_set = set(test_ids)
    val_set = set(val_ids)
    
    train_raw = [d for d in raw_dataset if d.subgraph_id not in test_set and d.subgraph_id not in val_set]
    val_raw = [d for d in raw_dataset if d.subgraph_id in val_set]
    test_raw = [d for d in raw_dataset if d.subgraph_id in test_set]
    
    _, train_norm, val_norm, test_norm = normalize_node_features_3way(train_raw, val_raw, test_raw)
    
    train_loader = DataLoader(train_norm, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_norm, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)
    
    model = IBMGraphSAGE(input_dim=7, hidden_dim=64, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_f1 = -1
    best_weights = None
    
    for epoch in range(30):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze(-1).float())
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out, _ = model(batch.x, batch.edge_index, batch.batch)
                prob = torch.sigmoid(out).cpu().numpy()
                val_preds.extend((prob >= 0.5).astype(int))
                val_targets.extend(batch.y.squeeze(-1).cpu().numpy().astype(int))
        
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_weights)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out, _ = model(batch.x, batch.edge_index, batch.batch)
            prob = torch.sigmoid(out).cpu().numpy()
            test_preds.extend((prob >= 0.5).astype(int))
            test_targets.extend(batch.y.squeeze(-1).cpu().numpy().astype(int))
            
    gnn_test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    print(f'Dataset B (IBM) - Seed {seed} | GNN Clean Test F1: {gnn_test_f1:.4f}')

if __name__ == '__main__':
    evaluate_clean_dataset_a(42)
    evaluate_clean_ibm(42)
