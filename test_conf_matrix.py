import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from src.ibm_graphsage_classifier import load_or_create_ibm_pyg_dataset, TARGET_COL, IBMGraphSAGE

def print_confusion_matrices(seeds=[42, 101, 2024, 7, 99]):
    raw_dataset, df_summary = load_or_create_ibm_pyg_dataset()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_val_ids, test_ids = train_test_split(
            df_summary['subgraph_id'].tolist(), test_size=0.20, random_state=seed, stratify=df_summary[TARGET_COL]
        )
        test_set = set(test_ids)
        test_raw = [d for d in raw_dataset if d.subgraph_id in test_set]
        
        df_train_val = df_summary[df_summary['subgraph_id'].isin(set(train_val_ids))]
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=0.125, random_state=seed, stratify=df_train_val[TARGET_COL]
        )
        val_set = set(val_ids)
        train_raw = [d for d in raw_dataset if d.subgraph_id not in test_set and d.subgraph_id not in val_set]
        val_raw = [d for d in raw_dataset if d.subgraph_id in val_set]
        
        from clean_5seed_benchmark import normalize_node_features_3way
        _, _, _, test_norm = normalize_node_features_3way(train_raw, val_raw, test_raw)
        test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)
        
        model = IBMGraphSAGE(input_dim=7, hidden_dim=64, dropout=0.20)
        model.load_state_dict(torch.load(f"models/clean_seed_checkpoints_ibm/seed{seed}.pt"))
        model.eval()
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                out, _ = model(batch.x, batch.edge_index, batch.batch)
                prob = torch.sigmoid(out).cpu().numpy()
                y_pred.extend((prob >= 0.50).astype(int).tolist())
                y_true.extend(batch.y.squeeze(-1).cpu().numpy().astype(int).tolist())
                
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"IBM Seed {seed:>4}: TP={tp:<3} FP={fp:<3} FN={fn:<3} TN={tn:<3} F1={2*tp/(2*tp+fp+fn):.4f}")

if __name__ == '__main__':
    print_confusion_matrices()
