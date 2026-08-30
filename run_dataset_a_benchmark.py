import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.graphsage_classifier import (
    load_all_graphs_dataset,
    normalize_node_features,
    DualHeadGraphSAGE,
    train_graphsage_model,
    set_seed
)

def run_benchmark():
    summary_file = Path("data/graph_summary.csv")
    raw_dataset, df_summary = load_all_graphs_dataset(summary_file)
    
    seeds = [42, 101, 2024, 7, 99]
    xgb_df = pd.read_csv("data/xgboost_multi_seed_evaluation.csv")
    xgb_f1s = xgb_df.set_index("seed")["f1"].to_dict()
    
    results = []
    
    for seed in seeds:
        set_seed(seed)
        
        # Split according to the exact same logic as XGBoost baseline
        train_df, test_df = train_test_split(
            df_summary,
            test_size=0.20,
            random_state=seed,
            stratify=df_summary["contains_suspicious_activity"]
        )
        
        train_cids = set(train_df["complaint_id"])
        test_cids = set(test_df["complaint_id"])
        
        train_raw = [d for d in raw_dataset if d.complaint_id in train_cids]
        test_raw = [d for d in raw_dataset if d.complaint_id in test_cids]
        
        train_norm, test_norm, mean_norm, std_norm = normalize_node_features(train_raw, test_raw)
        pos_weight = float((len(train_norm) - sum(d.y.item() for d in train_norm)) / max(1, sum(d.y.item() for d in train_norm)))
        
        model = DualHeadGraphSAGE(input_dim=13, hidden_dim=64, dropout=0.2)
        
        train_loader = DataLoader(train_norm, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)
        
        # Override save path for each seed
        seed_ckpt_path = Path(f"models/seed_checkpoints/seed{seed}.pt")
        seed_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        
        import src.graphsage_classifier as gc
        old_path = gc.GRAPHSAGE_MODEL_FILE
        gc.GRAPHSAGE_MODEL_FILE = seed_ckpt_path
        
        model, _, _ = train_graphsage_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            pos_weight_val=pos_weight,
            max_epochs=150,
            patience=20,
            lr=0.001
        )
        gc.GRAPHSAGE_MODEL_FILE = old_path
        
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch in test_loader:
                _, logits_graph, _ = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(logits_graph).cpu().numpy().flatten()
                y_probs.extend(probs)
                y_true.extend(batch.y.cpu().numpy().flatten())
                
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_pred = (y_probs >= 0.50).astype(int)
        
        f1 = f1_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_probs)
        
        results.append({
            "seed": seed,
            "n_test": len(y_true),
            "test_pos": int(sum(y_true)),
            "test_neg": int(len(y_true) - sum(y_true)),
            "xgb_f1": xgb_f1s[seed],
            "gnn_f1": f1,
            "f1_delta": f1 - xgb_f1s[seed],
            "gnn_pr_auc": pr_auc
        })
        print(f"Seed {seed}: F1={f1:.4f}, PR-AUC={pr_auc:.4f}")
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("data/model_multi_seed_comparison.csv", index=False)
    print("Multi-seed F1:", df_res["gnn_f1"].mean(), "+/-", df_res["gnn_f1"].std())

if __name__ == "__main__":
    run_benchmark()

if __name__ == "__main__":
    run_benchmark()
