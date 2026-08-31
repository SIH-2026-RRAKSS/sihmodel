import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_curve, f1_score
from torch_geometric.loader import DataLoader

from src.graphsage_classifier import DualHeadGraphSAGE, load_all_graphs_dataset
from src.ibm_graphsage_classifier import IBMGraphSAGE, TARGET_COL, load_or_create_ibm_pyg_dataset
from src.xgboost_baseline import engineer_baseline_features, EXCLUDED_COLUMNS
from src.adapters.elliptic_adapter import EllipticAdapter
from src.elliptic_benchmark import EllipticGraphSAGE

def get_opt(y, prob):
    p, r, t = precision_recall_curve(y, prob)
    f1s = 2 * (p * r) / (p + r + 1e-9)
    best_idx = np.argmax(f1s)
    return t[best_idx] if best_idx < len(t) else 0.50

def run_dataset_c():
    seeds = [42, 101, 2024, 7, 99]
    print("\n" + "="*50 + "\nDATASET C (ELLIPTIC) - CHRONOLOGICAL OOF\n" + "="*50)
    
    adapter = EllipticAdapter()
    data, _, _ = adapter.get_train_test_split(split_timestep=34)
    feat_csv = adapter.root_dir / "raw" / "elliptic_txs_features.csv"
    df_feat = pd.read_csv(feat_csv, header=None, usecols=[1])
    timesteps = torch.tensor(df_feat.iloc[:, 0].values, dtype=torch.long)
    
    labeled = (data.y == 0) | (data.y == 1)
    tm_all = labeled & (timesteps <= 29)
    vm = labeled & (timesteps > 29) & (timesteps <= 34)
    tem = labeled & (timesteps > 34)
    
    X_C, Y_C = data.x.numpy(), data.binary_y.numpy()
    
    folds = [(10, 11, 15), (15, 16, 20), (20, 21, 25), (25, 26, 29)]
    
    xgb_f1, gnn_f1, ens_05, ens_opt = [], [], [], []
    
    for s in seeds:
        print(f"Seed {s}...")
        torch.manual_seed(s)
        
        tm_xgb = labeled & (timesteps >= 11) & (timesteps <= 29)
        pw = float(np.sum(Y_C[tm_xgb] == 0) / max(1, np.sum(Y_C[tm_xgb] == 1)))
        bst = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(X_C[tm_xgb],Y_C[tm_xgb]), 100)
        xgb_f1.append(f1_score(Y_C[tem], bst.predict(xgb.DMatrix(X_C[tem])) >= 0.5))
        
        mdl = EllipticGraphSAGE(in_channels=data.x.size(1), hidden_channels=128, out_channels=64)
        opt = torch.optim.Adam(mdl.parameters(), lr=0.003, weight_decay=1e-4)
        pw_t = torch.tensor([float(np.sum(Y_C[tm_all] == 0) / max(1, np.sum(Y_C[tm_all] == 1)))])
        crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
        
        bf, bs = 0, None
        for _ in range(40):
            mdl.train()
            opt.zero_grad()
            out, _ = mdl(data.x, data.edge_index)
            crit(out[tm_all], data.binary_y.float()[tm_all]).backward()
            opt.step()
            mdl.eval()
            with torch.no_grad():
                val_out, _ = mdl(data.x, data.edge_index)
                val_prob = torch.sigmoid(val_out[vm]).numpy()
            vf1 = f1_score(Y_C[vm], val_prob>=0.5)
            if vf1 >= bf: bf, bs = vf1, copy.deepcopy(mdl.state_dict())
            
        mdl.load_state_dict(bs)
        mdl.eval()
        with torch.no_grad():
            logits, embs = mdl(data.x, data.edge_index)
            full_prob = torch.sigmoid(logits).numpy()
            E_C_test = embs.numpy()
        
        opt_t = get_opt(Y_C[vm], full_prob[vm])
        gnn_f1.append(f1_score(Y_C[tem], full_prob[tem]>=opt_t))
        
        E_C_oof = np.zeros((X_C.shape[0], 64)) 
        for train_end, val_start, val_end in folds:
            tm_fold = labeled & (timesteps <= train_end)
            vm_fold = labeled & (timesteps >= val_start) & (timesteps <= val_end)
            
            f_mdl = EllipticGraphSAGE(in_channels=data.x.size(1), hidden_channels=128, out_channels=64)
            f_opt = torch.optim.Adam(f_mdl.parameters(), lr=0.003, weight_decay=1e-4)
            f_pw_t = torch.tensor([float(np.sum(Y_C[tm_fold] == 0) / max(1, np.sum(Y_C[tm_fold] == 1)))])
            f_crit = nn.BCEWithLogitsLoss(pos_weight=f_pw_t)
            
            for _ in range(25): 
                f_mdl.train()
                f_opt.zero_grad()
                out, _ = f_mdl(data.x, data.edge_index)
                f_crit(out[tm_fold], data.binary_y.float()[tm_fold]).backward()
                f_opt.step()
                
            f_mdl.eval()
            with torch.no_grad():
                _, fold_embs = f_mdl(data.x, data.edge_index)
                E_C_oof[vm_fold] = fold_embs[vm_fold].numpy()
        
        F_tr = np.concatenate([X_C[tm_xgb], E_C_oof[tm_xgb]], axis=1)
        F_va = np.concatenate([X_C[vm], E_C_test[vm]], axis=1)
        F_te = np.concatenate([X_C[tem], E_C_test[tem]], axis=1)
        
        bst_ens = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(F_tr,Y_C[tm_xgb]), 100)
        ypv = bst_ens.predict(xgb.DMatrix(F_va))
        ypte = bst_ens.predict(xgb.DMatrix(F_te))
        
        ens_05.append(f1_score(Y_C[tem], ypte >= 0.5))
        ens_opt.append(f1_score(Y_C[tem], ypte >= get_opt(Y_C[vm], ypv)))
        
        if s == 42:
            scores = bst_ens.get_score(importance_type='gain')
            tab_gain = sum(v for k,v in scores.items() if int(k.replace('f','')) < 165)
            gnn_gain = sum(v for k,v in scores.items() if int(k.replace('f','')) >= 165)
            tot = tab_gain + gnn_gain
            print(f"  [Diagnostics] Tabular Gain: {tab_gain/tot*100:.1f}%, GNN Gain: {gnn_gain/tot*100:.1f}%")

    print(f"XGB: {np.mean(xgb_f1)*100:.2f}±{np.std(xgb_f1)*100:.2f}")
    print(f"GNN: {np.mean(gnn_f1)*100:.2f}±{np.std(gnn_f1)*100:.2f}")
    print(f"ENS(0.5): {np.mean(ens_05)*100:.2f}±{np.std(ens_05)*100:.2f}")
    print(f"ENS(Opt): {np.mean(ens_opt)*100:.2f}±{np.std(ens_opt)*100:.2f}")

if __name__ == "__main__":
    run_dataset_c()
