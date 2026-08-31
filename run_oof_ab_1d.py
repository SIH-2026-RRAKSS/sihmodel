import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

from src.graphsage_classifier import DualHeadGraphSAGE, load_all_graphs_dataset
from src.ibm_graphsage_classifier import IBMGraphSAGE, TARGET_COL, load_or_create_ibm_pyg_dataset
from src.xgboost_baseline import engineer_baseline_features, EXCLUDED_COLUMNS

seeds = [42, 101, 2024, 7, 99, 123, 456, 789, 111, 222]

def run_a():
    print("\nDATASET A - 10-SEED 1D-LOGIT OOF")
    raw_ds, df_sum = load_all_graphs_dataset(Path('data/graph_summary.csv'))
    df_eng = engineer_baseline_features(df_sum)
    feature_names = [c for c in df_eng.columns if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df_eng[c])]
    X_A, Y_A = df_eng[feature_names].fillna(0).values, df_eng['contains_suspicious_activity'].values
    
    xgb_f1, gnn_f1, ens_f1 = [], [], []
    id_to_data = {d.complaint_id: d for d in raw_ds}
    
    for s in seeds:
        tv_i, te_i = train_test_split(df_sum['complaint_id'], test_size=0.2, random_state=s, stratify=Y_A)
        df_tv = df_sum[df_sum['complaint_id'].isin(tv_i)]
        tr_i, va_i = train_test_split(tv_i, test_size=0.125, random_state=s, stratify=df_tv['contains_suspicious_activity'])
        
        tm, vm, tem = df_sum['complaint_id'].isin(tr_i), df_sum['complaint_id'].isin(va_i), df_sum['complaint_id'].isin(te_i)
        tr = [id_to_data[i] for i in df_sum[tm]['complaint_id'].tolist()]
        va = [id_to_data[i] for i in df_sum[vm]['complaint_id'].tolist()]
        te = [id_to_data[i] for i in df_sum[tem]['complaint_id'].tolist()]
        
        pw = float(np.sum(Y_A[tm] == 0) / max(1, np.sum(Y_A[tm] == 1)))
        bst = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(X_A[tm],Y_A[tm]), 200)
        xgb_f1.append(f1_score(Y_A[tem], bst.predict(xgb.DMatrix(X_A[tem])) >= 0.5))
        
        tr_norm, va_norm, te_norm = copy.deepcopy(tr), copy.deepcopy(va), copy.deepcopy(te)
        m, st = torch.cat([d.x for d in tr_norm], dim=0).mean(dim=0), torch.cat([d.x for d in tr_norm], dim=0).std(dim=0)
        st[st==0]=1
        for d in tr_norm: d.x = (d.x-m)/st
        for d in va_norm: d.x = (d.x-m)/st
        for d in te_norm: d.x = (d.x-m)/st
        
        mdl = DualHeadGraphSAGE(13, 64, 0.20)
        opt = torch.optim.Adam(mdl.parameters(), lr=0.005)
        pw_t = torch.tensor([(len(tr)-sum(d.y.item() for d in tr))/max(1, sum(d.y.item() for d in tr))])
        crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
        tl, vl, tel = DataLoader(tr_norm, 32, shuffle=True), DataLoader(va_norm, 32), DataLoader(te_norm, 32)
        
        bf, bs = 0, None
        for _ in range(25):
            mdl.train()
            for b in tl:
                opt.zero_grad()
                crit(mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1), b.y.float()).backward()
                opt.step()
            mdl.eval()
            yt, yp = [], []
            with torch.no_grad():
                for b in vl: yt.extend(b.y.tolist()); yp.extend(torch.sigmoid(mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1)).tolist())
            vf1 = f1_score(yt, np.array(yp)>=0.5)
            if vf1 >= bf: bf, bs = vf1, copy.deepcopy(mdl.state_dict())
            
        mdl.load_state_dict(bs)
        mdl.eval()
        yt, yp = [], []
        with torch.no_grad():
            for b in tel: yt.extend(b.y.tolist()); yp.extend(torch.sigmoid(mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1)).tolist())
        gnn_f1.append(f1_score(yt, np.array(yp)>=0.5))
        
        # 1D-Logit Extraction
        L_te = torch.cat([mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1) for b in DataLoader(te_norm, 32, shuffle=False)], dim=0).detach().cpu().numpy().reshape(-1, 1)
        
        L_oof = np.zeros((len(tr_norm), 1))
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        for tr_idx, val_idx in kf.split(tr_norm):
            fold_tr = [tr_norm[i] for i in tr_idx]
            fold_va = [tr_norm[i] for i in val_idx]
            
            f_mdl = DualHeadGraphSAGE(13, 64, 0.20)
            f_opt = torch.optim.Adam(f_mdl.parameters(), lr=0.005)
            f_pw_t = torch.tensor([(len(fold_tr)-sum(d.y.item() for d in fold_tr))/max(1, sum(d.y.item() for d in fold_tr))])
            f_crit = nn.BCEWithLogitsLoss(pos_weight=f_pw_t)
            ftl = DataLoader(fold_tr, 32, shuffle=True)
            fvl = DataLoader(fold_va, 32, shuffle=False)
            
            for _ in range(15):
                f_mdl.train()
                for b in ftl:
                    f_opt.zero_grad()
                    f_crit(f_mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1), b.y.float()).backward()
                    f_opt.step()
                    
            f_mdl.eval()
            with torch.no_grad():
                val_logits = torch.cat([f_mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1) for b in fvl], dim=0).detach().cpu().numpy().reshape(-1, 1)
            L_oof[val_idx] = val_logits
            
        F_tr = np.concatenate([X_A[tm], L_oof], axis=1)
        F_te = np.concatenate([X_A[tem], L_te], axis=1)
        
        bst_ens = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(F_tr,Y_A[tm]), 200)
        ens_f1.append(f1_score(Y_A[tem], bst_ens.predict(xgb.DMatrix(F_te)) >= 0.5))
        
    print(f"Dataset A XGB: {np.mean(xgb_f1)*100:.2f}±{np.std(xgb_f1)*100:.2f}")
    print(f"Dataset A GNN: {np.mean(gnn_f1)*100:.2f}±{np.std(gnn_f1)*100:.2f}")
    print(f"Dataset A ENS: {np.mean(ens_f1)*100:.2f}±{np.std(ens_f1)*100:.2f}")

def run_b():
    print("\nDATASET B - 10-SEED 1D-LOGIT OOF")
    raw_ds_b, df_b = load_or_create_ibm_pyg_dataset()
    FEATURE_COLS = ['num_nodes', 'num_edges', 'density', 'average_degree', 'total_transaction_value', 'velocity_tph', 'velocity_vph', 'fan_out_ratio']
    df_b['in_degree_incident'] = df_b['in_degree_incident'] if 'in_degree_incident' in df_b else 0
    df_b['out_degree_incident'] = df_b['out_degree_incident'] if 'out_degree_incident' in df_b else 0
    df_b['fan_out_ratio'] = df_b['out_degree_incident'] / (df_b['in_degree_incident'] + df_b['out_degree_incident'] + 1e-5)
    df_b['velocity_tph'] = df_b['num_edges'] / 72.0
    if 'total_transaction_value' not in df_b: df_b['total_transaction_value'] = 0
    df_b['velocity_vph'] = df_b['total_transaction_value'] / 72.0
    for c in FEATURE_COLS:
        if c not in df_b.columns: df_b[c] = 0
    X_B, Y_B = df_b[FEATURE_COLS].fillna(0).values, df_b[TARGET_COL].values
    
    xgb_f1, gnn_f1, ens_f1 = [], [], []
    id_to_data_b = {d.subgraph_id: d for d in raw_ds_b}
    
    for s in seeds:
        tv_i, te_i = train_test_split(df_b['subgraph_id'], test_size=0.2, random_state=s, stratify=Y_B)
        df_tv = df_b[df_b['subgraph_id'].isin(tv_i)]
        tr_i, va_i = train_test_split(tv_i, test_size=0.125, random_state=s, stratify=df_tv[TARGET_COL])
        
        tm, vm, tem = df_b['subgraph_id'].isin(tr_i), df_b['subgraph_id'].isin(va_i), df_b['subgraph_id'].isin(te_i)
        tr = [id_to_data_b[i] for i in df_b[tm]['subgraph_id'].tolist()]
        va = [id_to_data_b[i] for i in df_b[vm]['subgraph_id'].tolist()]
        te = [id_to_data_b[i] for i in df_b[tem]['subgraph_id'].tolist()]
        
        pw = np.sum(Y_B[tm] == 0) / max(1, np.sum(Y_B[tm] == 1))
        bst = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(X_B[tm],Y_B[tm]), 100)
        xgb_f1.append(f1_score(Y_B[tem], bst.predict(xgb.DMatrix(X_B[tem])) >= 0.5))
        
        tr_norm, va_norm, te_norm = copy.deepcopy(tr), copy.deepcopy(va), copy.deepcopy(te)
        m, st = torch.cat([d.x for d in tr_norm], dim=0).mean(dim=0), torch.cat([d.x for d in tr_norm], dim=0).std(dim=0)
        st[st==0]=1
        for d in tr_norm: d.x = (d.x-m)/st
        for d in va_norm: d.x = (d.x-m)/st
        for d in te_norm: d.x = (d.x-m)/st
        
        mdl = IBMGraphSAGE(7, 64, 0.20)
        opt = torch.optim.Adam(mdl.parameters(), lr=0.005)
        pw_t = torch.tensor([(len(tr)-sum(d.y.item() for d in tr))/max(1, sum(d.y.item() for d in tr))])
        crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
        tl, vl, tel = DataLoader(tr_norm, 32, shuffle=True), DataLoader(va_norm, 32), DataLoader(te_norm, 32)
        
        bf, bs = 0, None
        for _ in range(25):
            mdl.train()
            for b in tl:
                opt.zero_grad()
                crit(mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1), b.y.float()).backward()
                opt.step()
            mdl.eval()
            yt, yp = [], []
            with torch.no_grad():
                for b in vl: yt.extend(b.y.tolist()); yp.extend(torch.sigmoid(mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1)).tolist())
            vf1 = f1_score(yt, np.array(yp)>=0.5)
            if vf1 >= bf: bf, bs = vf1, copy.deepcopy(mdl.state_dict())
            
        mdl.load_state_dict(bs)
        mdl.eval()
        yt, yp = [], []
        with torch.no_grad():
            for b in tel: yt.extend(b.y.tolist()); yp.extend(torch.sigmoid(mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1)).tolist())
        gnn_f1.append(f1_score(yt, np.array(yp)>=0.5))
        
        # 1D-Logit Extraction
        L_te = torch.cat([mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1) for b in DataLoader(te_norm, 32, shuffle=False)], dim=0).detach().cpu().numpy().reshape(-1, 1)
        
        L_oof = np.zeros((len(tr_norm), 1))
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        for tr_idx, val_idx in kf.split(tr_norm):
            fold_tr = [tr_norm[i] for i in tr_idx]
            fold_va = [tr_norm[i] for i in val_idx]
            
            f_mdl = IBMGraphSAGE(7, 64, 0.20)
            f_opt = torch.optim.Adam(f_mdl.parameters(), lr=0.005)
            f_pw_t = torch.tensor([(len(fold_tr)-sum(d.y.item() for d in fold_tr))/max(1, sum(d.y.item() for d in fold_tr))])
            f_crit = nn.BCEWithLogitsLoss(pos_weight=f_pw_t)
            ftl = DataLoader(fold_tr, 32, shuffle=True)
            fvl = DataLoader(fold_va, 32, shuffle=False)
            
            for _ in range(15):
                f_mdl.train()
                for b in ftl:
                    f_opt.zero_grad()
                    f_crit(f_mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1), b.y.float()).backward()
                    f_opt.step()
                    
            f_mdl.eval()
            with torch.no_grad():
                val_logits = torch.cat([f_mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1) for b in fvl], dim=0).detach().cpu().numpy().reshape(-1, 1)
            L_oof[val_idx] = val_logits
            
        F_tr = np.concatenate([X_B[tm], L_oof], axis=1)
        F_te = np.concatenate([X_B[tem], L_te], axis=1)
        
        bst_ens = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(F_tr,Y_B[tm]), 100)
        ens_f1.append(f1_score(Y_B[tem], bst_ens.predict(xgb.DMatrix(F_te)) >= 0.5))
        
    print(f"Dataset B XGB: {np.mean(xgb_f1)*100:.2f}±{np.std(xgb_f1)*100:.2f}")
    print(f"Dataset B GNN: {np.mean(gnn_f1)*100:.2f}±{np.std(gnn_f1)*100:.2f}")
    print(f"Dataset B ENS: {np.mean(ens_f1)*100:.2f}±{np.std(ens_f1)*100:.2f}")

if __name__ == "__main__":
    run_a()
    run_b()
