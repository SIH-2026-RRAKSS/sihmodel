import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
from torch_geometric.loader import DataLoader

from src.graphsage_classifier import DualHeadGraphSAGE, load_all_graphs_dataset
from src.ibm_graphsage_classifier import IBMGraphSAGE, TARGET_COL, load_or_create_ibm_pyg_dataset
from src.xgboost_baseline import engineer_baseline_features, EXCLUDED_COLUMNS
from src.adapters.synthetic_adapter import SyntheticAdapter
from src.adapters.elliptic_adapter import EllipticAdapter
from src.elliptic_benchmark import EllipticGraphSAGE

def get_opt(y, prob):
    p, r, t = precision_recall_curve(y, prob)
    f1s = 2 * (p * r) / (p + r + 1e-9)
    best_idx = np.argmax(f1s)
    return t[best_idx] if best_idx < len(t) else 0.50

def run_audit():
    seeds = [42, 101, 2024, 7, 99]
    print("==================================================")
    
    # ------------------ DATASET A ------------------
    print("DATASET A (SYNTHETIC)")
    raw_ds, df_sum = load_all_graphs_dataset(Path('data/graph_summary.csv'))
    
    df_eng = engineer_baseline_features(df_sum)
    feature_names = [c for c in df_eng.columns if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df_eng[c])]
    X_A, Y_A = df_eng[feature_names].fillna(0).values, df_eng['contains_suspicious_activity'].values
    
    xg05, xgo, gn05, gno = [], [], [], []
    t_xgb, t_gnn = [], []
    
    for s in seeds:
        tv_i, te_i = train_test_split(df_sum['complaint_id'], test_size=0.2, random_state=s, stratify=Y_A)
        df_tv = df_sum[df_sum['complaint_id'].isin(tv_i)]
        tr_i, va_i = train_test_split(tv_i, test_size=0.125, random_state=s, stratify=df_tv['contains_suspicious_activity'])
        
        tm = df_sum['complaint_id'].isin(tr_i)
        vm = df_sum['complaint_id'].isin(va_i)
        tem = df_sum['complaint_id'].isin(te_i)
        
        pw = np.sum(Y_A[tm] == 0) / np.sum(Y_A[tm] == 1)
        bst = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(X_A[tm],Y_A[tm]), 200)
        ypv, ypte = bst.predict(xgb.DMatrix(X_A[vm])), bst.predict(xgb.DMatrix(X_A[tem]))
        opt_x = get_opt(Y_A[vm], ypv)
        t_xgb.append(opt_x)
        xg05.append(f1_score(Y_A[tem], ypte>=0.5))
        xgo.append(f1_score(Y_A[tem], ypte>=opt_x))
        
        # GNN
        tr = [d for d in raw_ds if d.complaint_id in set(tr_i)]
        va = [d for d in raw_ds if d.complaint_id in set(va_i)]
        te = [d for d in raw_ds if d.complaint_id in set(te_i)]
        ax = torch.cat([d.x for d in tr], dim=0)
        m, st = ax.mean(dim=0), ax.std(dim=0)
        st[st==0]=1
        for d in tr: d.x = (d.x-m)/st
        for d in va: d.x = (d.x-m)/st
        for d in te: d.x = (d.x-m)/st
        
        mdl = DualHeadGraphSAGE(13, 64, 0.20)
        opt = torch.optim.Adam(mdl.parameters(), lr=0.005)
        pw_t = torch.tensor([(len(tr)-sum(d.y.item() for d in tr))/max(1, sum(d.y.item() for d in tr))])
        crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
        tl, vl, tel = DataLoader(tr, 32, shuffle=True), DataLoader(va, 32), DataLoader(te, 32)
        
        bf, bs = 0, None
        for _ in range(25):
            mdl.train()
            for b in tl:
                opt.zero_grad()
                out = mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1)
                crit(out, b.y.float()).backward()
                opt.step()
            mdl.eval()
            yt, yp = [], []
            with torch.no_grad():
                for b in vl:
                    out = mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1)
                    yt.extend(b.y.tolist()); yp.extend(torch.sigmoid(out).tolist())
            vf1 = f1_score(yt, np.array(yp)>=0.5)
            if vf1 >= bf: bf, bs = vf1, copy.deepcopy(mdl.state_dict())
            
        mdl.load_state_dict(bs)
        mdl.eval()
        ytv, ypv = [], []
        with torch.no_grad():
            for b in vl:
                out = mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1)
                ytv.extend(b.y.tolist()); ypv.extend(torch.sigmoid(out).tolist())
        opt_g = get_opt(ytv, ypv)
        t_gnn.append(opt_g)
        
        ytt, ypt = [], []
        with torch.no_grad():
            for b in tel:
                out = mdl(b.x, b.edge_index, b.batch)[1].squeeze(-1)
                ytt.extend(b.y.tolist()); ypt.extend(torch.sigmoid(out).tolist())
        gn05.append(f1_score(ytt, np.array(ypt)>=0.5))
        gno.append(f1_score(ytt, np.array(ypt)>=opt_g))
        
    print(f"XGB_0.5 : {np.mean(xg05)*100:.2f} +/- {np.std(xg05)*100:.2f}")
    print(f"XGB_Opt : {np.mean(xgo)*100:.2f} +/- {np.std(xgo)*100:.2f}")
    print(f"GNN_0.5 : {np.mean(gn05)*100:.2f} +/- {np.std(gn05)*100:.2f}")
    print(f"GNN_Opt : {np.mean(gno)*100:.2f} +/- {np.std(gno)*100:.2f}")
    print(f"XGB Thresholds: {[round(t,3) for t in t_xgb]}")
    print(f"GNN Thresholds: {[round(t,3) for t in t_gnn]}")
    
    # ------------------ DATASET B ------------------
    print("\nDATASET B (IBM)")
    raw_ds_b, df_b = load_or_create_ibm_pyg_dataset()
    FEATURE_COLS = ['num_nodes', 'num_edges', 'density', 'average_degree', 
                'total_transaction_value', 'velocity_tph', 'velocity_vph', 
                'fan_out_ratio']
    # engineer basic b features
    df_b['in_degree_incident'] = df_b['in_degree_incident'] if 'in_degree_incident' in df_b else 0
    df_b['out_degree_incident'] = df_b['out_degree_incident'] if 'out_degree_incident' in df_b else 0
    df_b['fan_out_ratio'] = df_b['out_degree_incident'] / (df_b['in_degree_incident'] + df_b['out_degree_incident'] + 1e-5)
    df_b['velocity_tph'] = df_b['num_edges'] / 72.0
    if 'total_transaction_value' not in df_b: df_b['total_transaction_value'] = 0
    df_b['velocity_vph'] = df_b['total_transaction_value'] / 72.0
    
    for c in FEATURE_COLS:
        if c not in df_b.columns: df_b[c] = 0
    X_B, Y_B = df_b[FEATURE_COLS].fillna(0).values, df_b[TARGET_COL].values
    
    xg05, xgo, gn05, gno = [], [], [], []
    t_xgb, t_gnn = [], []
    for s in seeds:
        tv_i, te_i = train_test_split(df_b['subgraph_id'], test_size=0.2, random_state=s, stratify=Y_B)
        df_tv = df_b[df_b['subgraph_id'].isin(tv_i)]
        tr_i, va_i = train_test_split(tv_i, test_size=0.125, random_state=s, stratify=df_tv[TARGET_COL])
        
        tm = df_b['subgraph_id'].isin(tr_i)
        vm = df_b['subgraph_id'].isin(va_i)
        tem = df_b['subgraph_id'].isin(te_i)
        
        pw = np.sum(Y_B[tm] == 0) / max(1, np.sum(Y_B[tm] == 1))
        bst = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(X_B[tm],Y_B[tm]), 100)
        ypv, ypte = bst.predict(xgb.DMatrix(X_B[vm])), bst.predict(xgb.DMatrix(X_B[tem]))
        opt_x = get_opt(Y_B[vm], ypv)
        t_xgb.append(opt_x)
        xg05.append(f1_score(Y_B[tem], ypte>=0.5))
        xgo.append(f1_score(Y_B[tem], ypte>=opt_x))
        
        # GNN
        tr = [d for d in raw_ds_b if d.subgraph_id in set(tr_i)]
        va = [d for d in raw_ds_b if d.subgraph_id in set(va_i)]
        te = [d for d in raw_ds_b if d.subgraph_id in set(te_i)]
        ax = torch.cat([d.x for d in tr], dim=0)
        m, st = ax.mean(dim=0), ax.std(dim=0)
        st[st==0]=1
        for d in tr: d.x = (d.x-m)/st
        for d in va: d.x = (d.x-m)/st
        for d in te: d.x = (d.x-m)/st
        
        mdl = IBMGraphSAGE(7, 64, 0.20)
        opt = torch.optim.Adam(mdl.parameters(), lr=0.005)
        pw_t = torch.tensor([(len(tr)-sum(d.y.item() for d in tr))/max(1, sum(d.y.item() for d in tr))])
        crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
        tl, vl, tel = DataLoader(tr, 32, shuffle=True), DataLoader(va, 32), DataLoader(te, 32)
        
        bf, bs = 0, None
        for _ in range(25):
            mdl.train()
            for b in tl:
                opt.zero_grad()
                out = mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1)
                crit(out, b.y.float()).backward()
                opt.step()
            mdl.eval()
            yt, yp = [], []
            with torch.no_grad():
                for b in vl:
                    out = mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1)
                    yt.extend(b.y.tolist()); yp.extend(torch.sigmoid(out).tolist())
            vf1 = f1_score(yt, np.array(yp)>=0.5)
            if vf1 >= bf: bf, bs = vf1, copy.deepcopy(mdl.state_dict())
            
        mdl.load_state_dict(bs)
        mdl.eval()
        ytv, ypv = [], []
        with torch.no_grad():
            for b in vl:
                out = mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1)
                ytv.extend(b.y.tolist()); ypv.extend(torch.sigmoid(out).tolist())
        opt_g = get_opt(ytv, ypv)
        t_gnn.append(opt_g)
        
        ytt, ypt = [], []
        with torch.no_grad():
            for b in tel:
                out = mdl(b.x, b.edge_index, b.batch)[0].squeeze(-1)
                ytt.extend(b.y.tolist()); ypt.extend(torch.sigmoid(out).tolist())
        gn05.append(f1_score(ytt, np.array(ypt)>=0.5))
        gno.append(f1_score(ytt, np.array(ypt)>=opt_g))
        
    print(f"XGB_0.5 : {np.mean(xg05)*100:.2f} +/- {np.std(xg05)*100:.2f}")
    print(f"XGB_Opt : {np.mean(xgo)*100:.2f} +/- {np.std(xgo)*100:.2f}")
    print(f"GNN_0.5 : {np.mean(gn05)*100:.2f} +/- {np.std(gn05)*100:.2f}")
    print(f"GNN_Opt : {np.mean(gno)*100:.2f} +/- {np.std(gno)*100:.2f}")
    print(f"XGB Thresholds: {[round(t,3) for t in t_xgb]}")
    print(f"GNN Thresholds: {[round(t,3) for t in t_gnn]}")
    
    # ------------------ DATASET C ------------------
    print("\nDATASET C (ELLIPTIC)")
    adapter = EllipticAdapter()
    data, _, _ = adapter.get_train_test_split(split_timestep=34)
    feat_csv = adapter.root_dir / "raw" / "elliptic_txs_features.csv"
    df_feat = pd.read_csv(feat_csv, header=None, usecols=[1])
    timesteps = torch.tensor(df_feat.iloc[:, 0].values, dtype=torch.long)
    
    labeled_mask = (data.y == 0) | (data.y == 1)
    train_mask = labeled_mask & (timesteps <= 29)
    val_mask = labeled_mask & (timesteps > 29) & (timesteps <= 34)
    test_mask = labeled_mask & (timesteps > 34)
    
    y = data.binary_y.numpy()
    X = data.x.numpy()
    
    xg05, xgo, gn05, gno = [], [], [], []
    t_xgb, t_gnn = [], []
    for s in seeds:
        dtrain = xgb.DMatrix(X[train_mask], label=y[train_mask])
        dval = xgb.DMatrix(X[val_mask], label=y[val_mask])
        dtest = xgb.DMatrix(X[test_mask])
        
        pos_weight = float(np.sum(y[train_mask] == 0) / max(1, np.sum(y[train_mask] == 1)))
        # Using correct seed and subsample to prevent 0 variance
        bst = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pos_weight, 'subsample': 0.8, 'colsample_bytree': 0.8}, dtrain, 100)
        ypv = bst.predict(dval)
        ypt = bst.predict(dtest)
        opt_x = get_opt(y[val_mask], ypv)
        t_xgb.append(opt_x)
        xg05.append(f1_score(y[test_mask], ypt>=0.5))
        xgo.append(f1_score(y[test_mask], ypt>=opt_x))
        
        # GNN
        torch.manual_seed(s)
        model = EllipticGraphSAGE(in_channels=data.x.size(1), hidden_channels=128, out_channels=64)
        opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
        pw_t = torch.tensor([pos_weight])
        crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
        
        bf, bs = 0, None
        for _ in range(40):
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index).squeeze(-1)
            loss = crit(out[train_mask], data.binary_y.float()[train_mask])
            loss.backward()
            opt.step()
            
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index).squeeze(-1)
                val_prob = torch.sigmoid(val_out[val_mask]).numpy()
            vf1 = f1_score(y[val_mask], val_prob>=0.5)
            if vf1 >= bf: bf, bs = vf1, copy.deepcopy(model.state_dict())
            
        model.load_state_dict(bs)
        model.eval()
        with torch.no_grad():
            full_out = model(data.x, data.edge_index).squeeze(-1)
            full_prob = torch.sigmoid(full_out).numpy()
            
        opt_g = get_opt(y[val_mask], full_prob[val_mask])
        t_gnn.append(opt_g)
        gn05.append(f1_score(y[test_mask], full_prob[test_mask]>=0.5))
        gno.append(f1_score(y[test_mask], full_prob[test_mask]>=opt_g))
        
    print(f"XGB_0.5 : {np.mean(xg05)*100:.2f} +/- {np.std(xg05)*100:.2f}")
    print(f"XGB_Opt : {np.mean(xgo)*100:.2f} +/- {np.std(xgo)*100:.2f}")
    print(f"GNN_0.5 : {np.mean(gn05)*100:.2f} +/- {np.std(gn05)*100:.2f}")
    print(f"GNN_Opt : {np.mean(gno)*100:.2f} +/- {np.std(gno)*100:.2f}")
    print(f"XGB Thresholds: {[round(t,3) for t in t_xgb]}")
    print(f"GNN Thresholds: {[round(t,3) for t in t_gnn]}")

run_audit()
