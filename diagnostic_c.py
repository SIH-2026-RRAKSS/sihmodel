import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.decomposition import PCA

from src.adapters.elliptic_adapter import EllipticAdapter
from src.elliptic_benchmark import EllipticGraphSAGE

print("--- DATASET C FEATURE IMPORTANCE & DILUTION TEST ---")
adapter = EllipticAdapter()
data, _, _ = adapter.get_train_test_split(split_timestep=34)
feat_csv = adapter.root_dir / "raw" / "elliptic_txs_features.csv"
df_feat = pd.read_csv(feat_csv, header=None, usecols=[1])
timesteps = torch.tensor(df_feat.iloc[:, 0].values, dtype=torch.long)

labeled = (data.y == 0) | (data.y == 1)
tm = labeled & (timesteps <= 29)
vm = labeled & (timesteps > 29) & (timesteps <= 34)
tem = labeled & (timesteps > 34)

X_C, Y_C = data.x.numpy(), data.binary_y.numpy()

s = 42
pw = float(np.sum(Y_C[tm] == 0) / max(1, np.sum(Y_C[tm] == 1)))

print("Training GNN...")
torch.manual_seed(s)
mdl = EllipticGraphSAGE(in_channels=data.x.size(1), hidden_channels=128, out_channels=64)
opt = torch.optim.Adam(mdl.parameters(), lr=0.003, weight_decay=1e-4)
pw_t = torch.tensor([pw])
crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)

bf, bs = 0, None
for _ in range(40):
    mdl.train()
    opt.zero_grad()
    out, _ = mdl(data.x, data.edge_index)
    crit(out[tm], data.binary_y.float()[tm]).backward()
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
    X_E = embs.numpy()

print("\n--- 1. FEATURE IMPORTANCE TEST ---")
F_tr = np.concatenate([X_C[tm], X_E[tm]], axis=1)
F_te = np.concatenate([X_C[tem], X_E[tem]], axis=1)

# feature names mapping: 0 to 164 are Tabular. 165 to 292 are GNN.
bst_ens = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(F_tr,Y_C[tm]), 100)

scores = bst_ens.get_score(importance_type='gain')
tabular_gain = 0
gnn_gain = 0
for k, v in scores.items():
    idx = int(k.replace('f', ''))
    if idx < 165: tabular_gain += v
    else: gnn_gain += v

total = tabular_gain + gnn_gain
print(f"Total Gain: {total:.2f}")
print(f"Tabular Features Gain: {tabular_gain:.2f} ({tabular_gain/total*100:.2f}%)")
print(f"GNN Embeddings Gain: {gnn_gain:.2f} ({gnn_gain/total*100:.2f}%)")

print("\n--- 2. DIMENSIONALITY REDUCTION TEST (PCA=16) ---")
pca = PCA(n_components=16, random_state=s)
X_E_pca_tr = pca.fit_transform(X_E[tm])
X_E_pca_te = pca.transform(X_E[tem])

F_tr_pca = np.concatenate([X_C[tm], X_E_pca_tr], axis=1)
F_te_pca = np.concatenate([X_C[tem], X_E_pca_te], axis=1)

bst_pca = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(F_tr_pca,Y_C[tm]), 100)
f1_pca = f1_score(Y_C[tem], bst_pca.predict(xgb.DMatrix(F_te_pca)) >= 0.5)

# Baseline XGBoost to compare
bst_base = xgb.train({'objective':'binary:logistic','eval_metric':'auc','seed':s, 'scale_pos_weight': pw, 'subsample': 0.8, 'colsample_bytree': 0.8}, xgb.DMatrix(X_C[tm],Y_C[tm]), 100)
f1_base = f1_score(Y_C[tem], bst_base.predict(xgb.DMatrix(X_C[tem])) >= 0.5)

# Full ensemble F1
f1_full = f1_score(Y_C[tem], bst_ens.predict(xgb.DMatrix(F_te)) >= 0.5)

print(f"Standalone XGBoost F1 (0.50): {f1_base*100:.2f}%")
print(f"Full Ensemble F1 (0.50)     : {f1_full*100:.2f}%")
print(f"PCA-16 Ensemble F1 (0.50)   : {f1_pca*100:.2f}%")

