import os
import re

path = 'src/ibm_graphsage_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

split_code = """    train_val_ids, test_ids = train_test_split(
        df_summary["subgraph_id"].tolist(),
        test_size=0.20,
        random_state=seed,
        stratify=df_summary[TARGET_COL]
    )
    df_train_val = df_summary[df_summary["subgraph_id"].isin(set(train_val_ids))]
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.125, random_state=seed, stratify=df_train_val[TARGET_COL]
    )
    test_set = set(test_ids)
    val_set = set(val_ids)
    
    train_raw = [d for d in raw_dataset if d.subgraph_id not in test_set and d.subgraph_id not in val_set]
    val_raw = [d for d in raw_dataset if d.subgraph_id in val_set]
    test_raw = [d for d in raw_dataset if d.subgraph_id in test_set]
    
    train_norm, val_norm = normalize_node_features(train_raw, val_raw)
    _, test_norm = normalize_node_features(train_raw, test_raw)
    
    train_loader = DataLoader(train_norm, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_norm, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)"""

c = re.sub(r'    train_ids, test_ids = train_test_split\(.*?test_loader = DataLoader\(test_norm, batch_size=64, shuffle=False\)', split_code, c, flags=re.DOTALL)
c = c.replace('for batch in test_loader:', 'for batch in val_loader:')

end_eval = """    # Now evaluate best model on test set
    model.load_state_dict(torch.load(MODELS_DIR / f"ibm_seed_checkpoints/seed{seed}.pt"))
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            out, _ = model(batch.x, batch.edge_index, batch.batch)
            prob = torch.sigmoid(out).cpu().numpy()
            pred = (prob >= 0.50).astype(int)
            target = batch.y.squeeze(-1).cpu().numpy().astype(int)
            all_probs.extend(prob.tolist())
            all_preds.extend(pred.tolist())
            all_targets.extend(target.tolist())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    best_metrics["accuracy"] = accuracy_score(y_true, y_pred)
    best_metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    best_metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    best_metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    best_metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    best_metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    best_metrics["tp"] = int(np.sum((y_true == 1) & (y_pred == 1)))
    best_metrics["fp"] = int(np.sum((y_true == 0) & (y_pred == 1)))
    best_metrics["fn"] = int(np.sum((y_true == 1) & (y_pred == 0)))
    best_metrics["tn"] = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    return best_metrics"""

c = re.sub(r'    return best_metrics', end_eval, c)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('ibm done')
