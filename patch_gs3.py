import os
import re

path = 'src/graphsage_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# 1. Update the train function definition
c = c.replace('train_loader: DataLoader,\n    test_loader: DataLoader,', 'train_loader: DataLoader,\n    val_loader: DataLoader,')
c = c.replace('for batch in test_loader:', 'for batch in val_loader:')
c = c.replace('total_val_graphs = 0\n        all_val_probs = []\n        all_val_targets = []\n\n        with torch.no_grad():\n            for batch in test_loader:', 'total_val_graphs = 0\n        all_val_probs = []\n        all_val_targets = []\n\n        with torch.no_grad():\n            for batch in val_loader:')
c = c.replace('test_loader=test_loader,', 'val_loader=val_loader,')

# 2. Update the main split logic
split_code = """
    from sklearn.model_selection import train_test_split
    
    # Proper 70/10/20 split
    train_val_ids, test_ids = train_test_split(
        df_summary['complaint_id'].tolist(), test_size=0.20, random_state=RANDOM_SEED, stratify=df_summary['contains_suspicious_activity']
    )
    df_train_val = df_summary[df_summary['complaint_id'].isin(set(train_val_ids))]
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.125, random_state=RANDOM_SEED, stratify=df_train_val['contains_suspicious_activity']
    )
    
    test_set = set(test_ids)
    val_set = set(val_ids)
    
    train_raw = []
    val_raw = []
    test_raw = []
    for g in raw_dataset:
        cid = getattr(g, "complaint_id", [""])[0]
        if cid in test_set:
            test_raw.append(g)
        elif cid in val_set:
            val_raw.append(g)
        else:
            train_raw.append(g)
            
    print(f"Train graphs: {len(train_raw)}, Val graphs: {len(val_raw)}, Test graphs: {len(test_raw)}")
    
    train_dataset, val_dataset, mean_norm, std_norm = normalize_node_features(train_raw, val_raw)
    _, test_dataset, _, _ = normalize_node_features(train_raw, test_raw)
    all_dataset, _, _, _ = normalize_node_features(raw_dataset, raw_dataset)

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
"""
c = re.sub(
    r"train_raw = \[\]\n.*?test_loader = DataLoader\(test_dataset, batch_size=64, shuffle=False\)",
    split_code.strip(),
    c,
    flags=re.DOTALL
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('graphsage_classifier.py patched successfully.')
