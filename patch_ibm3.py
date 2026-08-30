import os

path = 'src/ibm_graphsage_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

import re
old_block = r"""    train_ids, test_ids = train_test_split\(
        df_summary\["subgraph_id"\].tolist\(\),
        test_size=0.20,
        random_state=seed,
        stratify=df_summary\[TARGET_COL\]
    \)
    test_set = set\(test_ids\)
    
    train_raw = \[d for d in raw_dataset if d.subgraph_id not in test_set\]
    test_raw = \[d for d in raw_dataset if d.subgraph_id in test_set\]
    
    train_norm, test_norm = normalize_node_features\(train_raw, test_raw\)
    
    train_loader = DataLoader\(train_norm, batch_size=32, shuffle=True\)
    test_loader = DataLoader\(test_norm, batch_size=64, shuffle=False\)"""

new_block = """    train_val_ids, test_ids = train_test_split(
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
    
    train_raw = [d for d in raw_dataset if getattr(d, "subgraph_id", [""])[0] not in test_set and getattr(d, "subgraph_id", [""])[0] not in val_set]
    val_raw = [d for d in raw_dataset if getattr(d, "subgraph_id", [""])[0] in val_set]
    test_raw = [d for d in raw_dataset if getattr(d, "subgraph_id", [""])[0] in test_set]
    
    train_norm, val_norm = normalize_node_features(train_raw, val_raw)
    _, test_norm = normalize_node_features(train_raw, test_raw)
    
    train_loader = DataLoader(train_norm, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_norm, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)"""

c = re.sub(old_block, new_block, c, flags=re.MULTILINE)
with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

print('ibm_graphsage_classifier.py replaced')
