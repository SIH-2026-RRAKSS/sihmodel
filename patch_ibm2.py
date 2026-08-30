import os
import re

path = 'src/ibm_graphsage_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

split_code = """
    from sklearn.model_selection import train_test_split
    
    train_val_ids, test_ids = train_test_split(
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
    
    train_raw = []
    val_raw = []
    test_raw = []
    for g in raw_dataset:
        sid = getattr(g, "subgraph_id", [""])[0]
        if sid in test_set:
            test_raw.append(g)
        elif sid in val_set:
            val_raw.append(g)
        else:
            train_raw.append(g)
            
    train_dataset, val_dataset, mean_norm, std_norm = normalize_node_features(train_raw, val_raw)
    _, test_dataset, _, _ = normalize_node_features(train_raw, test_raw)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
"""

c = re.sub(
    r'train_ids, test_ids = train_test_split\(.*?test_loader = DataLoader\(test_dataset, batch_size=64, shuffle=False\)',
    split_code.strip(),
    c,
    flags=re.DOTALL
)

c = c.replace('for batch in test_loader:', 'for batch in val_loader:')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

print('ibm_graphsage_classifier.py patched successfully.')
