import os
import re

path = 'src/graphsage_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Overwrite everything between `# 3. Train / Test Split Alignment` and `    # 6. Initialize GraphSAGE Model`
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
    
    train_raw = [d for d in raw_dataset if getattr(d, "complaint_id", [""])[0] not in test_set and getattr(d, "complaint_id", [""])[0] not in val_set]
    val_raw = [d for d in raw_dataset if getattr(d, "complaint_id", [""])[0] in val_set]
    test_raw = [d for d in raw_dataset if getattr(d, "complaint_id", [""])[0] in test_set]
    
    train_dataset, val_dataset, mean_norm, std_norm = normalize_node_features(train_raw, val_raw)
    _, test_dataset, _, _ = normalize_node_features(train_raw, test_raw)
    all_dataset, _, _, _ = normalize_node_features(raw_dataset, raw_dataset)

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
"""

c = re.sub(
    r"train_ids, test_ids = get_or_create_train_test_split\(df_summary\).*?test_loader = DataLoader\(test_dataset, batch_size=64, shuffle=False\)",
    split_code.strip(),
    c,
    flags=re.DOTALL
)

# And also replace `best_epoch = train_graphsage_model(...)` loop inside `evaluate_test_set` ... wait, `evaluate_test_set` takes test_loader, that's fine.
# But inside `train_graphsage_model`, we must evaluate `val_loader`.
# Let's fix test_loader to val_loader in `train_graphsage_model` if it's not already.

c = c.replace('test_loader=val_loader,', 'val_loader=val_loader,')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('graphsage_classifier.py patched successfully.')
