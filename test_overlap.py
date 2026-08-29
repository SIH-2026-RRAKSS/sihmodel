import pandas as pd
from sklearn.model_selection import train_test_split
from src.ibm_graphsage_classifier import load_or_create_ibm_pyg_dataset, TARGET_COL

_, df_summary = load_or_create_ibm_pyg_dataset()

_, test_ids_42 = train_test_split(
    df_summary['subgraph_id'].tolist(), test_size=0.20, random_state=42, stratify=df_summary[TARGET_COL]
)
_, test_ids_101 = train_test_split(
    df_summary['subgraph_id'].tolist(), test_size=0.20, random_state=101, stratify=df_summary[TARGET_COL]
)

print(f"Overlap: {len(set(test_ids_42).intersection(set(test_ids_101)))}")
