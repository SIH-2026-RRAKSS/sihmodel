import torch
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.graphsage_classifier import load_all_graphs_dataset
from src.ibm_graphsage_classifier import load_or_create_ibm_pyg_dataset, TARGET_COL

raw_dataset_a, df_summary_a = load_all_graphs_dataset(Path('data/graph_summary.csv'))
raw_dataset_ibm, df_summary_ibm = load_or_create_ibm_pyg_dataset()

for seed in [42, 101, 2024, 7, 99]:
    _, test_ids_a = train_test_split(
        df_summary_a['complaint_id'].tolist(), test_size=0.20, random_state=seed, stratify=df_summary_a['contains_suspicious_activity']
    )
    test_set_a = set(test_ids_a)
    test_raw_a = [d for d in raw_dataset_a if d.complaint_id in test_set_a]
    test_raw_multi_a = [d for d in test_raw_a if d.num_nodes > 1]
    print(f"Dataset A seed {seed:>4}: full test N={len(test_raw_a)}, multi-node N={len(test_raw_multi_a)}")

for seed in [42, 101, 2024, 7, 99]:
    _, test_ids_ibm = train_test_split(
        df_summary_ibm['subgraph_id'].tolist(), test_size=0.20, random_state=seed, stratify=df_summary_ibm[TARGET_COL]
    )
    test_set_ibm = set(test_ids_ibm)
    test_raw_ibm = [d for d in raw_dataset_ibm if d.subgraph_id in test_set_ibm]
    test_raw_multi_ibm = [d for d in test_raw_ibm if d.num_nodes > 1]
    print(f"IBM seed {seed:>4}: full test N={len(test_raw_ibm)}, multi-node N={len(test_raw_multi_ibm)}")
