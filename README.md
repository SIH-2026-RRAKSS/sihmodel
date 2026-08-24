# Cybercrime Predictive Analytics — Stages 0, 1, 2 & 3A

This repository implements the end-to-end data consolidation, graph construction, and machine-learning baseline stages of a predictive analytics framework designed for cybercrime complaints, entity resolution, and financial transaction graph analysis.

---

## 📌 Project Context & Pipeline Architecture

The broader vision of this project is to build transaction graphs, identify suspicious mule account networks, predict illicit cash-withdrawal hubs, and provide actionable intelligence to law enforcement.

```text
complaint (complaints.csv)
   ↓
Stage 0: Entity Resolution (Deterministic + Toy Fuzzy)
   ↓
resolved entity (predicted_entity_id in resolved_entities.csv & entity_master.csv)
   ↓
72-hour transaction window (transactions.csv & entity_locations.csv)
   ↓
3-hop neighborhood extraction (<= 3 hops from incident node)
   ↓
NetworkX incident subgraph (MultiDiGraph saved to data/graphs/<complaint_id>.graphml)
   ↓
graph-level features (data/graph_summary.csv)
   ↓
Stage 3A: XGBoost Baseline Classifier (models/xgboost_baseline.json)
   ↓
[Next Stage: Stage 3B — GraphSAGE Graph Neural Network]
```

---

## 🏛️ Stage 0 — Entity Resolution

Stage 0 collapses repeated complaint references to the same bank account across different police stations, districts, and dates into a **single resolved financial entity node** (`predicted_entity_id`).

### Layer 1: Deterministic Matching (Primary Key)
- **Composite Key**: `normalized_account_number + "_" + normalized_ifsc`
- **Account Normalization**: Converts account numbers to stripped 12-digit strings.
- **IFSC Normalization**: Converts IFSC codes to uppercase strings with whitespace stripped.
- **Assignment**: All complaints sharing the exact same `(account_number, ifsc)` composite key are deterministically mapped to the same `predicted_entity_id` and tagged with `match_type = EXACT`.

### Layer 2: Toy Fuzzy Name Matching (Secondary / Fallback Signal)
- **Tooling**: Built using the `RapidFuzz` library (`rapidfuzz.fuzz.token_sort_ratio`).
- **Name Normalization**: Lowercase, punctuation stripped, whitespace collapsed.
- **Application**: Used exclusively as a secondary signal for records where bank account or IFSC data is incomplete or unindexed.
- **Thresholding**:
  - Similarity $\ge 90\% \rightarrow$ tagged as `match_type = FUZZY_CANDIDATE`.
  - Similarity $< 90\% \rightarrow$ tagged as `match_type = UNRESOLVED`.
- **Safety Principle**: A fuzzy name match **never** overrides or breaks a deterministic `account_number + IFSC` identity match.

---

## 🔑 Entity Master Mapping (`data/entity_master.csv`)

`entity_master.csv` provides a stable, deduplicated reference bridge between Stage 0 entity resolution and subsequent transaction graph construction:

- **Stable Entity IDs**: `entity_id` is an internal system identifier (`ENT_000001` to `ENT_000700`) assigned sequentially and reproducibly across runs.
- **Deterministic Identity Key**: `identity_key` (`normalized_account_number + "_" + normalized_ifsc`) serves as the unique invariant key.
- **Canonical Name**: For each entity, a canonical account-holder name is selected (preferring the most frequent original name associated with that entity) for descriptive metadata.
- **Ground Truth Isolation**: `ground_truth_entity_id` is used **only** for offline evaluation and never influences entity master generation or resolution logic.

---

## 💳 Stage 1 — Synthetic Transaction Dataset

The transaction generator (`src/generate_transactions.py`) produces realistic synthetic financial transactions representing activity between the 700 resolved entities from Stage 0.

### Key Dataset Characteristics
- **Total Transactions**: Exactly **15,000 transactions** across the period `2026-01-01` to `2026-08-24`.
- **Entity Universe**: Uses the exact 700 master financial entities from `data/entity_master.csv`.
- **Class Balance**: 
  - **Normal Transactions**: ~81.4% (12,216 transactions) representing everyday financial behavior.
  - **Suspicious Transactions**: ~18.6% (2,784 transactions) representing organized mule ring activity across 25 distinct rings (`RING_001` to `RING_025`).
- **Topologies**: Linear Layering, Fan-in Aggregation, Fan-out Dispersion, Layered Network Mesh, and Multi-stage Chains with Terminal Cash-outs.
- **Terminal Cash-Out Nodes**: Dedicated namespace (`ATM_001` to `ATM_050`) mapped to physical Indian cities.
- **Geographic Mapping (`data/entity_locations.csv`)**: Deterministic geographic coordinates across 15 Indian cities.

---

## 🕸️ Stage 2 — Graph Construction & Incident Subgraph Extraction

Stage 2 (`src/graph_construction.py`) builds temporal, multi-hop incident subgraphs around every cybercrime complaint.

- **Incident Identification**: For each complaint, `predicted_entity_id` from `resolved_entities.csv` serves as the root node (`is_incident = True`).
- **Fixed 72-Hour Incident Window**: Anchored at `complaint_date 00:00:00` $\pm 72$ hours. *(Adaptive activity-based windowing is designated as future work).*
- **Directed MultiDiGraph (`networkx.MultiDiGraph`)**: Preserves multiple transactions between the same entities as distinct directed edges.
- **3-Hop Neighborhood**: Retains all nodes within $\le 3$ hops in the undirected projection to capture both upstream fund sources and downstream mule flow.
- **Typed Node Schema**: `ACCOUNT` nodes (`ENT_XXXXXX`) and `ATM` nodes (`ATM_XXX`, `is_terminal = True`).
- **Edge Attributes**: `transaction_id`, `amount`, `timestamp`, `transaction_type`, `channel`, `is_cash_out`, `is_suspicious` (ground truth), `ring_id` (ground truth).
- **Storage**: GraphML files (`data/graphs/<complaint_id>.graphml`) and tabular features (`data/graph_summary.csv`).

---

## 🤖 Stage 3A — XGBoost Baseline Classifier

Stage 3A (`src/xgboost_baseline.py`) trains a tabular gradient-boosted decision tree classifier on graph-level and financial features to establish a reliable machine-learning baseline before comparing against Graph Neural Networks (GraphSAGE).

### 1. Problem Formulation & Target
- **Target**: `contains_suspicious_activity` (0 = Normal Incident Graph, 1 = Suspicious Incident Graph / Potential Laundering Network).
- **Dataset**: 1,000 incident subgraphs from `data/graph_summary.csv`.

### 2. Strict Data Leakage Prevention
The model must predict suspicious activity exclusively from structural and financial features. All direct labels and non-feature identifiers are strictly excluded from model inputs:
- `contains_suspicious_activity` (target)
- `suspicious_ring_count` (ground truth label)
- `is_suspicious` / `ring_id` (ground truth metadata)
- `ground_truth_entity_id` (offline evaluation)
- `complaint_id`, `incident_entity_id`, `incident_time`, `window_start`, `window_end`

### 3. Model Features (15 Numerical Graph Metrics)
1. `num_nodes`
2. `num_edges`
3. `num_account_nodes`
4. `num_atm_nodes`
5. `num_terminal_nodes`
6. `max_hop`
7. `total_transaction_value`
8. `max_transaction_value`
9. `avg_transaction_value`
10. `num_cash_out_edges`
11. `in_degree_incident`
12. `out_degree_incident`
13. `density`
14. `number_of_connected_components`
15. `average_degree`

### 4. Training Methodology & Class Imbalance
- **Train / Test Split**: 80% Train (800 samples) / 20% Test (200 samples), strictly stratified on the target label (`random_state = 42`).
- **Class Imbalance**: The dataset contains 18.6% positive samples. To counter class imbalance, `scale_pos_weight = 4.3691` (derived from the training set only) is applied during tree construction.
- **Model Parameters**: `n_estimators = 200`, `max_depth = 4`, `learning_rate = 0.05`, `subsample = 0.8`, `colsample_bytree = 0.8`, `eval_metric = "logloss"`.

### 5. Evaluation Performance (Test Set @ Threshold = 0.50)
- **Accuracy**: **96.00%**
- **Precision**: **91.43%**
- **Recall**: **86.49%**
- **F1 Score**: **88.89%**
- **ROC-AUC**: **0.9790**
- **PR-AUC**: **0.9444**
- **Confusion Matrix**: TN = 160, FP = 3, FN = 5, TP = 32

### 6. Threshold Analysis (`data/xgboost_threshold_analysis.csv`)
| Threshold | Precision | Recall | F1 Score | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **0.10** | 76.74% | 89.19% | 82.50% | High-sensitivity triage |
| **0.30** | 84.21% | 86.49% | 85.33% | Balanced candidate pool |
| **0.50** | 91.43% | 86.49% | 88.89% | Standard decision threshold |
| **0.80** | 100.00% | 83.78% | **91.18%** | *Experimental optimal F1 threshold* |
| **0.90** | 100.00% | 78.38% | 87.88% | High-confidence alert filter |

### 7. Feature Importance (Gain Metric)
1. `total_transaction_value` (Gain: 26.21) — Elevated aggregate fund volume in the 72h window.
2. `num_nodes` (Gain: 13.54) — Network expansion and multi-account involvement.
3. `max_hop` (Gain: 13.39) — Multi-layering depth away from the incident account.
4. `num_edges` (Gain: 10.90) — Rapid transfer velocity.
5. `average_degree` (Gain: 6.65) — Mesh/hub connectivity among participating accounts.

### 8. Secondary Robustness: Chronological Temporal Evaluation
- Evaluated by training on the earliest 80% incidents (Jan 1 – Jul 6, 2026) and testing on the latest 20% incidents (Jul 6 – Aug 24, 2026).
- **Temporal Test Accuracy**: 96.00% | **Precision**: 88.57% | **Recall**: 88.57% | **F1 Score**: 88.57% | **ROC-AUC**: 0.9932 | **PR-AUC**: 0.9709.

---

## ⚠️ Limitations & Real-World Considerations

1. **Synthetic Nature**: All complaints, bank accounts, transactions, and mule rings are synthetically generated for prototype development and hackathon evaluation.
2. **Benchmark Scope**: The XGBoost baseline demonstrates strong performance on this structured benchmark; however, real-world cybercrime involves complex adversarial obfuscation, smurfing, and cross-border crypto off-ramps.
3. **Threshold Context**: The 0.50 threshold is standard for benchmarking, while the 0.80 threshold yielded the highest experimental F1. In real-world law enforcement triage, operational thresholds must be calibrated based on analyst caseload capacity and tolerance for false positives.
4. **Relational Limitation of Tabular Models**: XGBoost operates on aggregated graph-level summary metrics. It cannot learn topological node-level embeddings, localized edge orientations, or directional flow dynamics natively. This serves as the primary motivation for **Stage 3B: GraphSAGE Graph Neural Network**.

---

## 📁 Project Structure

```text
sihmodel/
│
├── data/
│   ├── complaints.csv                     # Input cybercrime complaints (1,000 records)
│   ├── entity_ground_truth.csv            # Ground truth entity mappings (700 entities)
│   ├── resolved_entities.csv              # Resolved entities mapping table
│   ├── entity_master.csv                  # Stable entity master table (700 unique entities)
│   ├── entity_resolution_summary.csv      # Stage 0 performance summary metrics
│   ├── entity_locations.csv               # Geographic coordinates for 700 entities
│   ├── transactions.csv                   # 15,000 synthetic financial transactions
│   ├── graph_summary.csv                  # 1,000 incident subgraph metrics
│   ├── xgboost_predictions.csv            # Model test set predictions & probabilities
│   ├── xgboost_threshold_analysis.csv     # Threshold sweep evaluation
│   ├── xgboost_feature_importance.csv     # Gain-based feature importances
│   ├── xgboost_temporal_evaluation.csv    # Chronological temporal evaluation
│   ├── xgboost_pr_curve.png               # Precision-Recall curve
│   ├── xgboost_roc_curve.png              # ROC curve
│   ├── xgboost_feature_importance.png     # Feature importance plot
│   └── graphs/                            # Extracted incident subgraphs (GraphML)
│       ├── C000001.graphml ... C001000.graphml
│       └── demo_graph.png                 # Demonstration subgraph visualization
│
├── models/
│   ├── xgboost_baseline.json              # Serialized trained XGBoost model
│   └── xgboost_features.json              # Feature schema definition
│
├── src/
│   ├── entity_resolution.py               # Stage 0 Entity Resolution engine
│   ├── generate_transactions.py           # Stage 1 Transaction dataset generator
│   ├── graph_construction.py              # Stage 2 Incident subgraph extraction engine
│   └── xgboost_baseline.py                # Stage 3A XGBoost baseline classifier
│
├── generate_complaints_dataset.py         # Synthetic complaint generator
├── requirements.txt                       # Project dependencies
└── README.md                              # System documentation
```

---

## 🚀 How to Run

### 1. Run Entity Resolution (Stage 0)
```bash
python3 src/entity_resolution.py
```

### 2. Generate Synthetic Transactions & Locations (Stage 1)
```bash
python3 src/generate_transactions.py
```

### 3. Extract Incident Subgraphs & Graph Features (Stage 2)
```bash
python3 src/graph_construction.py
```

### 4. Train & Evaluate XGBoost Baseline (Stage 3A)
```bash
python3 src/xgboost_baseline.py
```

---

## 📊 Summary Performance Across Stages

| Stage | Task / Model | Primary Metric | Result | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Stage 0** | Deterministic + Fuzzy Entity Resolution | Pairwise F1 Score | **100.00%** | COMPLETE |
| **Stage 1** | Synthetic Financial Transaction Dataset | Transaction Volume | **15,000 Tx** | COMPLETE |
| **Stage 2** | 72h 3-Hop Incident Subgraph Extraction | Incident Graphs | **1,000 Graphs** | COMPLETE |
| **Stage 3A** | Tabular XGBoost Baseline Classifier | Test PR-AUC / F1 | **0.9444 / 88.89%** | COMPLETE |
| **Stage 3B** | GraphSAGE Graph Neural Network | Graph Classification | *Pending* | NEXT |

---

## ⚡ Engineering & Scalability Notes

1. **NetworkX vs Production Graph Stores**: NetworkX is used for prototype-scale graph construction and feature extraction. At national transaction scale, an indexed graph store (e.g., Neo4j) or distributed graph framework (e.g., Apache Spark GraphX / PyG) would handle temporal neighbor sampling.
2. **Feature Explainability**: Rule-based feature summaries provide human-interpretable rationale alongside model probabilities for investigating officers.
3. **Reproducibility**: All data generation, splitting, and model training routines use fixed random seeds (`seed = 42`).
