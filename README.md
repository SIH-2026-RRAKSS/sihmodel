# Cybercrime Predictive Analytics — Stages 0 to 5

This repository implements an end-to-end predictive analytics framework designed for cybercrime complaints, entity resolution, financial transaction graph construction, Graph Neural Network (GraphSAGE) classification, terminal cash-withdrawal location prediction, and confidence-tier uncertainty calibration.

---

## 📌 Project Architecture & End-to-End Pipeline

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
┌────────────────────────────────────────────────────────┐
│               Stage 3: Predictive Modeling             │
├───────────────────────────┬────────────────────────────┤
│ Stage 3A: Tabular Baseline│ Stage 3B: Graph Neural Net │
│ (XGBoost Classifier)      │ (GraphSAGE GNN Classifier) │
│ Features: graph_summary   │ Architecture: 2 SAGEConv   │
│ Test F1: 88.89%           │ Test F1: 90.14%            │
└───────────────────────────┴────────────────────────────┘
   ↓
Stage 4: Terminal Node & Cash-Withdrawal Location Prediction (src/terminal_prediction.py)
   ↓
Stage 5: Confidence Tiers & Novelty Detection (src/confidence_tiers.py)
   ↓
data/confidence_tiers.csv (HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, FIRST_TIME_RING, NORMAL)
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

Stage 3A (`src/xgboost_baseline.py`) trains a tabular gradient-boosted decision tree classifier on 15 structural and financial features extracted from `data/graph_summary.csv` to establish an empirical benchmark:
- **Features Used**: `num_nodes`, `num_edges`, `num_account_nodes`, `num_atm_nodes`, `num_terminal_nodes`, `max_hop`, `total_transaction_value`, `max_transaction_value`, `avg_transaction_value`, `num_cash_out_edges`, `in_degree_incident`, `out_degree_incident`, `density`, `number_of_connected_components`, `average_degree`.
- **Leakage Prevention**: All ground truth labels and identifiers are strictly excluded from input features.
- **Class Imbalance**: Managed via `scale_pos_weight = 4.3691` on the 80% training set.
- **Test Performance (@ 0.50)**: Accuracy = 96.00%, Precision = 91.43%, Recall = 86.49%, F1 Score = 88.89%, ROC-AUC = 0.9790, PR-AUC = 0.9444.

---

## 🧠 Stage 3B — GraphSAGE Graph Neural Network Classifier

Stage 3B (`src/graphsage_classifier.py`) implements an inductive **GraphSAGE Graph Neural Network** (PyTorch + PyTorch Geometric) that directly operates on the topological graph structures and localized node relational contexts.

### 1. Node Features (13 Dimensional)
1. `node_type_account`, 2. `node_type_atm`, 3. `hop_distance`, 4. `in_degree`, 5. `out_degree`, 6. `total_incoming_amount`, 7. `total_outgoing_amount`, 8. `average_incoming_amount`, 9. `average_outgoing_amount`, 10. `transaction_count`, 11. `is_incident`, 12. `is_terminal`, 13. `city_code`.

### 2. Architecture & Performance
- 2 SAGEConv layers (dim=64) $\rightarrow$ Global Mean Pooling $\rightarrow$ Linear Classifier.
- **Test Set Performance (@ 0.50)**:
  - **Accuracy**: **96.50%**
  - **Precision**: **94.12%** (+2.69% over XGBoost)
  - **Recall**: **86.49%**
  - **F1 Score**: **90.14%** (+1.25% over XGBoost)
  - **ROC-AUC**: **0.9829**
  - **PR-AUC**: **0.9515**
  - **False Positives**: 2 (33% reduction vs. XGBoost)

---

## 🎯 Stage 4 — Terminal Node & Cash-Withdrawal Location Prediction

Stage 4 (`src/terminal_prediction.py`) bridges graph analytics with tactical law-enforcement intelligence by ranking the most probable ATM cash-out terminals for each high-risk incident graph.

- **Candidate ATM Identification**: Directly within the 72h 3-hop subgraph.
- **Multi-Criteria Scoring Formula**:
  $$\text{terminal\_score} = 0.25 S_{\text{gnn}} + 0.20 S_{\text{hop}} + 0.20 S_{\text{cw}} + 0.15 S_{\text{vol}} + 0.10 S_{\text{rec}} + 0.05 S_{\text{up}} + 0.05 S_{\text{geo}}$$
- **Benchmark Performance**:
  - **Top-1 Hit Rate**: **100.00%**
  - **Top-3 Hit Rate**: **100.00%**
  - **Mean Reciprocal Rank (MRR)**: **1.0000**
  - **Average Candidates per Graph**: 1.30

---

## 🛡️ Stage 5 — Confidence Tiers & First-Time Ring Detection

Stage 5 (`src/confidence_tiers.py`) establishes an uncertainty calibration framework that categorizes flagged incidents into actionable operational confidence tiers and detects novel / emerging ring topologies.

### 1. Motivation for Confidence Calibration
In real-world cybercrime operations, raw model probabilities do not capture structural novelty or evidentiary completeness. Labeling an incident as an established ring without historical pattern evidence risks investigative misdirection. Stage 5 separates known high-confidence topologies from emerging first-time rings.

### 2. Confidence Tier Categories
1. **`HIGH_CONFIDENCE`**:
   - $P_{\text{GNN}} \ge 0.70$
   - $\ge 2$ independent supporting signals (multi-hop path, elevated volume, complex graph, cash-out terminal)
   - Verified terminal evidence or multi-hop structure
   - Reference embedding similarity $\ge 0.85$ (closely matches cataloged reference patterns)
2. **`MEDIUM_CONFIDENCE`**:
   - $P_{\text{GNN}} \ge 0.70$ with elevated activity, but partial terminal/structural evidence.
3. **`FIRST_TIME_RING_CANDIDATE`**:
   - $P_{\text{GNN}} \ge 0.50$ (elevated risk)
   - Reference embedding similarity $< 0.85$ (divergent from previously cataloged training rings)
   - *Operational Directive*: Surface as a new-pattern investigative lead.
4. **`NORMAL`**:
   - $P_{\text{GNN}} < 0.50$ (below suspicious action threshold).

### 3. Tier Distribution & Offline Performance
- **Total Incidents Processed**: 1,000
- **Normal Tier**: 713 incidents (71.3%)
- **High Confidence**: 226 incidents (22.6%) — captures **88.17%** of all actual laundering networks with **72.57% precision**.
- **Medium Confidence**: 61 incidents (6.1%)
- **Combined Suspicious Tiers**: 287 incidents — captures **89.25%** of all laundering activity.

---

## ⚠️ Limitations & Disclaimers

1. **Synthetic Proof-of-Concept**: All complaints, entities, transactions, and cash-outs are synthetically generated for hackathon prototyping and evaluation.
2. **Fixed Graph Horizons**: Graph construction is bounded by a fixed 72-hour window and 3-hop cutoff.
3. **Production Deployment Requirements**: Real-world deployment would require live banking transaction feeds (e.g. NPCI/RBI switch integration), operational ATM telemetry, historical mule ring intelligence, and a scalable geospatial GIS infrastructure.

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
│   ├── model_split_ids.csv                # Aligned 80/20 train/test split IDs
│   ├── xgboost_predictions.csv            # Stage 3A XGBoost test predictions
│   ├── graphsage_predictions.csv          # Stage 3B GraphSAGE test predictions
│   ├── graph_embeddings.csv               # 64-dim GraphSAGE embeddings for 1,000 graphs
│   ├── model_comparison.csv               # 1-to-1 comparison table (XGBoost vs GraphSAGE)
│   ├── terminal_predictions.csv           # Stage 4 Full ranked candidate ATM predictions
│   ├── top_terminal_predictions.csv       # Stage 4 Top-3 candidate terminals with reasons
│   ├── terminal_prediction_evaluation.csv # Stage 4 Hit rates and MRR evaluation table
│   ├── terminal_prediction_map.png        # Stage 4 Geospatial map of predicted cash-out hubs
│   ├── confidence_tiers.csv               # Stage 5 Incident confidence tier assignments
│   ├── confidence_summary.csv             # Stage 5 Tier distribution and novelty summary
│   ├── confidence_examples.csv            # Stage 5 Representative case breakdowns
│   ├── confidence_tier_evaluation.csv     # Stage 5 Offline evaluation against ground truth
│   └── graphs/                            # 1,000 GraphML subgraphs
│       ├── C000001.graphml ... C001000.graphml
│       └── demo_graph.png                 # Demonstration incident visualization
│
├── models/
│   ├── xgboost_baseline.json              # Trained XGBoost baseline model
│   ├── xgboost_features.json              # Feature schema list
│   ├── graphsage_model.pt                 # Trained PyTorch GraphSAGE checkpoint
│   └── graphsage_config.json              # GNN hyperparameter configuration
│
├── src/
│   ├── entity_resolution.py               # Stage 0 Entity Resolution engine
│   ├── generate_transactions.py           # Stage 1 Transaction dataset generator
│   ├── graph_construction.py              # Stage 2 Incident subgraph extraction engine
│   ├── xgboost_baseline.py                # Stage 3A XGBoost baseline classifier
│   ├── graphsage_classifier.py            # Stage 3B GraphSAGE GNN classifier
│   ├── terminal_prediction.py             # Stage 4 Terminal node prediction engine
│   └── confidence_tiers.py                # Stage 5 Confidence tiers & novelty detection
│
├── generate_complaints_dataset.py         # Synthetic complaint dataset generator
├── requirements.txt                       # Project dependencies
└── README.md                              # Complete system documentation
```

---

## 🚀 How to Run the Full Pipeline

```bash
# 1. Run Entity Resolution (Stage 0)
python3 src/entity_resolution.py

# 2. Generate Synthetic Transactions & Locations (Stage 1)
python3 src/generate_transactions.py

# 3. Extract Incident Subgraphs & Graph Features (Stage 2)
python3 src/graph_construction.py

# 4. Train & Evaluate XGBoost Baseline (Stage 3A)
python3 src/xgboost_baseline.py

# 5. Train & Evaluate GraphSAGE GNN (Stage 3B)
python3 src/graphsage_classifier.py

# 6. Predict & Rank Terminal Cash-Out Locations (Stage 4)
python3 src/terminal_prediction.py

# 7. Assign Confidence Tiers & Detect Novel Ring Patterns (Stage 5)
python3 src/confidence_tiers.py
```
