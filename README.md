# Cybercrime Predictive Analytics — Stage 0, Stage 1 & Stage 2

This repository implements the foundational and graph-processing stages of a predictive analytics framework designed for cybercrime complaints, entity resolution, and financial transaction graph analysis.

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
graph features (data/graph_summary.csv)
   ↓
[Future Stages: XGBoost baseline / GraphSAGE GNN]
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
- **Graph Connection**: In Stage 2, `entity_master.csv` is used to connect raw bank transaction streams (`transactions.csv`) directly to resolved entity nodes in the transaction graph.

---

## 💳 Stage 1 — Synthetic Transaction Dataset

The transaction generator (`src/generate_transactions.py`) produces realistic synthetic financial transactions representing activity between the 700 resolved entities from Stage 0.

### Key Dataset Characteristics
- **Total Transactions**: Exactly **15,000 transactions** across the period `2026-01-01` to `2026-08-24`.
- **Entity Universe**: Uses the exact 700 master financial entities from `data/entity_master.csv`.
- **Class Balance**: 
  - **Normal Transactions**: ~81.4% (12,216 transactions) representing everyday financial behavior (peer-to-peer, merchant purchases, bill payments, and normal ATM withdrawals).
  - **Suspicious Transactions**: ~18.6% (2,784 transactions) representing organized mule ring activity.
- **Suspicious Rings**: 25 distinct multi-hop mule rings (`RING_001` to `RING_025`) exhibiting realistic laundering topologies:
  1. *Linear Layering*: $E_0 \rightarrow E_1 \rightarrow E_2 \rightarrow \dots \rightarrow \text{ATM}$
  2. *Fan-in Aggregation*: Multiple source mules $\rightarrow$ Aggregator $\rightarrow \text{ATM}$
  3. *Fan-out Dispersion*: Primary inlet $\rightarrow$ Multiple mules $\rightarrow \text{ATMs}$
  4. *Layered Network Mesh*: Bipartite/mesh forwarding among mules before cash-out
  5. *Multi-stage Mule Chains with Terminal Cash-outs*
- **Temporal Windows**: Suspicious ring transactions occur in rapid multi-hop progression (15–90 minutes between hops) within strict $\le 72$-hour incident windows.
- **Terminal Cash-Out Nodes**: Represented using a dedicated namespace (`ATM_001` to `ATM_050`) mapped to physical Indian cities.
- **Geographic Mapping (`data/entity_locations.csv`)**: Deterministic geographic coordinates across 15 Indian cities.

---

## 🕸️ Stage 2 — Graph Construction & Incident Subgraph Extraction

Stage 2 (`src/graph_construction.py`) builds temporal, multi-hop incident subgraphs around every cybercrime complaint.

### Architecture & Extraction Methodology
1. **Incident Identification**: For each complaint, `predicted_entity_id` is looked up from `resolved_entities.csv` to serve as the root node (`is_incident = True`).
2. **Fixed 72-Hour Incident Window**: 
   - Anchored at `complaint_date 00:00:00`.
   - Window: `[complaint_date - 72h, complaint_date + 72h]`.
   - Captures transactions occurring before and after the reported incident. *(Adaptive activity-based windowing is designated as future work).*
3. **Directed MultiDiGraph (`networkx.MultiDiGraph`)**:
   - Preserves multiple transactions between the same pair of entities as distinct directed edges.
4. **3-Hop Neighborhood Extraction**:
   - Computes shortest path distances in the undirected projection to capture both upstream fund sources and downstream mule flow.
   - Retains all nodes within $\le 3$ hops and induces the directed subgraph.
5. **Typed Node Schema**:
   - `ACCOUNT` nodes (`ENT_XXXXXX`): `canonical_name`, `state`, `city`, `latitude`, `longitude`, `is_incident`, `is_terminal = False`, `hop_distance`.
   - `ATM` nodes (`ATM_XXX`): `state`, `city`, `latitude`, `longitude`, `is_incident = False`, `is_terminal = True`, `hop_distance`.
6. **Edge Schema**:
   - Directed transaction edges with `transaction_id`, `amount`, `timestamp`, `transaction_type`, `channel`, `is_cash_out`, `is_suspicious` (ground truth), `ring_id` (ground truth).
   - Ground truth labels are preserved as metadata for evaluation and are **never** used to construct or filter the graph.
7. **Graph Storage & Output**:
   - **`data/graphs/<complaint_id>.graphml`**: Individual standard GraphML file for each of the 1,000 complaints.
   - **`data/graph_summary.csv`**: Comprehensive summary of 22 structural, degree, density, and financial metrics per graph.
   - **`data/graphs/demo_graph.png`**: Visual rendering of a representative incident subgraph.

---

## 📁 Project Structure

```text
sihmodel/
│
├── data/
│   ├── complaints.csv                   # Input cybercrime complaints (1,000 records)
│   ├── entity_ground_truth.csv          # Ground truth entity mappings (700 entities)
│   ├── resolved_entities.csv            # Resolved entities mapping table
│   ├── entity_master.csv                # Stable entity master table (700 unique entities)
│   ├── entity_resolution_summary.csv    # Stage 0 performance summary metrics
│   ├── entity_locations.csv             # Geographic coordinates for 700 entities
│   ├── transactions.csv                 # 15,000 synthetic financial transactions
│   ├── graph_summary.csv                # 1,000 incident subgraph metrics
│   └── graphs/                          # Extracted incident subgraphs (GraphML)
│       ├── C000001.graphml
│       ├── C000002.graphml
│       ├── ...
│       ├── C001000.graphml
│       └── demo_graph.png               # Demonstration visualization
│
├── src/
│   ├── entity_resolution.py             # Stage 0 Entity Resolution engine
│   ├── generate_transactions.py         # Transaction & location dataset generator
│   └── graph_construction.py            # Stage 2 Incident subgraph extraction engine
│
├── generate_complaints_dataset.py       # Synthetic complaint dataset generator
├── requirements.txt                     # Project dependencies
└── README.md                            # Documentation and architecture guide
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

### 3. Extract Incident Subgraphs & Summary (Stage 2)
```bash
python3 src/graph_construction.py
```

---

## 📊 Summary & Performance Metrics

### Stage 0 Performance
| Metric | Score | Simple Explanation |
| :--- | :---: | :--- |
| **Accuracy** | **100.00%** | All 499,500 complaint pairs correctly categorized. |
| **Precision** | **100.00%** | 0 false merges (no distinct accounts merged). |
| **Recall** | **100.00%** | 0 missed merges (all repeated accounts linked). |
| **F1 Score** | **100.00%** | Optimal harmonic mean of precision and recall. |
| **False Merges (FP)** | **0** | No cross-account merging errors. |
| **Missed Merges (FN)** | **0** | No split-account errors. |

### Transaction Dataset Statistics
- **Total Transactions**: 15,000
- **Unique Master Financial Entities**: 700
- **Normal Transactions**: 12,216 (81.4%)
- **Suspicious Transactions**: 2,784 (18.6%)
- **Suspicious Rings**: 25
- **Cash-Out Transactions**: 1,195 (8.0%)
- **Average Amount**: ₹46,784.46
- **Max Amount**: ₹498,797.66
- **Date Range**: 2026-01-01 05:13:57 → 2026-08-24 22:11:19

### Stage 2 Graph Construction Statistics
- **Total Complaints Processed**: 1,000
- **Total Incident Graphs Saved**: 1,000 GraphML files
- **Average Nodes per Graph**: 4.58
- **Average Edges per Graph**: 3.82
- **Maximum Nodes in a Graph**: 38
- **Maximum Edges in a Graph**: 43
- **Graphs with Suspicious Activity**: 186 (18.6%)
- **Graphs with ATM Cash-Out Edges**: 148 (14.8%)

---

## ⚡ Scalability & Engineering Notes

1. **NetworkX Scope**: NetworkX is used for prototype-scale graph construction. A production deployment at national transaction volumes would require an indexed graph database (e.g. Neo4j) or distributed graph-processing architecture (e.g. Apache Spark GraphX / DGL).
2. **Deterministic Resolution ($O(N)$)**: Exact `account_number + IFSC` matching operates as a hash-indexed join, scaling linearly with complaint volumes.
3. **Fuzzy Matching Consideration ($O(N^2)$)**: At enterprise scale, candidate blocking / canopy clustering (e.g. LSH, phonetic blocking) must precede pairwise fuzzy comparisons.
4. **Machine Learning Readiness**: The combination of `entity_master.csv`, `graph_summary.csv`, and GraphML subgraphs in `data/graphs/` provides clean tabular features, node embeddings, adjacency structures, and edge attributes ready for tabular baseline training (XGBoost) and Graph Neural Network training (GraphSAGE).

---

Stage 2: Graph Construction / Incident Subgraph Extraction has been implemented in graph_construction.py and executed to extract 72-hour 3-hop directed incident subgraphs across all 1,000 complaints.
  ──────
  ### Pipeline Execution Summary

    =======================================================
          STAGE 2 INCIDENT GRAPH EXTRACTION SUMMARY
    =======================================================
    Total Complaints Processed : 1000
    Total Incident Graphs Saved: 1000 GraphML files (data/graphs/)
    Average Nodes per Graph    : 4.58
    Average Edges per Graph    : 3.82
    Maximum Nodes in a Graph   : 38
    Maximum Edges in a Graph   : 43
    -------------------------------------------------------
    Graphs with Suspicious Activity: 186 (18.6%)
    Graphs with ATM Cash-Out Edges : 148 (14.8%)
    =======================================================
  ──────
  ### Generated Output Artifacts

  1. : 1,000 individual GraphML files (C000001.graphml through C001000.graphml) storing directed networkx.MultiDiGraph incident subgraphs with complete node and edge attributes.
  2. **graph_summary.csv**: 1,000 rows containing 22 topological, temporal, financial, and evaluation metrics per incident graph.
  3. **demo_graph.png**: Visual rendering of the demonstration incident subgraph.
  4. **requirements.txt**: Python dependencies specification.
  ──────
  ### 5 Sample Rows from data/graph_summary.csv

   complaint_id | incident_entity_… | incident_time       | window_start        | window_end          | num_nod… | num_edges | num_acco… | num_atm_… | max_hop | total_transa… | num_cash… | contains… | suspicio…
  --------------|-------------------|---------------------|---------------------|---------------------|----------|-----------|-----------|-----------|---------|---------------|-----------|-----------|-----------
   C000001      | ENT_000090        | 2026-07-20 00:00:00 | 2026-07-17 00:00:00 | 2026-07-23 00:00:00 |    1     |     0     |     1     |     0     |    0    |     ₹0.00     |     0     |     0     |     0
   C000002      | ENT_000451        | 2026-01-11 00:00:00 | 2026-01-08 00:00:00 | 2026-01-14 00:00:00 |    1     |     0     |     1     |     0     |    0    |     ₹0.00     |     0     |     0     |     0
   C000003      | ENT_000325        | 2026-04-22 00:00:00 | 2026-04-19 00:00:00 | 2026-04-25 00:00:00 |    12    |    13     |    10     |     2     |    3    | ₹1,061,795.51 |     2     |     1     |     1
   C000004      | ENT_000493        | 2026-06-02 00:00:00 | 2026-05-30 00:00:00 | 2026-06-05 00:00:00 |    4     |     3     |     4     |     0     |    2    |  ₹12,669.47   |     0     |     0     |     0
   C000005      | ENT_000650        | 2026-06-07 00:00:00 | 2026-06-04 00:00:00 | 2026-06-10 00:00:00 |    1     |     0     |     1     |     0     |    0    |     ₹0.00     |     0     |     0     |     0
  ──────
  ### Demonstration Graph Details: Complaint C000014

  • Incident Entity (Root): ENT_000292
  • Time Window: 2026-05-29 00:00:00 to 2026-06-04 00:00:00
  • Graph Topology: 18 Nodes (16 Accounts, 2 ATMs), 19 Edges, Max Hop = 3, Total Value = ₹1,629,314.82
  • Rings Observed: RING_008, RING_011

  #### Example Nodes from C000014.graphml:

  • ENT_000292: node_type="ACCOUNT", is_incident=True, hop_distance=0, city="Bengaluru", name="Sneha Choudhury"
  • ENT_000093: node_type="ACCOUNT", is_incident=False, hop_distance=1, city="Delhi", name="Sunil Sharma"
  • ENT_000573: node_type="ACCOUNT", is_incident=False, hop_distance=2, city="Ahmedabad", name="Swati Dubey"
  • ATM_018: node_type="ATM", is_terminal=True, is_incident=False, hop_distance=1, city="Bengaluru"
  • ATM_023: node_type="ATM", is_terminal=True, is_incident=False, hop_distance=3, city="Ahmedabad"

  #### Example Edges from C000014.graphml:

  • ENT_000093 → ENT_000292 : ₹98,548.39 via NEFT | is_cash_out=0 | RING_008 (Mule fund transfer to incident node)
  • ENT_000292 → ATM_018 : ₹95,538.10 via CASH_WITHDRAWAL | is_cash_out=1 | RING_008 (Direct cash-out exit)
  • ENT_000093 → ENT_000573 : ₹93,884.20 via UPI | is_cash_out=0 | RING_008 (Fan-out dispersion)
  • ENT_000573 → ATM_023 : ₹89,703.74 via CASH_WITHDRAWAL | is_cash_out=1 | RING_008 (Parallel terminal cash-out)
  • ENT_000610 → ENT_000573 : ₹445,550.06 via RTGS | is_cash_out=0 | NORMAL (Benign baseline transfer)
  ──────
  ### Validation Suite Status

  All 11 automated validation checks passed:

  1. Every complaint has a valid predicted_entity_id resolved from Stage 0.
  2. Every predicted_entity_id and non-ATM node belongs strictly to entity_master.csv.
  3. Every graph has exactly one root incident entity (is_incident == True).
  4. All nodes strictly respect hop_distance ≤ 3.
  5. All graph edges correspond to transactions from transactions.csv within the 72-hour window.
  6. ATM nodes are typed (node_type="ATM", is_terminal=True).
  7. GraphML serialization and deserialization roundtrip verified.
  8. Ground-truth labels (is_suspicious, ring_id) were not used to construct or filter the graph.
  ──────
  ### Confirmation: Readiness for Machine Learning

  Stage 2 Graph Construction is complete, verified, and ready for Stage 3 (XGBoost Baseline and GraphSAGE GNN):

  • Tabular graph metrics in graph_summary.csv provide structural and financial features for XGBoost.
  • MultiDiGraph subgraphs in  provide node features, edge weights, and adjacency matrices for Graph Neural Network training.
    
