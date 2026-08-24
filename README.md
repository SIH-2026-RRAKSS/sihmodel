# Cybercrime Predictive Analytics — Stage 0 & Stage 1/2

This repository implements the foundational stages of a predictive analytics framework designed for cybercrime complaints and financial transaction graph analysis.

---

## 📌 Project Context & Pipeline Architecture

The broader vision of this project is to build transaction graphs, identify suspicious mule account networks, predict illicit cash-withdrawal hubs, and provide actionable intelligence to law enforcement.

```text
complaints.csv
      ↓
Stage 0: Entity Resolution (Deterministic + Toy Fuzzy)
      ↓
resolved_entities.csv ──> entity_master.csv (700 Master Entity Nodes)
                                ↓
                      transactions.csv (15,000 Transactions)
                      entity_locations.csv (Geographic Coordinates)
                                ↓
                      Stage 2+: Graph Construction (GraphSAGE / XGBoost)
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

`entity_master.csv` provides a stable, deduplicated reference bridge between Stage 0 entity resolution and subsequent transaction graph construction in Stage 2:

- **Stable Entity IDs**: `entity_id` is an internal system identifier (`ENT_000001` to `ENT_000700`) assigned sequentially and reproducibly across runs.
- **Deterministic Identity Key**: `identity_key` (`normalized_account_number + "_" + normalized_ifsc`) serves as the unique invariant key.
- **Canonical Name**: For each entity, a canonical account-holder name is selected (preferring the most frequent original name associated with that entity) for descriptive metadata.
- **Ground Truth Isolation**: `ground_truth_entity_id` is used **only** for offline evaluation and never influences entity master generation or resolution logic.
- **Stage 2 Interface**: In Stage 2, `entity_master.csv` is used to connect raw bank transaction streams (`transactions.csv`) directly to resolved entity nodes in the transaction graph.

---

## 💳 Stage 1/2 — Synthetic Transaction Dataset

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
- **Geographic Mapping (`data/entity_locations.csv`)**: Deterministic geographic coordinates across 15 Indian cities (Kolkata, Mumbai, Pune, Delhi, Bengaluru, Hyderabad, Chennai, Ahmedabad, Jaipur, Lucknow, Patna, Bhubaneswar, Kochi, Bhopal, Chandigarh).

> **Disclaimer**: The transaction dataset, mule rings, and geographic coordinates are 100% synthetic for prototype graph modeling and ML benchmarking. They do not contain real banking information and do not represent real-world individuals or authorized banking data.

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
│   └── transactions.csv                 # 15,000 synthetic financial transactions
│
├── src/
│   ├── entity_resolution.py             # Stage 0 Entity Resolution engine
│   └── generate_transactions.py         # Transaction & location dataset generator
│
├── generate_complaints_dataset.py       # Synthetic complaint dataset generator
└── README.md                            # Documentation and architecture guide
```

---

## 🚀 How to Run

### 1. Run Entity Resolution (Stage 0)
```bash
python3 src/entity_resolution.py
```

### 2. Generate Synthetic Transactions & Locations
```bash
python3 src/generate_transactions.py
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

---

## ⚡ Scalability & Engineering Notes

1. **Deterministic Resolution ($O(N)$)**: Exact `account_number + IFSC` matching operates as a hash-indexed join, scaling linearly with complaint volumes.
2. **Fuzzy Matching Consideration ($O(N^2)$)**: At enterprise scale, candidate blocking / canopy clustering (e.g. LSH, phonetic blocking) must precede pairwise fuzzy comparisons.
3. **Graph Readiness**: The combination of `entity_master.csv`, `transactions.csv`, and `entity_locations.csv` provides clean node tables, edge lists, timestamps, and spatial attributes ready for 72-hour incident subgraph extraction and Graph Neural Network training.


---

The synthetic financial transaction generator has been implemented in generate_transactions.py and executed to produce transactions.csv and entity_locations.csv.
  ──────
  ### Key Highlights of the Generator
  1. Entity Alignment & Isolation:
      • Uses the exact 700 master financial entities (ENT_000001–ENT_000700) from entity_master.csv.
      • Cash-out terminal nodes use a distinct namespace (ATM_001–ATM_050) mapped to physical Indian cities.
      • ground_truth_entity_id is never accessed. is_suspicious and ring_id serve strictly as ground-truth evaluation labels.
  2. Transaction Size & Class Distribution:
      • Total Transactions: Exactly 15,000.
      • Normal Transactions: 12,216 (~81.4%) across all 700 entities, with realistic diurnal curves and amount distributions.
      • Suspicious Transactions: 2,784 (~18.6%) across 25 distinct mule rings (RING_001–RING_025).
  3. Mule Ring Topologies & Temporal Dynamics:
      • Implements 5 distinct graph patterns:
          • Linear Layering: E₀ → E₁ → E₂ → … → ATM
          • Fan-in Aggregation: Multiple source mules → Aggregator →ATM
          • Fan-out Dispersion: Inlet → Multiple downstream mules →ATMs
          • Layered Network Mesh: Multi-path forwarding before aggregation and cash-out
          • Multi-Stage Chains with Terminal Cash-Outs
      • Transactions within each laundering burst occur in rapid sequence (15–90 minutes between hops) within strict ≤72-hour incident windows.
      • Amounts decrease progressively along chains to simulate mule commission / fee retention.
  4. Geographic Coordinates (data/entity_locations.csv):
      • Generates synthetic coordinates across 15 Indian cities (Kolkata, Mumbai, Pune, Delhi, Bengaluru, Hyderabad, Chennai, Ahmedabad, Jaipur, Lucknow, Patna, Bhubaneswar, Kochi, Bhopal, Chandigarh).

  ──────
  ### Actual Execution Summary Statistics

    ================================================
          SYNTHETIC TRANSACTION DATASET
    ================================================
    Total transactions         : 15000
    Unique financial entities  : 700
    Suspicious transactions    : 2784 (18.6%)
    Normal transactions        : 12216 (81.4%)
    Suspicious rings           : 25
    Cash-out transactions      : 1195 (8.0%)
    ------------------------------------------------
    Date range:
    2026-01-01 05:13:57 → 2026-08-24 22:11:19
    ------------------------------------------------
    Average transaction amount : ₹46,784.46
    Maximum transaction amount : ₹498,797.66
    ------------------------------------------------
    Transaction type distribution:
      - UPI                :  7241 (48.3%)
      - IMPS               :  3268 (21.8%)
      - NEFT               :  2214 (14.8%)
      - CASH_WITHDRAWAL    :  1195 ( 8.0%)
      - CARD               :   857 ( 5.7%)
      - RTGS               :   225 ( 1.5%)
    ================================================
  ──────
  ### Sample 10 Transactions from data/transactions.csv

   Tx ID                 | Sender                | Receiver              |     Amount (INR)     | Timestamp            | Type                 |       CashOut        |         Susp         | Ring
  -----------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------
   T000001               | ENT_000550            | ENT_000412            |     ₹322,804.63      | 2026-01-01 05:13:57  | RTGS                 |          0           |          0           | NORMAL
   T000002               | ENT_000418            | ENT_000613            |      ₹55,186.74      | 2026-01-01 06:18:00  | UPI                  |          0           |          1           | RING_002
   T000003               | ENT_000395            | ENT_000101            |      ₹3,222.29       | 2026-01-01 06:22:21  | IMPS                 |          0           |          0           | NORMAL
   T000004               | ENT_000178            | ENT_000339            |      ₹2,050.10       | 2026-01-01 06:34:24  | IMPS                 |          0           |          0           | NORMAL
   T000005               | ENT_000632            | ENT_000370            |       ₹503.03        | 2026-01-01 06:46:46  | UPI                  |          0           |          0           | NORMAL
   T000006               | ENT_000106            | ENT_000613            |      ₹52,785.33      | 2026-01-01 06:47:00  | IMPS                 |          0           |          1           | RING_002
   T000007               | ENT_000532            | ENT_000616            |      ₹3,437.70       | 2026-01-01 06:51:12  | UPI                  |          0           |          0           | NORMAL
   T000008               | ENT_000027            | ENT_000613            |      ₹57,212.66      | 2026-01-01 07:02:00  | IMPS                 |          0           |          1           | RING_002
   T000009               | ENT_000121            | ENT_000502            |      ₹4,388.24       | 2026-01-01 07:16:18  | UPI                  |          0           |          0           | NORMAL
   T000010               | ENT_000092            | ENT_000613            |      ₹54,686.88      | 2026-01-01 07:17:00  | UPI                  |          0           |          1           | RING_002
  ──────
  ### 5 Example Suspicious Multi-Hop Sequences (RING_001)

   Tx ID                       | Flow                        |        Amount (INR)         | Timestamp                   | Type                        |           CashOut           | Ring
  -----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------
   T000165                     | ENT_000084 → ENT_000539     |         ₹209,080.81         | 2026-01-03 11:30:05         | NEFT                        |              0              | RING_001
   T000167                     | ENT_000539 → ENT_000458     |         ₹199,053.09         | 2026-01-03 11:59:05         | UPI                         |              0              | RING_001
   T000169                     | ENT_000458 → ENT_000541     |         ₹190,406.54         | 2026-01-03 12:14:05         | IMPS                        |              0              | RING_001
   T000170                     | ENT_000541 → ENT_000371     |         ₹181,231.52         | 2026-01-03 12:52:05         | NEFT                        |              0              | RING_001
   T000174                     | ENT_000371 → ENT_000080     |         ₹177,247.53         | 2026-01-03 13:49:05         | IMPS                        |              0              | RING_001
  ──────
  ### 5 Example Normal Transactions

   Tx ID                       | Flow                        |        Amount (INR)         | Timestamp                   | Type                        |           CashOut           | Ring
  -----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------
   T000001                     | ENT_000550 → ENT_000412     |         ₹322,804.63         | 2026-01-01 05:13:57         | RTGS                        |              0              | NORMAL
   T000003                     | ENT_000395 → ENT_000101     |          ₹3,222.29          | 2026-01-01 06:22:21         | IMPS                        |              0              | NORMAL
   T000004                     | ENT_000178 → ENT_000339     |          ₹2,050.10          | 2026-01-01 06:34:24         | IMPS                        |              0              | NORMAL
   T000005                     | ENT_000632 → ENT_000370     |           ₹503.03           | 2026-01-01 06:46:46         | UPI                         |              0              | NORMAL
   T000007                     | ENT_000532 → ENT_000616     |          ₹3,437.70          | 2026-01-01 06:51:12         | UPI                         |              0              | NORMAL
  ──────
  ### Validation Suite Status

  All 14 automated validation assertions passed:

  • transaction_id unique format (T000001–T015000).
  • Senders and receivers strictly validated against entity_master.csv (700 entities) and ATM nodes (ATM_001–ATM_050).
  • No self-transactions (sender != receiver).
  • All amounts >0.
  • Timestamps verified within 2026-01-01 to 2026-08-24.
  • All 700 entities active in transactions.
  • All 25 suspicious rings verified with multi-hop sequences within ≤72 hours.
  • Dataset is 100% reproducible with random seed 42.

  ---
  ----
