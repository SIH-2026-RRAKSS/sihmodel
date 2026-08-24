# Cybercrime Predictive Analytics — Stage 0: Entity Resolution

This repository implements **Stage 0: Entity Resolution**, the foundational layer for a predictive analytics framework designed to analyze cybercrime complaint datasets.

---

## 📌 Project Context & Objective

The broader vision of the project is to build transaction graphs, identify suspicious mule account networks, predict illicit cash-withdrawal hubs, and provide actionable intelligence to law enforcement.

**Stage 0 is the foundational data consolidation layer.** When citizens and victims file cybercrime complaints across different police stations, districts, and dates, the same fraudulent mule bank account may be reported multiple times under slightly different victim statements or naming variations.

The goal of Stage 0 is to collapse repeated complaint references to the same bank account into a **single resolved financial entity/node** (`predicted_entity_id`) that can later form nodes in downstream graph neural networks (e.g., GraphSAGE) and transaction flow analyses:

```text
Complaint C000027 ─┐
Complaint C000370 ─┼──> Resolved Entity (ENT_000373) ──> Node in Transaction Graph
Complaint C000568 ─┘
```

---

## 🏛️ Architecture & Pipeline Flow

The overall system architecture bridges cybercrime complaint intake with downstream transaction graph construction:

```text
complaints.csv
      ↓
Entity Resolution (Deterministic + Toy Fuzzy)
      ↓
resolved_entities.csv
      ↓
entity_master.csv
      ↓
transactions.csv (Stage 1)
      ↓
Transaction Graph (Stage 1+)
```

### Layer 1: Deterministic Matching (Primary Key)
- **Composite Key**: `normalized_account_number + "_" + normalized_ifsc`
- **Account Normalization**: Converts account numbers to stripped 12-digit strings.
- **IFSC Normalization**: Converts IFSC codes to uppercase strings with whitespace stripped.
- **Assignment**: All complaints sharing the exact same `(account_number, ifsc)` composite key are deterministically mapped to the same `predicted_entity_id` and tagged with `match_type = EXACT`.

### Layer 2: Toy Fuzzy Name Matching (Secondary / Fallback Signal)
- **Tooling**: Built using the `RapidFuzz` library (`rapidfuzz.fuzz.token_sort_ratio`).
- **Name Normalization**: Names are converted to lowercase, special characters/punctuation are removed, and whitespace is collapsed.
- **Application**: Used exclusively as a secondary signal for records where bank account or IFSC data is incomplete or unindexed.
- **Thresholding**:
  - Similarity $\ge 90\% \rightarrow$ tagged as `match_type = FUZZY_CANDIDATE` (candidate suggestion for analyst review).
  - Similarity $< 90\% \rightarrow$ tagged as `match_type = UNRESOLVED`.
- **Safety Principle**: A fuzzy name match **never** overrides or breaks a deterministic `account_number + IFSC` identity match.

---

## 🔑 Entity Master Mapping (`data/entity_master.csv`)

`entity_master.csv` provides a stable, deduplicated reference bridge between Stage 0 entity resolution and subsequent transaction graph construction in Stage 1:

- **Stable Entity IDs**: `entity_id` is an internal system identifier (e.g., `ENT_000001`, `ENT_000002`) assigned sequentially and reproducibly across runs.
- **Deterministic Identity Key**: `identity_key` (`normalized_account_number + "_" + normalized_ifsc`) serves as the unique invariant key.
- **Canonical Name**: For each entity, a canonical account-holder name is selected (preferring the most frequent original name associated with that entity) for descriptive metadata.
- **Ground Truth Isolation**: `ground_truth_entity_id` is used **only** for offline evaluation and never influences entity master generation or resolution logic.
- **Stage 1 Interface**: In Stage 1, `entity_master.csv` will be used to connect raw bank transaction streams (`transactions.csv`) directly to resolved entity nodes in the transaction graph.

---

## 📁 Project Structure

```text
sihmodel/
│
├── data/
│   ├── complaints.csv                   # Input cybercrime complaints (1,000 records)
│   ├── entity_ground_truth.csv          # Ground truth entity mappings (700 entities)
│   ├── resolved_entities.csv            # Output resolved entities mapping table
│   ├── entity_master.csv                # Stable entity master table (700 unique entities)
│   └── entity_resolution_summary.csv    # Output performance summary metrics
│
├── src/
│   └── entity_resolution.py             # Stage 0 Entity Resolution engine
│
├── generate_complaints_dataset.py       # Synthetic dataset generation script
└── README.md                            # Documentation and architecture guide
```

---

## 🚀 How to Run

To run the Entity Resolution pipeline:

```bash
python3 src/entity_resolution.py
```

### Generated Output Files
1. **`data/resolved_entities.csv`**: Contains all 1,000 complaints mapped to their `predicted_entity_id`, `match_type`, `name_similarity`, and evaluation ground truth.
2. **`data/entity_master.csv`**: Deduplicated master table (exactly 700 rows) linking each `entity_id` to its `account_number`, `ifsc`, `canonical_name`, and `identity_key`.
3. **`data/entity_resolution_summary.csv`**: Summary table recording key counts, execution metrics, and clustering performance scores.

---

## 📊 Evaluation & Metrics Explained

The resolution engine evaluates predicted cluster groupings against the offline `ground_truth_entity_id` column:

| Metric | Score | Simple Explanation |
| :--- | :---: | :--- |
| **Accuracy** | **100.00%** | Overall percentage of complaint pairs whose relationship (same entity vs. different entities) was correctly determined. |
| **Precision** | **100.00%** | When the algorithm grouped two complaints into the same entity, it was correct 100% of the time (0 false merges). |
| **Recall** | **100.00%** | Out of all complaints that truly belonged to the same underlying entity, the algorithm successfully grouped 100% of them (0 missed merges). |
| **F1 Score** | **100.00%** | Harmonic mean of Precision and Recall, reflecting clustering quality. |
| **False Merges (FP)** | **0** | Instances where two complaints referring to *different* bank accounts were mistakenly merged into the same entity. |
| **Missed Merges (FN)** | **0** | Instances where two complaints referring to the *same* bank account were mistakenly split into different entities. |

---

## ⚡ Scalability & Engineering Notes

1. **Deterministic Matching Scales Linearly ($O(N)$)**:
   - Exact `account_number + IFSC` matching is equivalent to a dictionary lookup / indexed hash join.
   - It executes in $O(N)$ time and memory, allowing it to scale straightforwardly to large complaint volumes.

2. **Fuzzy Matching Bottleneck ($O(N^2)$)**:
   - Naive all-pairs pairwise string comparison scales quadratically ($O(N^2)$), which quickly becomes a major compute bottleneck as complaint volumes grow.
   - For high-volume production pipelines, **Candidate Blocking / Canopy Clustering** (e.g., locality-sensitive hashing, phonetic blocking via Double Metaphone, or geographic/state bins) must be applied prior to fuzzy scoring.

3. **Prototype Scope**:
   - The hackathon prototype uses an exact-match primary layer paired with a lightweight toy fuzzy layer for incomplete records.
   - Handling adversarial identity obfuscation, deliberate typo generation, and multi-script transliteration (e.g., Indic scripts to Latin) are identified as future enhancements for later stages.

   
   
   ---
   ---
    The modification to entity_resolution.py and README.md has been implemented and verified.
  ──────
  ### 1. Modification Status

  Status: Succeeded

  • The stable entity master table entity_master.csv is automatically constructed and validated alongside resolved_entities.csv and entity_resolution_summary.csv.
  • All existing matching logic, evaluation metrics, and output schemas remain intact.
  • Ground truth (ground_truth_entity_id) is strictly isolated for evaluation and was never accessed during matching or master entity construction.
  ──────
  ### 2. Actual Stage 0 Metrics

    =============================================
          STAGE 0 — ENTITY RESOLUTION
    =============================================
    Input complaints           : 1000
    Ground-truth entities      : 700
    Predicted entities         : 700
    ---------------------------------------------
    Deterministic exact matches: 1000
    Fuzzy candidates           : 0
    Unresolved                 : 0
    ---------------------------------------------
    Entity Resolution Performance
    -----------------------------
    Accuracy  : 100.00%
    Precision : 100.00%
    Recall    : 100.00%
    F1 Score  : 100.00%
    Correctly Grouped: 1000 / 1000 complaints
    False merges: 0
    Missed merges: 0
    =============================================
  ──────
  ### 3 & 4. Entity Master Counts & Summary

  • Total rows in entity_master.csv: 700 (exactly one row per resolved entity node)
  • Unique entity_id values: 700
  • Unique identity_key values: 700
  ──────
  ### 5. 10 Sample Rows from data/entity_master.csv

   entity_id                               | account_number                          | ifsc                                    | canonical_name                          | identity_key
  -----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------
   ENT_000001                              | 101843567009                            | BARB0006789                             | Anita Ghosh                             | 101843567009_BARB0006789
   ENT_000002                              | 101948728483                            | UBIN0009900                             | Sneha Shinde                            | 101948728483_UBIN0009900
   ENT_000003                              | 103944899549                            | BARB0006789                             | Rajesh Patil                            | 103944899549_BARB0006789
   ENT_000004                              | 105360321413                            | ICIC0002468                             | Manish Ghosh                            | 105360321413_ICIC0002468
   ENT_000005                              | 106937867936                            | CNRB0006677                             | Shaurya Verma                           | 106937867936_CNRB0006677
   ENT_000006                              | 108901504899                            | AXIS0006543                             | Preeti Shah                             | 108901504899_AXIS0006543
   ENT_000007                              | 109238601984                            | ICIC0004321                             | Varun Deshmukh                          | 109238601984_ICIC0004321
   ENT_000008                              | 110895003819                            | CNRB0004455                             | Shri Pranav Ali                         | 110895003819_CNRB0004455
   ENT_000009                              | 111453993673                            | CNRB0006677                             | Sanjay Dutta                            | 111453993673_CNRB0006677
   ENT_000010                              | 112637644335                            | UBIN0005566                             | Advaith Srivastava                      | 112637644335_UBIN0005566
  ──────
  ### 6. Confirmation: Readiness for Stage 1

  Stage 0 is validated and ready for Stage 1:

  • All 1,000 complaints trace unambiguously to the 700 master entity nodes in entity_master.csv.
  • When raw banking transaction streams (transactions.csv) are introduced in Stage 1, transactions can be joined on account_number + ifsc (identity_key) directly to entity_id nodes to construct transaction
  graphs
  
  ---
  ---
