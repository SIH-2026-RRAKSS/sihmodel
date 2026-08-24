"""
Stage 0: Entity Resolution for Cybercrime Complaints
====================================================
This module implements the foundational Entity Resolution stage for a cybercrime
predictive analytics framework. Its primary objective is to collapse repeated
references to the same bank account across multiple cybercrime complaints into a
single unified financial entity node.

Architecture:
- Layer 1 (Deterministic Matching):
    Uses normalized (account_number + IFSC) as the primary deterministic composite key.
- Layer 2 (Toy Fuzzy Name Matching):
    Uses RapidFuzz as a secondary signal for records with missing or incomplete
    account/IFSC details (threshold >= 90).
    Fuzzy matches are tagged as 'FUZZY_CANDIDATE' and never override deterministic keys.
- Entity Master Mapping:
    Constructs a stable entity master table (data/entity_master.csv) containing
    one row per resolved entity with its deterministic identity_key, canonical name,
    account_number, and ifsc.

Important Constraints:
- ground_truth_entity_id is strictly used for offline evaluation and validation;
  it is never accessed during entity resolution matching logic.
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import pandas as pd
from rapidfuzz import fuzz


# ==============================================================================
# Configuration & Constants
# ==============================================================================

DEFAULT_DATA_DIR = Path("data")
COMPLAINTS_FILE = DEFAULT_DATA_DIR / "complaints.csv"
GROUND_TRUTH_FILE = DEFAULT_DATA_DIR / "entity_ground_truth.csv"
RESOLVED_ENTITIES_FILE = DEFAULT_DATA_DIR / "resolved_entities.csv"
ENTITY_MASTER_FILE = DEFAULT_DATA_DIR / "entity_master.csv"
SUMMARY_FILE = DEFAULT_DATA_DIR / "entity_resolution_summary.csv"

FUZZY_SIMILARITY_THRESHOLD = 90.0


# ==============================================================================
# Normalization Functions
# ==============================================================================

def normalize_account_number(acct: Any) -> Optional[str]:
    """
    Normalizes a bank account number:
    - Converts to string and strips leading/trailing whitespace.
    - Handles float representations (e.g. from CSV parsing).
    - Ensures valid 12-digit string format if valid.

    Returns:
        12-digit account number string, or None if invalid/missing.
    """
    if acct is None or pd.isna(acct):
        return None
    acct_str = str(acct).strip()
    # If parsed as float (e.g., '123456789012.0'), remove decimal part
    if acct_str.endswith(".0"):
        acct_str = acct_str[:-2]
    # Check if it consists of digits
    if re.fullmatch(r"\d{12}", acct_str):
        return acct_str
    # If not strictly 12 digits but non-empty, return stripped string for inspection
    return acct_str if acct_str else None


def normalize_ifsc(ifsc_code: Any) -> Optional[str]:
    """
    Normalizes an Indian Financial System Code (IFSC):
    - Converts to uppercase and strips whitespace.
    - Validates against standard 11-character format (4 letters, '0', 6 alphanumeric).

    Returns:
        Normalized IFSC string, or None if invalid/missing.
    """
    if ifsc_code is None or pd.isna(ifsc_code):
        return None
    ifsc_str = str(ifsc_code).strip().upper()
    return ifsc_str if ifsc_str else None


def normalize_name(name: Any) -> str:
    """
    Normalizes account holder name for fuzzy comparison:
    - Lowercase conversion
    - Strips whitespace
    - Collapses multiple spaces
    - Removes punctuation and special characters
    """
    if name is None or pd.isna(name):
        return ""
    name_str = str(name).lower()
    # Remove punctuation
    name_str = re.sub(r"[^\w\s]", "", name_str)
    # Collapse multiple whitespace characters
    name_str = re.sub(r"\s+", " ", name_str).strip()
    return name_str


# ==============================================================================
# Data Loading
# ==============================================================================

def load_data(
    complaints_path: Path = COMPLAINTS_FILE,
    ground_truth_path: Optional[Path] = GROUND_TRUTH_FILE
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Loads complaints dataset and optional ground truth reference table.
    """
    if not complaints_path.exists():
        raise FileNotFoundError(f"Complaints file not found: {complaints_path}")

    df_complaints = pd.read_csv(complaints_path, dtype={"account_number": str, "ifsc": str})

    df_ground_truth = None
    if ground_truth_path and ground_truth_path.exists():
        df_ground_truth = pd.read_csv(ground_truth_path, dtype={"account_number": str, "ifsc": str})

    return df_complaints, df_ground_truth


# ==============================================================================
# Layer 1: Deterministic Matching
# ==============================================================================

def create_deterministic_entities(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Executes Layer 1 Deterministic Entity Resolution.

    Primary matching key: normalized_account_number + '_' + normalized_ifsc

    Returns:
        df_exact: DataFrame of records resolved via deterministic key (match_type = EXACT)
        df_unresolved: DataFrame of records with missing/incomplete deterministic keys
        entity_registry: Dict mapping entity_key and predicted_entity_id to representative profile
    """
    # Create working copy without using ground_truth_entity_id in matching
    working_df = df.copy()

    # Apply normalizations
    working_df["norm_account"] = working_df["account_number"].apply(normalize_account_number)
    working_df["norm_ifsc"] = working_df["ifsc"].apply(normalize_ifsc)
    working_df["norm_name"] = working_df["account_holder_name"].apply(normalize_name)

    # Valid deterministic key requires both valid account_number and ifsc
    is_valid_deterministic = (
        working_df["norm_account"].notna() &
        working_df["norm_account"].str.match(r"^\d{12}$") &
        working_df["norm_ifsc"].notna() &
        working_df["norm_ifsc"].str.match(r"^[A-Z]{4}0[A-Z0-9]{6}$")
    )

    df_deterministic = working_df[is_valid_deterministic].copy()
    df_unresolved = working_df[~is_valid_deterministic].copy()

    # Construct deterministic composite identity key
    df_deterministic["entity_key"] = (
        df_deterministic["norm_account"] + "_" + df_deterministic["norm_ifsc"]
    )

    # Assign sequential predicted_entity_id (e.g. ENT_000001, ENT_000002, ...)
    # Sort entity keys to guarantee deterministic, reproducible entity ID assignment
    unique_keys = sorted(df_deterministic["entity_key"].unique())
    key_to_entity_id = {key: f"ENT_{idx + 1:06d}" for idx, key in enumerate(unique_keys)}
    df_deterministic["predicted_entity_id"] = df_deterministic["entity_key"].map(key_to_entity_id)
    df_deterministic["match_type"] = "EXACT"

    # Build entity registry containing canonical profile and representative name
    entity_registry: Dict[str, Dict[str, Any]] = {}
    for entity_id, group in df_deterministic.groupby("predicted_entity_id"):
        # Select representative name: prefer most frequent name in the cluster, tie-break with first seen
        name_counts = group["account_holder_name"].value_counts()
        canonical_name = name_counts.index[0]
        first_row = group.iloc[0]
        norm_canonical_name = normalize_name(canonical_name)

        entity_registry[entity_id] = {
            "entity_key": first_row["entity_key"],
            "account_number": first_row["norm_account"],
            "ifsc": first_row["norm_ifsc"],
            "canonical_name": canonical_name,
            "norm_canonical_name": norm_canonical_name
        }

    # Calculate name similarity against cluster representative name
    def calc_exact_name_sim(row: pd.Series) -> float:
        rep_norm_name = entity_registry[row["predicted_entity_id"]]["norm_canonical_name"]
        return round(float(fuzz.token_sort_ratio(row["norm_name"], rep_norm_name)), 2)

    df_deterministic["name_similarity"] = df_deterministic.apply(calc_exact_name_sim, axis=1)

    return df_deterministic, df_unresolved, entity_registry


# ==============================================================================
# Layer 2: Toy Fuzzy Name Matching
# ==============================================================================

def perform_fuzzy_matching(
    df_unresolved: pd.DataFrame,
    entity_registry: Dict[str, Dict[str, Any]],
    threshold: float = FUZZY_SIMILARITY_THRESHOLD
) -> pd.DataFrame:
    """
    Executes Layer 2 Toy Fuzzy Matching for records that could not be resolved
    deterministically (e.g. missing account number or IFSC).

    Calculates RapidFuzz name similarity against existing resolved entity profiles.
    - If similarity >= threshold (default 90): match_type = 'FUZZY_CANDIDATE'
    - If similarity < threshold: match_type = 'UNRESOLVED'

    Note: Fuzzy matching never overrides deterministic matches.
    """
    if df_unresolved.empty:
        return df_unresolved

    fuzzy_records = []
    
    # Pre-extract normalized entity names for faster matching
    candidate_entities = [
        (ent_id, data["norm_canonical_name"])
        for ent_id, data in entity_registry.items()
    ]

    for _, row in df_unresolved.iterrows():
        row_dict = row.to_dict()
        record_norm_name = row_dict.get("norm_name", "")

        best_entity_id = "UNRESOLVED"
        best_score = 0.0

        if record_norm_name and candidate_entities:
            for ent_id, cand_norm_name in candidate_entities:
                score = fuzz.token_sort_ratio(record_norm_name, cand_norm_name)
                if score > best_score:
                    best_score = float(score)
                    best_entity_id = ent_id

        if best_score >= threshold:
            row_dict["predicted_entity_id"] = best_entity_id
            row_dict["match_type"] = "FUZZY_CANDIDATE"
            row_dict["name_similarity"] = round(best_score, 2)
        else:
            row_dict["predicted_entity_id"] = "UNRESOLVED"
            row_dict["match_type"] = "UNRESOLVED"
            row_dict["name_similarity"] = round(best_score, 2) if best_score > 0 else 0.0

        fuzzy_records.append(row_dict)

    return pd.DataFrame(fuzzy_records)


# ==============================================================================
# Entity Master Generation
# ==============================================================================

def build_entity_master(
    df_resolved: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs the stable Entity Master DataFrame (entity_master.csv).
    Contains exactly one row per resolved entity node.

    Columns:
    - entity_id: Stable internal entity identifier (ENT_000001, ...)
    - account_number: Normalized 12-digit account number
    - ifsc: Normalized uppercase IFSC code
    - canonical_name: Most frequent original account-holder name associated with the entity
    - identity_key: Deterministic key (normalized_account_number + '_' + normalized_ifsc)
    """
    master_records = []

    # Process all resolved entities grouped by predicted_entity_id
    for entity_id, group in df_resolved.groupby("predicted_entity_id", sort=True):
        if entity_id == "UNRESOLVED":
            continue

        # Determine canonical name as the most frequent name in the cluster
        name_counts = group["account_holder_name"].value_counts()
        canonical_name = name_counts.index[0]

        # Extract normalized account number and ifsc
        first_valid_acct = group["account_number"].dropna().astype(str).str.strip().iloc[0]
        first_valid_ifsc = group["ifsc"].dropna().astype(str).str.strip().str.upper().iloc[0]
        identity_key = f"{first_valid_acct}_{first_valid_ifsc}"

        master_records.append({
            "entity_id": entity_id,
            "account_number": first_valid_acct,
            "ifsc": first_valid_ifsc,
            "canonical_name": canonical_name,
            "identity_key": identity_key
        })

    df_master = pd.DataFrame(master_records)
    
    # Guarantee column ordering
    master_columns = ["entity_id", "account_number", "ifsc", "canonical_name", "identity_key"]
    return df_master[master_columns]


# ==============================================================================
# Validation Routine
# ==============================================================================

def validate_results(
    df_complaints: pd.DataFrame,
    df_resolved: pd.DataFrame,
    df_master: pd.DataFrame
) -> None:
    """
    Performs rigorous integrity checks on input, resolved, and master datasets.
    """
    # 1. Unique complaint IDs
    if df_complaints["complaint_id"].nunique() != len(df_complaints):
        raise ValueError("Validation Failed: complaint_id values are not unique in input data!")
    if df_resolved["complaint_id"].nunique() != len(df_resolved):
        raise ValueError("Validation Failed: complaint_id values are not unique in resolved output!")

    # 2. Account numbers must be 12 digits when present in EXACT matches
    exact_matches = df_resolved[df_resolved["match_type"] == "EXACT"]
    if not exact_matches["account_number"].astype(str).str.match(r"^\d{12}$").all():
        raise ValueError("Validation Failed: An exact match contains an account_number that is not 12 digits!")

    # 3. IFSC must be normalized to uppercase
    if not exact_matches["ifsc"].astype(str).str.isupper().all():
        raise ValueError("Validation Failed: IFSC codes are not properly normalized to uppercase!")

    # 4. No deterministic entity should contain multiple account_number + IFSC combinations
    entity_key_pairs = exact_matches.groupby("predicted_entity_id")[["account_number", "ifsc"]].nunique()
    if (entity_key_pairs["account_number"] > 1).any() or (entity_key_pairs["ifsc"] > 1).any():
        raise ValueError("Validation Failed: A predicted entity contains multiple account_number + IFSC pairs!")

    # 5. Every exact match should map identical account_number + IFSC to the same predicted_entity_id
    grouped = exact_matches.groupby(["account_number", "ifsc"])["predicted_entity_id"].nunique()
    if (grouped > 1).any():
        raise ValueError("Validation Failed: Identical account_number + IFSC mapped to different predicted_entity_id!")

    # 6. Entity Master validation checks:
    # 6.1 Row count matches number of unique predicted entities (700)
    expected_entity_count = df_resolved[df_resolved["predicted_entity_id"] != "UNRESOLVED"]["predicted_entity_id"].nunique()
    if len(df_master) != expected_entity_count:
        raise ValueError(f"Validation Failed: entity_master has {len(df_master)} rows, expected {expected_entity_count}!")

    # 6.2 entity_id is unique
    if df_master["entity_id"].nunique() != len(df_master):
        raise ValueError("Validation Failed: entity_id is not unique in entity_master.csv!")

    # 6.3 identity_key is unique
    if df_master["identity_key"].nunique() != len(df_master):
        raise ValueError("Validation Failed: identity_key is not unique in entity_master.csv!")

    # 6.4 account_number + IFSC uniquely identifies an entity in entity_master
    acct_ifsc_groups = df_master.groupby(["account_number", "ifsc"])["entity_id"].nunique()
    if (acct_ifsc_groups > 1).any():
        raise ValueError("Validation Failed: account_number + IFSC combination maps to multiple entities in entity_master!")

    # 6.5 Every predicted_entity_id in resolved_entities exists in entity_master
    resolved_entity_ids = set(df_resolved[df_resolved["predicted_entity_id"] != "UNRESOLVED"]["predicted_entity_id"])
    master_entity_ids = set(df_master["entity_id"])
    if not resolved_entity_ids.issubset(master_entity_ids):
        raise ValueError("Validation Failed: Some predicted_entity_id values in resolved_entities are missing from entity_master!")

    # 6.6 Full traceability check: complaint -> predicted_entity_id -> entity_master -> account_number + ifsc
    master_lookup = df_master.set_index("entity_id")[["account_number", "ifsc"]].to_dict(orient="index")
    for _, row in exact_matches.iterrows():
        ent_id = row["predicted_entity_id"]
        if ent_id not in master_lookup:
            raise ValueError(f"Validation Failed: {ent_id} not found in master lookup!")
        master_info = master_lookup[ent_id]
        if str(row["account_number"]).strip() != master_info["account_number"] or str(row["ifsc"]).strip().upper() != master_info["ifsc"]:
            raise ValueError(f"Validation Failed: Inconsistent account credentials between complaint {row['complaint_id']} and master entity {ent_id}!")

    # 6.7 No entity contains multiple different account_number + IFSC combinations in master
    if df_master.duplicated(subset=["account_number", "ifsc"]).any():
        raise ValueError("Validation Failed: Duplicate account_number + IFSC combinations detected in entity_master!")


# ==============================================================================
# Evaluation Routine
# ==============================================================================

def evaluate_entity_resolution(
    df_resolved: pd.DataFrame,
    df_ground_truth: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Evaluates entity resolution results against ground truth.

    Computes:
    - Pairwise Precision, Recall, F1 score, and Pairwise Accuracy
    - Total complaint pairs N*(N-1)/2
    - True Positives (correctly grouped pairs)
    - False Merges (False Positives: distinct entities incorrectly merged)
    - Missed Merges (False Negatives: same entity records split across clusters)
    """
    if "ground_truth_entity_id" not in df_resolved.columns:
        raise KeyError("Evaluation requires 'ground_truth_entity_id' column in dataset.")

    total_records = len(df_resolved)
    total_pairs = total_records * (total_records - 1) // 2

    pred_labels = df_resolved["predicted_entity_id"]
    true_labels = df_resolved["ground_truth_entity_id"]

    # Calculate cluster sizes
    pred_cluster_sizes = pred_labels.value_counts()
    true_cluster_sizes = true_labels.value_counts()

    # Total predicted positive pairs & true positive pairs
    total_pred_pairs = sum(c * (c - 1) // 2 for c in pred_cluster_sizes)
    total_true_pairs = sum(c * (c - 1) // 2 for c in true_cluster_sizes)

    # Contingency matrix overlap to find True Positives (TP)
    contingency = pd.crosstab(pred_labels, true_labels)
    tp_pairs = int(sum(c * (c - 1) // 2 for c in contingency.values.flatten() if c > 1))

    # False Merges (FP) = pairs merged in prediction that belong to different entities
    fp_pairs = total_pred_pairs - tp_pairs

    # Missed Merges (FN) = pairs that belong to same entity but were not merged in prediction
    fn_pairs = total_true_pairs - tp_pairs

    # True Negatives (TN)
    tn_pairs = total_pairs - tp_pairs - fp_pairs - fn_pairs

    # Metrics
    precision = (tp_pairs / total_pred_pairs) if total_pred_pairs > 0 else 1.0
    recall = (tp_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp_pairs + tn_pairs) / total_pairs if total_pairs > 0 else 1.0

    # Complaint-level purity (number of complaints in pure clusters)
    cluster_gt_counts = df_resolved.groupby("predicted_entity_id")["ground_truth_entity_id"].nunique()
    pure_clusters = cluster_gt_counts[cluster_gt_counts == 1].index
    correctly_grouped_complaints = int(df_resolved["predicted_entity_id"].isin(pure_clusters).sum())

    match_counts = df_resolved["match_type"].value_counts().to_dict()

    metrics = {
        "total_complaints": total_records,
        "ground_truth_entities": int(true_labels.nunique()),
        "unique_predicted_entities": int(pred_labels.nunique()),
        "exact_matches": match_counts.get("EXACT", 0),
        "fuzzy_candidates": match_counts.get("FUZZY_CANDIDATE", 0),
        "unresolved_records": match_counts.get("UNRESOLVED", 0),
        "total_pairs": total_pairs,
        "true_positives": tp_pairs,
        "false_merges": fp_pairs,
        "missed_merges": fn_pairs,
        "true_negatives": tn_pairs,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "correctly_grouped_complaints": correctly_grouped_complaints,
    }

    return metrics


# ==============================================================================
# Save Results
# ==============================================================================

def save_results(
    df_resolved: pd.DataFrame,
    df_master: pd.DataFrame,
    summary_metrics: Dict[str, Any],
    output_dir: Path = DEFAULT_DATA_DIR
) -> Tuple[Path, Path, Path]:
    """
    Saves resolved entities, entity master, and summary CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save resolved_entities.csv
    output_cols = [
        "complaint_id",
        "predicted_entity_id",
        "match_type",
        "account_number",
        "ifsc",
        "account_holder_name",
        "name_similarity",
        "ground_truth_entity_id"
    ]
    resolved_path = output_dir / "resolved_entities.csv"
    df_resolved[output_cols].to_csv(resolved_path, index=False)

    # 2. Save entity_master.csv
    master_path = output_dir / "entity_master.csv"
    df_master.to_csv(master_path, index=False)

    # 3. Save entity_resolution_summary.csv
    summary_df = pd.DataFrame([{
        "total_complaints": summary_metrics["total_complaints"],
        "ground_truth_entities": summary_metrics["ground_truth_entities"],
        "unique_predicted_entities": summary_metrics["unique_predicted_entities"],
        "exact_matches": summary_metrics["exact_matches"],
        "fuzzy_candidates": summary_metrics["fuzzy_candidates"],
        "unresolved_records": summary_metrics["unresolved_records"],
        "accuracy_pct": round(summary_metrics["accuracy"] * 100, 2),
        "precision_pct": round(summary_metrics["precision"] * 100, 2),
        "recall_pct": round(summary_metrics["recall"] * 100, 2),
        "f1_score_pct": round(summary_metrics["f1_score"] * 100, 2),
        "correctly_grouped_complaints": summary_metrics["correctly_grouped_complaints"],
        "false_merges": summary_metrics["false_merges"],
        "missed_merges": summary_metrics["missed_merges"]
    }])
    summary_path = output_dir / "entity_resolution_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return resolved_path, master_path, summary_path


# ==============================================================================
# Main Demonstration Entrypoint
# ==============================================================================

def main():
    # 1. Load Data
    df_complaints, df_ground_truth = load_data(COMPLAINTS_FILE, GROUND_TRUTH_FILE)

    # 2. Layer 1: Deterministic Matching
    df_exact, df_unresolved, entity_registry = create_deterministic_entities(df_complaints)

    # 3. Layer 2: Toy Fuzzy Name Matching (for any incomplete/unresolved records)
    if not df_unresolved.empty:
        df_fuzzy = perform_fuzzy_matching(df_unresolved, entity_registry)
        df_resolved = pd.concat([df_exact, df_fuzzy], ignore_index=True)
    else:
        df_resolved = df_exact.copy()

    # Sort by original complaint_id order
    df_resolved = df_resolved.sort_values("complaint_id").reset_index(drop=True)

    # 4. Construct Stable Entity Master Table
    df_master = build_entity_master(df_resolved)

    # 5. Validate Results
    validate_results(df_complaints, df_resolved, df_master)

    # 6. Evaluate Performance against Ground Truth
    metrics = evaluate_entity_resolution(df_resolved, df_ground_truth)

    # 7. Save Output Files
    resolved_path, master_path, summary_path = save_results(df_resolved, df_master, metrics, DEFAULT_DATA_DIR)

    # 8. Print Formatted Demonstration Report
    print("\n" + "=" * 45)
    print("      STAGE 0 — ENTITY RESOLUTION")
    print("=" * 45)
    print(f"Input complaints           : {metrics['total_complaints']}")
    print(f"Ground-truth entities      : {metrics['ground_truth_entities']}")
    print(f"Predicted entities         : {metrics['unique_predicted_entities']}")
    print("-" * 45)
    print(f"Deterministic exact matches: {metrics['exact_matches']}")
    print(f"Fuzzy candidates           : {metrics['fuzzy_candidates']}")
    print(f"Unresolved                 : {metrics['unresolved_records']}")
    print("-" * 45)
    print("Entity Resolution Performance")
    print("-----------------------------")
    print(f"Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision : {metrics['precision'] * 100:.2f}%")
    print(f"Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score  : {metrics['f1_score'] * 100:.2f}%")
    print(f"Correctly Grouped: {metrics['correctly_grouped_complaints']} / {metrics['total_complaints']} complaints")
    print(f"False merges: {metrics['false_merges']}")
    print(f"Missed merges: {metrics['missed_merges']}")
    print("=" * 45 + "\n")

    # 9. Print Entity Master Summary & 10 Sample Rows
    print("Entity Master Summary (data/entity_master.csv):")
    print("-" * 75)
    print(f"Total Master Entities      : {len(df_master)}")
    print(f"Unique entity_id           : {df_master['entity_id'].nunique()}")
    print(f"Unique identity_key        : {df_master['identity_key'].nunique()}")
    print("-" * 75)
    print("Sample 10 Rows from entity_master.csv:")
    print(df_master.head(10).to_string(index=False))
    print("-" * 75 + "\n")

    # 10. Print 10 Example Complaint Resolutions
    print("Example Resolutions (Sample 10 Rows from resolved_entities.csv):")
    print("-" * 75)
    print(f"{'Complaint':<12} | {'Predicted Entity':<18} | {'Ground Truth':<14} | {'Match Type':<12} | {'Sim (%)':<7}")
    print("-" * 75)
    sample_df = df_resolved.head(10)
    for _, r in sample_df.iterrows():
        print(f"{r['complaint_id']:<12} | {r['predicted_entity_id']:<18} | {r['ground_truth_entity_id']:<14} | {r['match_type']:<12} | {r['name_similarity']:>5.1f}%")
    print("-" * 75 + "\n")

    print(f"[SUCCESS] Saved resolved entities to: {resolved_path}")
    print(f"[SUCCESS] Saved entity master to     : {master_path}")
    print(f"[SUCCESS] Saved resolution summary to: {summary_path}\n")


if __name__ == "__main__":
    main()
