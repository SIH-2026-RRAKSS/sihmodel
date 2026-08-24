"""
Synthetic Cybercrime Complaints Dataset Generator
=================================================
This script generates a synthetic dataset of cybercrime complaints designed for
developing and testing Stage 0 Entity Resolution (identifying when multiple
cybercrime complaints refer to the same underlying bank account entity).

Goal:
- Generate exactly 1,000 synthetic cybercrime complaint records.
- Produce two CSV files:
  1. data/complaints.csv - The main dataset for entity resolution modeling.
  2. data/entity_ground_truth.csv - The ground truth mapping table.

Data Distribution:
- 700 complaints from unique base accounts (initial appearance)
- 200 complaints that are duplicate/repeated references to an existing account (exact canonical name)
- 100 complaints that refer to an existing account with slightly modified account-holder names
Total: 1,000 complaints across 700 unique synthetic entities.

Privacy & Synthetic Data Guarantee:
- All generated names, account numbers, and IFSC codes are 100% synthetic/fictional.
- No real personal data (PII) or real-world bank account numbers are used.
"""

import os
import random
import re
from datetime import date, timedelta
from typing import List, Dict, Tuple, Any
import pandas as pd


# ==============================================================================
# Configuration & Constants
# ==============================================================================

RANDOM_SEED = 42
TOTAL_COMPLAINTS = 1000
NUM_BASE_ACCOUNTS = 700
NUM_EXACT_DUPLICATES = 200
NUM_NAME_VARIATIONS = 100

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 8, 24)

OUTPUT_DIR = "data"
COMPLAINTS_FILE = os.path.join(OUTPUT_DIR, "complaints.csv")
GROUND_TRUTH_FILE = os.path.join(OUTPUT_DIR, "entity_ground_truth.csv")


# Synthetic Bank IFSC codes (Standard format: 4 uppercase alphabetic bank code + '0' + 6 alphanumeric branch code)
SYNTHETIC_IFSC_CODES = [
    "SBIN0001234", "SBIN0005678", "SBIN0009101",
    "HDFC0001234", "HDFC0005678", "HDFC0004321",
    "ICIC0004321", "ICIC0008765", "ICIC0002468",
    "PUNB0009876", "PUNB0005432", "PUNB0001357",
    "AXIS0003456", "AXIS0007890", "AXIS0006543",
    "KKBK0007890", "KKBK0003210", "KKBK0001928",
    "BARB0002345", "BARB0006789", "BARB0003579",
    "UBIN0005566", "UBIN0007788", "UBIN0009900",
    "IDIB0001122", "IDIB0003344", "IDIB0005566",
    "CNRB0004455", "CNRB0006677", "CNRB0008899"
]

# Synthetic Indian First and Last Names for realistic combinations
FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Aayan",
    "Krishna", "Ishan", "Shaurya", "Atharv", "Advik", "Pranav", "Advaith", "Aryan",
    "Dhruv", "Kabir", "Ritvik", "Ananya", "Diya", "Saanvi", "Aadhya", "Pari",
    "Anika", "Navya", "Angel", "Myra", "Sara", "Ira", "Riya", "Priya", "Pooja",
    "Sneha", "Neha", "Rohit", "Rahul", "Amit", "Vikram", "Suresh", "Ramesh",
    "Rajesh", "Sunita", "Geeta", "Meena", "Anita", "Kavita", "Deepa", "Swati",
    "Manish", "Gaurav", "Siddharth", "Kunal", "Alok", "Nikhil", "Pankaj", "Deepak",
    "Anil", "Sunil", "Sanjay", "Ajay", "Vijay", "Ashok", "Karan", "Varun",
    "Tanvi", "Shreya", "Kavya", "Ishita", "Meera", "Roshni", "Sweta", "Preeti"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Shah", "Joshi",
    "Rao", "Reddy", "Nair", "Menon", "Banerjee", "Chatterjee", "Mukherjee", "Ghosh",
    "Das", "Sen", "Dutta", "Roy", "Mehta", "Jain", "Agarwal", "Bhatia",
    "Kapoor", "Malhotra", "Khan", "Ali", "Ahmed", "Deshmukh", "Kulkarni", "Patil",
    "Shinde", "Pawar", "Gaikwad", "Yadav", "Mishra", "Pandey", "Tiwari", "Choudhury",
    "Dubey", "Tripathi", "Saxena", "Srivastava", "Bose", "Dey", "Paul", "Biswas"
]

# Cybercrime complaint types
COMPLAINT_TYPES = [
    "UPI Fraud",
    "Online Banking Fraud",
    "Phishing",
    "Investment Fraud",
    "Payment Fraud",
    "Identity Theft",
    "Shopping Fraud"
]

# Indian States and plausible corresponding districts/cities
STATE_DISTRICT_MAP = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik", "Aurangabad", "Solapur"],
    "Karnataka": ["Bengaluru Urban", "Mysuru", "Hubballi-Dharwad", "Mangaluru", "Belagavi", "Kalaburagi"],
    "West Bengal": ["Kolkata", "Howrah", "North 24 Parganas", "South 24 Parganas", "Darjeeling", "Siliguri", "Hooghly"],
    "Delhi": ["New Delhi", "North Delhi", "South Delhi", "East Delhi", "West Delhi", "Central Delhi"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli"],
    "Telangana": ["Hyderabad", "Warangal", "Rangareddy", "Nizamabad", "Karimnagar", "Khammam"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Noida", "Prayagraj", "Agra", "Ghaziabad", "Meerut"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner", "Ajmer"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Tirupati"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat", "Ambala", "Karnal"]
}


# ==============================================================================
# Helper Functions
# ==============================================================================

def generate_12_digit_account_number(rng: random.Random) -> str:
    """Generate a strictly 12-digit synthetic bank account number string."""
    return f"{rng.randint(100000000000, 999999999999)}"


def generate_random_date(start: date, end: date, rng: random.Random) -> str:
    """Generate a random ISO date between start and end inclusive."""
    delta = (end - start).days
    random_days = rng.randint(0, delta)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


def generate_reported_amount(rng: random.Random) -> float:
    """
    Generate a realistic synthetic reported fraud amount between 500 and 500,000 INR.
    Mixes rounded/common amounts with exact 2-decimal numbers.
    """
    if rng.random() < 0.40:
        # Common round figures in fraud complaints
        round_amounts = [
            500, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 15000, 20000,
            25000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 250000, 500000
        ]
        return float(rng.choice(round_amounts))
    else:
        # Continuous distribution with 2 decimal places
        amount = rng.uniform(500.0, 500000.0)
        return round(amount, 2)


def generate_name_variation(canonical_name: str, rng: random.Random) -> str:
    """
    Apply realistic typographical, formatting, or stylistic variations to a canonical name.
    Examples:
    - Rahul Sharma -> Rahul S Sharma / Rahul S. Sharma
    - Rahul Sharma -> R Sharma / R. Sharma
    - Rahul Sharma -> Rahul Sharma  (trailing whitespace)
    - Rahul Sharma ->  Rahul Sharma (leading whitespace)
    - Rahul Sharma -> rahul sharma (lowercase)
    - Rahul Sharma -> RAHUL SHARMA (uppercase)
    - Rahul Sharma -> Rahul  Sharma (double space)
    - Rahul Sharma -> Sharma, Rahul (surname first)
    - Rahul Sharma -> Mr. Rahul Sharma / Dr. Rahul Sharma (honorific prefix)
    """
    parts = canonical_name.strip().split()
    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ""

    middle_initials = ["A", "B", "C", "D", "K", "M", "P", "R", "S", "V"]
    variation_type = rng.randint(1, 8)

    if variation_type == 1 and last:
        # Middle initial
        mid = rng.choice(middle_initials)
        return f"{first} {mid} {last}" if rng.random() < 0.5 else f"{first} {mid}. {last}"

    elif variation_type == 2 and last:
        # First name initial only
        return f"{first[0]} {last}" if rng.random() < 0.5 else f"{first[0]}. {last}"

    elif variation_type == 3:
        # Leading / Trailing whitespace
        return f"{canonical_name} " if rng.random() < 0.5 else f" {canonical_name}"

    elif variation_type == 4:
        # Lowercase
        return canonical_name.lower()

    elif variation_type == 5:
        # Uppercase
        return canonical_name.upper()

    elif variation_type == 6 and last:
        # Double space
        return f"{first}  {last}"

    elif variation_type == 7 and last:
        # Surname first with/without comma
        return f"{last}, {first}" if rng.random() < 0.5 else f"{last} {first}"

    elif variation_type == 8:
        # Honorific / Title prefix
        titles = ["Mr.", "Ms.", "Dr.", "Shri", "Smt."]
        return f"{rng.choice(titles)} {canonical_name}"

    # Fallback to lowercase
    return canonical_name.lower()


# ==============================================================================
# Core Dataset Generation Logic
# ==============================================================================

def generate_synthetic_dataset(
    seed: int = RANDOM_SEED,
    num_base: int = NUM_BASE_ACCOUNTS,
    num_exact_dup: int = NUM_EXACT_DUPLICATES,
    num_name_var: int = NUM_NAME_VARIATIONS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generates synthetic complaints and ground truth datasets with strict entity mapping.

    Returns:
        df_complaints: DataFrame of complaints (1,000 rows)
        df_ground_truth: DataFrame of unique entity ground truth (num_base rows)
        stats: Dictionary containing distribution and generation metrics
    """
    rng = random.Random(seed)
    states = list(STATE_DISTRICT_MAP.keys())

    # Step 1: Create unique synthetic entities (Base Accounts)
    entities: List[Dict[str, Any]] = []
    used_account_ifsc_pairs = set()

    for i in range(num_base):
        entity_id = f"ENT_{i + 1:06d}"
        
        # Ensure unique (account_number, ifsc) pair for each entity
        while True:
            acct_no = generate_12_digit_account_number(rng)
            ifsc = rng.choice(SYNTHETIC_IFSC_CODES)
            if (acct_no, ifsc) not in used_account_ifsc_pairs:
                used_account_ifsc_pairs.add((acct_no, ifsc))
                break

        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        canonical_name = f"{first} {last}"

        entities.append({
            "ground_truth_entity_id": entity_id,
            "account_number": acct_no,
            "ifsc": ifsc,
            "canonical_name": canonical_name
        })

    # Step 2: Generate the 1,000 complaints dataset
    # Category 1: 700 Base complaints (1 initial complaint for each entity)
    complaints: List[Dict[str, Any]] = []

    for entity in entities:
        state = rng.choice(states)
        district = rng.choice(STATE_DISTRICT_MAP[state])
        complaints.append({
            "account_number": entity["account_number"],
            "ifsc": entity["ifsc"],
            "account_holder_name": entity["canonical_name"],
            "complaint_date": generate_random_date(START_DATE, END_DATE, rng),
            "complaint_type": rng.choice(COMPLAINT_TYPES),
            "reported_amount": generate_reported_amount(rng),
            "state": state,
            "district": district,
            "ground_truth_entity_id": entity["ground_truth_entity_id"],
            "_record_category": "base_unique"
        })

    # Category 2: 200 Exact-repeat complaints referencing existing accounts
    # Pick a random entity, retain exact account_number, ifsc, and canonical name
    for _ in range(num_exact_dup):
        target_entity = rng.choice(entities)
        state = rng.choice(states)
        district = rng.choice(STATE_DISTRICT_MAP[state])
        complaints.append({
            "account_number": target_entity["account_number"],
            "ifsc": target_entity["ifsc"],
            "account_holder_name": target_entity["canonical_name"],
            "complaint_date": generate_random_date(START_DATE, END_DATE, rng),
            "complaint_type": rng.choice(COMPLAINT_TYPES),
            "reported_amount": generate_reported_amount(rng),
            "state": state,
            "district": district,
            "ground_truth_entity_id": target_entity["ground_truth_entity_id"],
            "_record_category": "exact_duplicate"
        })

    # Category 3: 100 Name-variation complaints referencing existing accounts
    # Pick a random entity, retain exact account_number and ifsc, but apply name variation
    for _ in range(num_name_var):
        target_entity = rng.choice(entities)
        state = rng.choice(states)
        district = rng.choice(STATE_DISTRICT_MAP[state])
        modified_name = generate_name_variation(target_entity["canonical_name"], rng)
        complaints.append({
            "account_number": target_entity["account_number"],
            "ifsc": target_entity["ifsc"],
            "account_holder_name": modified_name,
            "complaint_date": generate_random_date(START_DATE, END_DATE, rng),
            "complaint_type": rng.choice(COMPLAINT_TYPES),
            "reported_amount": generate_reported_amount(rng),
            "state": state,
            "district": district,
            "ground_truth_entity_id": target_entity["ground_truth_entity_id"],
            "_record_category": "name_variation"
        })

    # Step 3: Shuffle complaints and assign unique complaint_id
    rng.shuffle(complaints)

    for idx, c in enumerate(complaints):
        c["complaint_id"] = f"C{idx + 1:06d}"

    # Step 4: Build DataFrames
    complaint_columns = [
        "complaint_id",
        "account_number",
        "ifsc",
        "account_holder_name",
        "complaint_date",
        "complaint_type",
        "reported_amount",
        "state",
        "district",
        "ground_truth_entity_id"
    ]

    df_complaints = pd.DataFrame(complaints)
    record_categories = df_complaints["_record_category"].value_counts().to_dict()
    df_complaints = df_complaints[complaint_columns]

    df_ground_truth = pd.DataFrame(entities)[[
        "ground_truth_entity_id",
        "account_number",
        "ifsc",
        "canonical_name"
    ]]

    stats = {
        "total_complaints": len(df_complaints),
        "unique_accounts": df_complaints["account_number"].nunique(),
        "unique_entities": df_complaints["ground_truth_entity_id"].nunique(),
        "num_base_unique": record_categories.get("base_unique", 0),
        "num_exact_duplicates": record_categories.get("exact_duplicate", 0),
        "num_name_variations": record_categories.get("name_variation", 0),
        "date_min": df_complaints["complaint_date"].min(),
        "date_max": df_complaints["complaint_date"].max(),
        "type_counts": df_complaints["complaint_type"].value_counts().to_dict(),
    }

    return df_complaints, df_ground_truth, stats


# ==============================================================================
# Validation Routine
# ==============================================================================

def validate_dataset(df_complaints: pd.DataFrame, df_ground_truth: pd.DataFrame) -> None:
    """
    Runs rigorous assertions to validate dataset integrity according to project requirements.
    """
    # 1. Total records check
    assert len(df_complaints) == TOTAL_COMPLAINTS, (
        f"Expected {TOTAL_COMPLAINTS} complaints, got {len(df_complaints)}"
    )

    # 2. Unique complaint IDs
    assert df_complaints["complaint_id"].nunique() == len(df_complaints), (
        "All complaint_id values must be unique!"
    )
    assert df_complaints["complaint_id"].str.match(r"^C\d{6}$").all(), (
        "Complaint ID must follow format C000001, C000002, etc."
    )

    # 3. Exactly 12-digit account numbers
    assert df_complaints["account_number"].str.match(r"^\d{12}$").all(), (
        "All account_number values must be strictly 12 digits!"
    )

    # 4. Valid IFSC format
    ifsc_regex = r"^[A-Z]{4}0[A-Z0-9]{6}$"
    assert df_complaints["ifsc"].str.match(ifsc_regex).all(), (
        "All IFSC codes must follow valid Indian banking IFSC format!"
    )

    # 5. Entity mapping consistency:
    # Every (account_number, ifsc) pair must map to exactly one ground_truth_entity_id
    grouped_entity = df_complaints.groupby(["account_number", "ifsc"])["ground_truth_entity_id"].nunique()
    assert (grouped_entity == 1).all(), (
        "Inconsistency detected: An account_number + IFSC combination maps to multiple entity IDs!"
    )

    # Every ground_truth_entity_id must map to exactly one (account_number, ifsc)
    grouped_acct = df_complaints.groupby("ground_truth_entity_id")[["account_number", "ifsc"]].nunique()
    assert (grouped_acct["account_number"] == 1).all() and (grouped_acct["ifsc"] == 1).all(), (
        "Inconsistency detected: An entity ID maps to multiple account numbers or IFSCs!"
    )

    # 6. Date bounds
    assert (df_complaints["complaint_date"] >= "2026-01-01").all(), "Date before 2026-01-01 detected!"
    assert (df_complaints["complaint_date"] <= "2026-08-24").all(), "Date after 2026-08-24 detected!"

    # 7. Reported amount bounds
    assert (df_complaints["reported_amount"] >= 500.0).all(), "Reported amount < 500 detected!"
    assert (df_complaints["reported_amount"] <= 500000.0).all(), "Reported amount > 500,000 detected!"

    # 8. Ground truth file consistency
    assert len(df_ground_truth) == df_complaints["ground_truth_entity_id"].nunique(), (
        "Ground truth table must match unique entity count!"
    )
    assert df_ground_truth["ground_truth_entity_id"].str.match(r"^ENT_\d{6}$").all(), (
        "Entity ID must follow format ENT_000001, ENT_000002, etc."
    )

    print("All integrity validations PASSED successfully!")


# ==============================================================================
# Summary Display Routine
# ==============================================================================

def print_summary(stats: Dict[str, Any]) -> None:
    """Prints a structured summary of the generated dataset metrics."""
    print("\n" + "=" * 65)
    print("      SYNTHETIC CYBERCRIME COMPLAINTS DATASET SUMMARY")
    print("=" * 65)
    print(f"Total Complaints Generated     : {stats['total_complaints']}")
    print(f"Unique Bank Accounts           : {stats['unique_accounts']}")
    print(f"Unique Entity IDs (Ground Truth): {stats['unique_entities']}")
    print("-" * 65)
    print(f"Base Unique Account Complaints : {stats['num_base_unique']}")
    print(f"Repeated-Account Complaints    : {stats['num_exact_duplicates']} (Exact name match)")
    print(f"Name-Variation Complaints      : {stats['num_name_variations']} (Fuzzy/modified name match)")
    print(f"Total Repeated References      : {stats['num_exact_duplicates'] + stats['num_name_variations']}")
    print("-" * 65)
    print(f"Date Range                     : {stats['date_min']} to {stats['date_max']}")
    print("-" * 65)
    print("Complaint Type Distribution:")
    for ctype, count in stats["type_counts"].items():
        pct = (count / stats["total_complaints"]) * 100
        print(f"  - {ctype:<25} : {count:>4} ({pct:>5.1f}%)")
    print("=" * 65 + "\n")


# ==============================================================================
# Main Execution Entrypoint
# ==============================================================================

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating {TOTAL_COMPLAINTS} synthetic complaints (Seed: {RANDOM_SEED})...")
    df_complaints, df_ground_truth, stats = generate_synthetic_dataset(
        seed=RANDOM_SEED,
        num_base=NUM_BASE_ACCOUNTS,
        num_exact_dup=NUM_EXACT_DUPLICATES,
        num_name_var=NUM_NAME_VARIATIONS
    )

    # Validate generated datasets
    validate_dataset(df_complaints, df_ground_truth)

    # Save to CSV files
    df_complaints.to_csv(COMPLAINTS_FILE, index=False)
    print(f"Saved complaints dataset to: {COMPLAINTS_FILE}")

    df_ground_truth.to_csv(GROUND_TRUTH_FILE, index=False)
    print(f"Saved entity ground truth to: {GROUND_TRUTH_FILE}")

    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()
