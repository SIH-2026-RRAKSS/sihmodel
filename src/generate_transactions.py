"""
Synthetic Financial Transaction Dataset Generator
=================================================
This module generates a realistic synthetic financial transaction dataset between
the 700 resolved entities from Stage 0 (data/entity_master.csv) for Stage 2
Graph Construction and downstream ML (XGBoost / GraphSAGE).

Outputs:
1. data/transactions.csv - Exactly 15,000 transactions (Normal + Suspicious Mule Rings).
2. data/entity_locations.csv - Geographic location mapping for all 700 entities.

Key Features:
- 100% synthetic, reproducible (fixed random seed = 42).
- Approximately 80–90% normal activity and 10–20% suspicious mule-ring patterns.
- 25 distinct multi-hop suspicious rings with realistic topologies:
  * Linear Layering (A -> B -> C -> D -> E -> ATM)
  * Fan-in Aggregation (A, B, C -> D -> E -> ATM)
  * Fan-out Dispersion (A -> B, C, D -> ATM)
  * Layered Network Mesh (A -> B, C -> D -> E -> ATM)
  * Extended Mule Chains with Terminal Cash-outs
- Suspicious chains execute in rapid temporal sequence (<= 72-hour incident windows).
- Terminal cash-out nodes use a distinct namespace (ATM_001 to ATM_050).
- Explicit ground truth evaluation labels (is_suspicious, ring_id).
"""

import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


# ==============================================================================
# Configuration & Constants
# ==============================================================================

RANDOM_SEED = 42
TOTAL_TRANSACTIONS_TARGET = 15000
NUM_SUSPICIOUS_RINGS = 25

DATA_DIR = Path("data")
ENTITY_MASTER_FILE = DATA_DIR / "entity_master.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
ENTITY_LOCATIONS_FILE = DATA_DIR / "entity_locations.csv"

START_DATE = datetime(2026, 1, 1, 0, 0, 0)
END_DATE = datetime(2026, 8, 24, 23, 59, 59)

# Plausible Indian Cities with Geographic Coordinates
INDIAN_CITIES = [
    {"city": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639},
    {"city": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777},
    {"city": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567},
    {"city": "Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"city": "Bengaluru", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946},
    {"city": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867},
    {"city": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707},
    {"city": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714},
    {"city": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873},
    {"city": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
    {"city": "Patna", "state": "Bihar", "lat": 25.5941, "lon": 85.1376},
    {"city": "Bhubaneswar", "state": "Odisha", "lat": 20.2961, "lon": 85.8245},
    {"city": "Kochi", "state": "Kerala", "lat": 9.9312, "lon": 76.2673},
    {"city": "Bhopal", "state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126},
    {"city": "Chandigarh", "state": "Punjab", "lat": 30.7333, "lon": 76.7794}
]

NUM_ATM_NODES = 50


# ==============================================================================
# Helper Functions & Location Setup
# ==============================================================================

def generate_entity_locations(
    entities: List[str],
    rng: random.Random
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Generates deterministic synthetic geographic locations for all 700 entities.
    Returns DataFrame and fast lookup dictionary.
    """
    location_records = []
    location_lookup = {}

    for ent_id in entities:
        city_info = rng.choice(INDIAN_CITIES)
        # Small coordinate jitter within ~1-4 km of city center
        lat_jitter = rng.uniform(-0.035, 0.035)
        lon_jitter = rng.uniform(-0.035, 0.035)

        lat = round(city_info["lat"] + lat_jitter, 6)
        lon = round(city_info["lon"] + lon_jitter, 6)
        state = city_info["state"]
        city = city_info["city"]

        rec = {
            "entity_id": ent_id,
            "state": state,
            "city": city,
            "latitude": lat,
            "longitude": lon
        }
        location_records.append(rec)
        location_lookup[ent_id] = rec

    df_locations = pd.DataFrame(location_records)
    return df_locations, location_lookup


def generate_atm_nodes(rng: random.Random) -> Dict[str, Dict[str, Any]]:
    """
    Generates terminal ATM cash-out nodes (ATM_001 to ATM_050) across cities.
    """
    atm_lookup = {}
    for i in range(1, NUM_ATM_NODES + 1):
        atm_id = f"ATM_{i:03d}"
        city_info = rng.choice(INDIAN_CITIES)
        lat_jitter = rng.uniform(-0.04, 0.04)
        lon_jitter = rng.uniform(-0.04, 0.04)
        atm_lookup[atm_id] = {
            "entity_id": atm_id,
            "state": city_info["state"],
            "city": city_info["city"],
            "latitude": round(city_info["lat"] + lat_jitter, 6),
            "longitude": round(city_info["lon"] + lon_jitter, 6)
        }
    return atm_lookup


def random_timestamp(start: datetime, end: datetime, rng: random.Random) -> datetime:
    """Generates a random datetime between start and end."""
    delta_seconds = int((end - start).total_seconds())
    offset = rng.randint(0, delta_seconds)
    return start + timedelta(seconds=offset)


def generate_realistic_amount(category: str, rng: random.Random) -> float:
    """
    Generates realistic transaction amounts in INR:
    - small: 100 to 5,000 (many transactions)
    - medium: 5,000 to 50,000 (moderate transactions)
    - large: 50,000 to 500,000 (fewer transactions)
    """
    if category == "small":
        val = rng.uniform(100.0, 5000.0)
    elif category == "medium":
        val = rng.uniform(5000.0, 50000.0)
    elif category == "large":
        val = rng.uniform(50000.0, 500000.0)
    else:
        r = rng.random()
        if r < 0.65:
            val = rng.uniform(100.0, 5000.0)
        elif r < 0.90:
            val = rng.uniform(5000.0, 50000.0)
        else:
            val = rng.uniform(50000.0, 500000.0)
    return round(val, 2)


# ==============================================================================
# Suspicious Mule Ring Generation
# ==============================================================================

def generate_suspicious_rings(
    entities: List[str],
    location_lookup: Dict[str, Dict[str, Any]],
    atm_lookup: Dict[str, Dict[str, Any]],
    num_rings: int = NUM_SUSPICIOUS_RINGS,
    rng: random.Random = None
) -> List[Dict[str, Any]]:
    """
    Generates realistic multi-hop suspicious mule ring transactions.
    Creates 20-30 rings with various network topologies executing in rapid temporal sequence (<= 72h).
    """
    suspicious_txs: List[Dict[str, Any]] = []
    atm_ids = list(atm_lookup.keys())

    topologies = [
        "linear_chain",
        "fan_in_aggregation",
        "fan_out_dispersion",
        "layered_network",
        "mule_chain_cashout"
    ]

    for ring_idx in range(1, num_rings + 1):
        ring_id = f"RING_{ring_idx:03d}"
        topology = topologies[(ring_idx - 1) % len(topologies)]
        
        # Each ring uses 5 to 9 distinct entities
        ring_size = rng.randint(5, 9)
        ring_entities = rng.sample(entities, ring_size)
        target_atms = rng.sample(atm_ids, rng.randint(1, 3))

        # Each ring conducts multiple incident episodes across the 8-month window
        num_episodes = rng.randint(12, 18)

        for _ in range(num_episodes):
            ep_start = random_timestamp(START_DATE, END_DATE - timedelta(days=4), rng)
            base_amount = rng.uniform(40000.0, 450000.0)

            if topology == "linear_chain":
                # Linear Layering: E0 -> E1 -> E2 -> ... -> Ek-1 -> ATM
                curr_time = ep_start
                curr_amt = base_amount
                for hop in range(len(ring_entities) - 1):
                    sender = ring_entities[hop]
                    receiver = ring_entities[hop + 1]
                    s_state = location_lookup[sender]["state"]
                    r_state = location_lookup[receiver]["state"]

                    tx_type = rng.choice(["UPI", "IMPS", "NEFT"])
                    channel = "UPI App" if tx_type == "UPI" else ("Mobile Banking" if tx_type == "IMPS" else "Internet Banking")

                    suspicious_txs.append({
                        "sender_entity_id": sender,
                        "receiver_entity_id": receiver,
                        "amount": round(curr_amt, 2),
                        "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "transaction_type": tx_type,
                        "channel": channel,
                        "sender_state": s_state,
                        "receiver_state": r_state,
                        "is_cash_out": 0,
                        "is_suspicious": 1,
                        "ring_id": ring_id
                    })

                    curr_time += timedelta(minutes=rng.randint(15, 75))
                    curr_amt *= rng.uniform(0.95, 0.98)

                # Terminal ATM withdrawal
                last_entity = ring_entities[-1]
                atm_node = rng.choice(target_atms)
                suspicious_txs.append({
                    "sender_entity_id": last_entity,
                    "receiver_entity_id": atm_node,
                    "amount": round(curr_amt, 2),
                    "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "transaction_type": "CASH_WITHDRAWAL",
                    "channel": "ATM",
                    "sender_state": location_lookup[last_entity]["state"],
                    "receiver_state": atm_lookup[atm_node]["state"],
                    "is_cash_out": 1,
                    "is_suspicious": 1,
                    "ring_id": ring_id
                })

            elif topology == "fan_in_aggregation":
                # Multiple source mules -> Aggregator mule -> ATM
                aggregator = ring_entities[-1]
                sources = ring_entities[:-1]
                curr_time = ep_start
                total_aggregated = 0.0

                for src in sources:
                    part_amt = (base_amount / len(sources)) * rng.uniform(0.92, 1.08)
                    total_aggregated += part_amt
                    s_state = location_lookup[src]["state"]
                    r_state = location_lookup[aggregator]["state"]

                    tx_type = rng.choice(["UPI", "IMPS"])
                    channel = "UPI App" if tx_type == "UPI" else "Mobile Banking"

                    suspicious_txs.append({
                        "sender_entity_id": src,
                        "receiver_entity_id": aggregator,
                        "amount": round(part_amt, 2),
                        "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "transaction_type": tx_type,
                        "channel": channel,
                        "sender_state": s_state,
                        "receiver_state": r_state,
                        "is_cash_out": 0,
                        "is_suspicious": 1,
                        "ring_id": ring_id
                    })
                    curr_time += timedelta(minutes=rng.randint(10, 45))

                # Aggregator withdraws at ATM
                curr_time += timedelta(minutes=rng.randint(20, 60))
                atm_node = rng.choice(target_atms)
                suspicious_txs.append({
                    "sender_entity_id": aggregator,
                    "receiver_entity_id": atm_node,
                    "amount": round(total_aggregated * rng.uniform(0.95, 0.98), 2),
                    "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "transaction_type": "CASH_WITHDRAWAL",
                    "channel": "ATM",
                    "sender_state": location_lookup[aggregator]["state"],
                    "receiver_state": atm_lookup[atm_node]["state"],
                    "is_cash_out": 1,
                    "is_suspicious": 1,
                    "ring_id": ring_id
                })

            elif topology == "fan_out_dispersion":
                # Primary mule disperses to multiple layer-1 mules -> ATMs
                source = ring_entities[0]
                recipients = ring_entities[1:]
                curr_time = ep_start

                for rec in recipients:
                    part_amt = (base_amount / len(recipients)) * rng.uniform(0.92, 1.05)
                    s_state = location_lookup[source]["state"]
                    r_state = location_lookup[rec]["state"]

                    tx_type = rng.choice(["IMPS", "UPI", "NEFT"])
                    channel = "Mobile Banking" if tx_type == "IMPS" else ("UPI App" if tx_type == "UPI" else "Internet Banking")

                    suspicious_txs.append({
                        "sender_entity_id": source,
                        "receiver_entity_id": rec,
                        "amount": round(part_amt, 2),
                        "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "transaction_type": tx_type,
                        "channel": channel,
                        "sender_state": s_state,
                        "receiver_state": r_state,
                        "is_cash_out": 0,
                        "is_suspicious": 1,
                        "ring_id": ring_id
                    })
                    curr_time += timedelta(minutes=rng.randint(15, 45))

                    atm_node = rng.choice(target_atms)
                    cash_time = curr_time + timedelta(minutes=rng.randint(15, 60))
                    suspicious_txs.append({
                        "sender_entity_id": rec,
                        "receiver_entity_id": atm_node,
                        "amount": round(part_amt * rng.uniform(0.95, 0.98), 2),
                        "timestamp": cash_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "transaction_type": "CASH_WITHDRAWAL",
                        "channel": "ATM",
                        "sender_state": location_lookup[rec]["state"],
                        "receiver_state": atm_lookup[atm_node]["state"],
                        "is_cash_out": 1,
                        "is_suspicious": 1,
                        "ring_id": ring_id
                    })

            elif topology == "layered_network":
                # 5 distinct nodes: E0 -> E1 & E2; E1 -> E3, E2 -> E3; E3 -> E4 -> ATM
                e0, e1, e2, e3, e4 = ring_entities[0], ring_entities[1], ring_entities[2], ring_entities[3], ring_entities[4]
                curr_time = ep_start
                amt_1 = base_amount * 0.52
                amt_2 = base_amount * 0.48

                for dest, amt in [(e1, amt_1), (e2, amt_2)]:
                    suspicious_txs.append({
                        "sender_entity_id": e0,
                        "receiver_entity_id": dest,
                        "amount": round(amt, 2),
                        "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "transaction_type": "IMPS",
                        "channel": "Mobile Banking",
                        "sender_state": location_lookup[e0]["state"],
                        "receiver_state": location_lookup[dest]["state"],
                        "is_cash_out": 0,
                        "is_suspicious": 1,
                        "ring_id": ring_id
                    })
                    curr_time += timedelta(minutes=rng.randint(20, 60))

                curr_time += timedelta(minutes=rng.randint(30, 90))
                for src, amt in [(e1, amt_1 * 0.96), (e2, amt_2 * 0.96)]:
                    suspicious_txs.append({
                        "sender_entity_id": src,
                        "receiver_entity_id": e3,
                        "amount": round(amt, 2),
                        "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "transaction_type": "UPI",
                        "channel": "UPI App",
                        "sender_state": location_lookup[src]["state"],
                        "receiver_state": location_lookup[e3]["state"],
                        "is_cash_out": 0,
                        "is_suspicious": 1,
                        "ring_id": ring_id
                    })
                    curr_time += timedelta(minutes=rng.randint(15, 45))

                curr_time += timedelta(minutes=rng.randint(30, 120))
                total_d = (amt_1 + amt_2) * 0.92
                suspicious_txs.append({
                    "sender_entity_id": e3,
                    "receiver_entity_id": e4,
                    "amount": round(total_d, 2),
                    "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "transaction_type": "NEFT",
                    "channel": "Internet Banking",
                    "sender_state": location_lookup[e3]["state"],
                    "receiver_state": location_lookup[e4]["state"],
                    "is_cash_out": 0,
                    "is_suspicious": 1,
                    "ring_id": ring_id
                })

                curr_time += timedelta(minutes=rng.randint(20, 60))
                atm_node = rng.choice(target_atms)
                suspicious_txs.append({
                    "sender_entity_id": e4,
                    "receiver_entity_id": atm_node,
                    "amount": round(total_d * 0.97, 2),
                    "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "transaction_type": "CASH_WITHDRAWAL",
                    "channel": "ATM",
                    "sender_state": location_lookup[e4]["state"],
                    "receiver_state": atm_lookup[atm_node]["state"],
                    "is_cash_out": 1,
                    "is_suspicious": 1,
                    "ring_id": ring_id
                })

            else:  # mule_chain_cashout
                # Multi-stage chain: E0 -> E1 -> E2 -> ... -> Ek-1 -> ATM
                curr_time = ep_start
                curr_amt = base_amount
                for hop in range(len(ring_entities) - 1):
                    sender = ring_entities[hop]
                    receiver = ring_entities[hop + 1]
                    s_state = location_lookup[sender]["state"]
                    r_state = location_lookup[receiver]["state"]

                    tx_type = rng.choice(["UPI", "IMPS", "NEFT"])
                    channel = "UPI App" if tx_type == "UPI" else ("Mobile Banking" if tx_type == "IMPS" else "Internet Banking")

                    suspicious_txs.append({
                        "sender_entity_id": sender,
                        "receiver_entity_id": receiver,
                        "amount": round(curr_amt, 2),
                        "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "transaction_type": tx_type,
                        "channel": channel,
                        "sender_state": s_state,
                        "receiver_state": r_state,
                        "is_cash_out": 0,
                        "is_suspicious": 1,
                        "ring_id": ring_id
                    })

                    curr_time += timedelta(minutes=rng.randint(20, 90))
                    curr_amt *= rng.uniform(0.94, 0.97)

                terminal_entity = ring_entities[-1]
                atm_node = rng.choice(target_atms)
                suspicious_txs.append({
                    "sender_entity_id": terminal_entity,
                    "receiver_entity_id": atm_node,
                    "amount": round(curr_amt, 2),
                    "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "transaction_type": "CASH_WITHDRAWAL",
                    "channel": "ATM",
                    "sender_state": location_lookup[terminal_entity]["state"],
                    "receiver_state": atm_lookup[atm_node]["state"],
                    "is_cash_out": 1,
                    "is_suspicious": 1,
                    "ring_id": ring_id
                })

    return suspicious_txs


# ==============================================================================
# Normal Transactions Generation
# ==============================================================================

def generate_normal_transactions(
    entities: List[str],
    location_lookup: Dict[str, Dict[str, Any]],
    atm_lookup: Dict[str, Dict[str, Any]],
    target_count: int,
    rng: random.Random
) -> List[Dict[str, Any]]:
    """
    Generates realistic benign financial transaction behavior.
    Includes peer-to-peer, merchant, recurring payments, and normal ATM cash-outs.
    """
    normal_txs: List[Dict[str, Any]] = []
    atm_ids = list(atm_lookup.keys())

    # Pre-generate social/business transaction affinities
    entity_affinity: Dict[str, List[str]] = {}
    for ent in entities:
        k = rng.randint(4, 12)
        entity_affinity[ent] = rng.sample([e for e in entities if e != ent], k)

    tx_type_choices = ["UPI", "IMPS", "NEFT", "CARD", "RTGS", "CASH_WITHDRAWAL"]
    tx_type_weights = [0.52, 0.20, 0.14, 0.07, 0.02, 0.05]

    for _ in range(target_count):
        sender = rng.choice(entities)
        tx_type = rng.choices(tx_type_choices, weights=tx_type_weights)[0]

        # Diurnal time distribution
        base_dt = random_timestamp(START_DATE, END_DATE, rng)
        hour_weights = [
            0.01, 0.01, 0.005, 0.005, 0.01, 0.02, 0.03, 0.05,
            0.07, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06,
            0.07, 0.08, 0.08, 0.07, 0.05, 0.04, 0.02, 0.01
        ]
        hour = rng.choices(range(24), weights=hour_weights)[0]
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        tx_time = base_dt.replace(hour=hour, minute=minute, second=second)

        if tx_type == "CASH_WITHDRAWAL":
            receiver = rng.choice(atm_ids)
            channel = "ATM"
            is_cash_out = 1
            amount = generate_realistic_amount("small" if rng.random() < 0.85 else "medium", rng)
            s_state = location_lookup[sender]["state"]
            r_state = atm_lookup[receiver]["state"]

        else:
            is_cash_out = 0
            if rng.random() < 0.70:
                receiver = rng.choice(entity_affinity[sender])
            else:
                receiver = rng.choice([e for e in entities if e != sender])

            s_state = location_lookup[sender]["state"]
            r_state = location_lookup[receiver]["state"]

            if tx_type == "UPI":
                channel = "UPI App"
                amount = generate_realistic_amount("small", rng)
            elif tx_type == "IMPS":
                channel = rng.choice(["Mobile Banking", "Internet Banking"])
                amount = generate_realistic_amount("medium" if rng.random() < 0.6 else "small", rng)
            elif tx_type == "NEFT":
                channel = rng.choice(["Internet Banking", "Bank Branch"])
                amount = generate_realistic_amount("medium" if rng.random() < 0.7 else "large", rng)
            elif tx_type == "CARD":
                channel = rng.choice(["POS", "Internet Banking"])
                amount = generate_realistic_amount("small" if rng.random() < 0.8 else "medium", rng)
            else:  # RTGS
                channel = rng.choice(["Internet Banking", "Bank Branch"])
                amount = rng.uniform(200000.0, 500000.0)

        normal_txs.append({
            "sender_entity_id": sender,
            "receiver_entity_id": receiver,
            "amount": round(amount, 2),
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_type": tx_type,
            "channel": channel,
            "sender_state": s_state,
            "receiver_state": r_state,
            "is_cash_out": is_cash_out,
            "is_suspicious": 0,
            "ring_id": "NORMAL"
        })

    return normal_txs


# ==============================================================================
# Validation Routine
# ==============================================================================

def validate_transaction_dataset(
    df_tx: pd.DataFrame,
    df_locations: pd.DataFrame,
    entities: List[str],
    atm_lookup: Dict[str, Dict[str, Any]]
) -> None:
    """
    Performs comprehensive validation checks on generated transactions and locations.
    """
    valid_entity_set = set(entities)
    valid_atm_set = set(atm_lookup.keys())
    valid_receiver_set = valid_entity_set.union(valid_atm_set)

    # 1. Unique transaction_id
    assert len(df_tx) == TOTAL_TRANSACTIONS_TARGET, (
        f"Expected {TOTAL_TRANSACTIONS_TARGET} transactions, got {len(df_tx)}"
    )
    assert df_tx["transaction_id"].nunique() == len(df_tx), "transaction_id values are not unique!"
    assert df_tx["transaction_id"].str.match(r"^T\d{6}$").all(), "transaction_id must follow format T000001, etc."

    # 2. Sender validation
    assert df_tx["sender_entity_id"].isin(valid_entity_set).all(), (
        "Found sender_entity_id not present in entity_master.csv!"
    )

    # 3. Receiver validation
    assert df_tx["receiver_entity_id"].isin(valid_receiver_set).all(), (
        "Found receiver_entity_id not present in entity_master.csv or ATM registry!"
    )

    # 4. No self transactions
    assert (df_tx["sender_entity_id"] != df_tx["receiver_entity_id"]).all(), (
        "Found illegal transaction where sender_entity_id == receiver_entity_id!"
    )

    # 5. Positive amounts
    assert (df_tx["amount"] > 0).all(), "Found non-positive transaction amount!"

    # 6. Valid timestamps
    parsed_dates = pd.to_datetime(df_tx["timestamp"])
    assert (parsed_dates >= START_DATE).all() and (parsed_dates <= END_DATE).all(), (
        "Found timestamp outside expected date range (2026-01-01 to 2026-08-24)!"
    )

    # 7. is_cash_out is 0 or 1
    assert df_tx["is_cash_out"].isin([0, 1]).all(), "is_cash_out must be strictly 0 or 1!"

    # 8. is_suspicious is 0 or 1
    assert df_tx["is_suspicious"].isin([0, 1]).all(), "is_suspicious must be strictly 0 or 1!"

    # 9. ring_id validation
    suspicious_mask = df_tx["is_suspicious"] == 1
    normal_mask = df_tx["is_suspicious"] == 0

    assert df_tx.loc[suspicious_mask, "ring_id"].str.match(r"^RING_\d{3}$").all(), (
        "Every suspicious transaction must have ring_id following format RING_XXX!"
    )
    assert (df_tx.loc[normal_mask, "ring_id"] == "NORMAL").all(), (
        "Every normal transaction must have ring_id = 'NORMAL'!"
    )

    # 10. Entity universe coverage: verify all 700 entities appear
    active_entities = set(df_tx["sender_entity_id"]).union(set(df_tx["receiver_entity_id"].loc[~df_tx["receiver_entity_id"].isin(valid_atm_set)]))
    assert len(active_entities) == len(valid_entity_set), (
        f"Expected all {len(valid_entity_set)} entities to participate, but only {len(active_entities)} found!"
    )

    # 11. Multi-hop verification in suspicious rings
    suspicious_df = df_tx[suspicious_mask]
    for ring_id, ring_group in suspicious_df.groupby("ring_id"):
        assert len(ring_group) >= 4, f"{ring_id} contains fewer than 4 transactions!"
        assert ring_group["sender_entity_id"].nunique() >= 3, f"{ring_id} does not span multiple sender entities!"
        assert (ring_group["is_cash_out"] == 1).any(), f"{ring_id} does not contain any cash-out event!"

    # 12. Location mapping validation
    assert len(df_locations) == len(entities), "entity_locations.csv must contain exactly 700 rows!"
    assert df_locations["entity_id"].nunique() == len(entities), "entity_id must be unique in entity_locations.csv!"

    print("All data quality validations PASSED successfully!")


# ==============================================================================
# Summary Report & Output Routine
# ==============================================================================

def print_dataset_summary(
    df_tx: pd.DataFrame,
    df_locations: pd.DataFrame
) -> None:
    """Prints a structured summary report of the generated synthetic transactions."""
    total_tx = len(df_tx)
    suspicious_count = (df_tx["is_suspicious"] == 1).sum()
    normal_count = (df_tx["is_suspicious"] == 0).sum()
    num_rings = df_tx.loc[df_tx["is_suspicious"] == 1, "ring_id"].nunique()
    cash_out_count = (df_tx["is_cash_out"] == 1).sum()

    dt_min = df_tx["timestamp"].min()
    dt_max = df_tx["timestamp"].max()

    avg_amt = df_tx["amount"].mean()
    max_amt = df_tx["amount"].max()

    type_counts = df_tx["transaction_type"].value_counts()

    print("\n" + "=" * 48)
    print("      SYNTHETIC TRANSACTION DATASET")
    print("=" * 48)
    print(f"Total transactions         : {total_tx}")
    print(f"Unique financial entities  : {df_locations['entity_id'].nunique()}")
    print(f"Suspicious transactions    : {suspicious_count} ({suspicious_count / total_tx * 100:.1f}%)")
    print(f"Normal transactions        : {normal_count} ({normal_count / total_tx * 100:.1f}%)")
    print(f"Suspicious rings           : {num_rings}")
    print(f"Cash-out transactions      : {cash_out_count} ({cash_out_count / total_tx * 100:.1f}%)")
    print("-" * 48)
    print(f"Date range:\n{dt_min} → {dt_max}")
    print("-" * 48)
    print(f"Average transaction amount : ₹{avg_amt:,.2f}")
    print(f"Maximum transaction amount : ₹{max_amt:,.2f}")
    print("-" * 48)
    print("Transaction type distribution:")
    for t_type, count in type_counts.items():
        print(f"  - {t_type:<18} : {count:>5} ({count / total_tx * 100:>4.1f}%)")
    print("=" * 48 + "\n")


# ==============================================================================
# Main Generation Pipeline
# ==============================================================================

def main():
    if not ENTITY_MASTER_FILE.exists():
        raise FileNotFoundError(f"Missing required entity master file: {ENTITY_MASTER_FILE}")

    # Load 700 resolved entities from Stage 0
    df_entity_master = pd.read_csv(ENTITY_MASTER_FILE)
    entities = df_entity_master["entity_id"].tolist()
    print(f"Loaded {len(entities)} master entities from {ENTITY_MASTER_FILE}")

    rng = random.Random(RANDOM_SEED)

    # Step 1: Generate Entity Geographic Locations & ATM Nodes
    print("Generating entity location mapping and ATM terminal nodes...")
        df_locations, location_lookup = generate_entity_locations(entities, rng)
    atm_lookup = generate_atm_nodes(rng)
    
    # ADD ATMs to df_locations so they are saved
    atm_records = []
    for atm_id, atm_data in atm_lookup.items():
        atm_records.append({
            "entity_id": atm_id,
            "latitude": atm_data["latitude"],
            "longitude": atm_data["longitude"]
        })
    df_atms = pd.DataFrame(atm_records)
    df_locations = pd.concat([df_locations, df_atms], ignore_index=True)


    # Step 2: Generate Suspicious Mule Ring Transactions (~2,700 transactions)
    print(f"Generating synthetic suspicious activity across {NUM_SUSPICIOUS_RINGS} mule rings...")
    suspicious_txs = generate_suspicious_rings(
        entities=entities,
        location_lookup=location_lookup,
        atm_lookup=atm_lookup,
        num_rings=NUM_SUSPICIOUS_RINGS,
        rng=rng
    )

    # Step 3: Generate Normal Transactions to reach exactly TOTAL_TRANSACTIONS_TARGET
    normal_target = TOTAL_TRANSACTIONS_TARGET - len(suspicious_txs)
    print(f"Generating {normal_target} benign financial transactions...")
    normal_txs = generate_normal_transactions(
        entities=entities,
        location_lookup=location_lookup,
        atm_lookup=atm_lookup,
        target_count=normal_target,
        rng=rng
    )

    # Step 4: Combine, Chronologically Sort, and Assign Transaction IDs
    all_txs = suspicious_txs + normal_txs
    df_tx = pd.DataFrame(all_txs)

    # Sort strictly by timestamp
    df_tx["dt"] = pd.to_datetime(df_tx["timestamp"])
    df_tx = df_tx.sort_values("dt").reset_index(drop=True)
    df_tx = df_tx.drop(columns=["dt"])

    # Assign sequential transaction IDs
    df_tx["transaction_id"] = [f"T{i + 1:06d}" for i in range(len(df_tx))]

    # Order columns
    ordered_cols = [
        "transaction_id",
        "sender_entity_id",
        "receiver_entity_id",
        "amount",
        "timestamp",
        "transaction_type",
        "channel",
        "sender_state",
        "receiver_state",
        "is_cash_out",
        "is_suspicious",
        "ring_id"
    ]
    df_tx = df_tx[ordered_cols]

    # Step 5: Validate Datasets
    validate_transaction_dataset(df_tx, df_locations, entities, atm_lookup)

    # Step 6: Save CSV Files
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_tx.to_csv(TRANSACTIONS_FILE, index=False)
    print(f"[SUCCESS] Saved transactions dataset to: {TRANSACTIONS_FILE}")

    df_locations.to_csv(ENTITY_LOCATIONS_FILE, index=False)
    print(f"[SUCCESS] Saved entity locations to    : {ENTITY_LOCATIONS_FILE}")

    # Step 7: Print Summary Report
    print_dataset_summary(df_tx, df_locations)

    # Step 8: Demonstration Samples
    print("Sample 10 Transactions (from data/transactions.csv):")
    print("-" * 120)
    print(f"{'Tx ID':<8} | {'Sender':<11} | {'Receiver':<11} | {'Amount (INR)':>12} | {'Timestamp':<19} | {'Type':<15} | {'CashOut':<7} | {'Susp':<4} | {'Ring'}")
    print("-" * 120)
    for _, r in df_tx.head(10).iterrows():
        print(f"{r['transaction_id']:<8} | {r['sender_entity_id']:<11} | {r['receiver_entity_id']:<11} | ₹{r['amount']:>10.2f} | {r['timestamp']:<19} | {r['transaction_type']:<15} | {r['is_cash_out']:<7} | {r['is_suspicious']:<4} | {r['ring_id']}")
    print("-" * 120 + "\n")

    print("5 Example Suspicious Transaction Sequences (from RING_001):")
    print("-" * 120)
    ring1_sample = df_tx[df_tx["ring_id"] == "RING_001"].head(5)
    for _, r in ring1_sample.iterrows():
        print(f"{r['transaction_id']:<8} | {r['sender_entity_id']:<11} -> {r['receiver_entity_id']:<11} | ₹{r['amount']:>10.2f} | {r['timestamp']:<19} | {r['transaction_type']:<15} | CashOut: {r['is_cash_out']} | {r['ring_id']}")
    print("-" * 120 + "\n")

    print("5 Example Normal Transactions (Sample Rows):")
    print("-" * 120)
    normal_sample = df_tx[df_tx["ring_id"] == "NORMAL"].head(5)
    for _, r in normal_sample.iterrows():
        print(f"{r['transaction_id']:<8} | {r['sender_entity_id']:<11} -> {r['receiver_entity_id']:<11} | ₹{r['amount']:>10.2f} | {r['timestamp']:<19} | {r['transaction_type']:<15} | CashOut: {r['is_cash_out']} | {r['ring_id']}")
    print("-" * 120 + "\n")


if __name__ == "__main__":
    main()
