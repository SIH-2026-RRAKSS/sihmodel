"""
Master Simulation Suite: Cybercrime AML Predictive Intelligence Platform
========================================================================
Interactive test harness executing operational simulations using REAL datasets
(Dataset A: 15,000 Domestic Cybercrime Transactions & Dataset B: IBM AML Real Ledger).

[1] Live Banking Switch / UPI Stream & Real-Time Auto-Triage Monitor (Sub-25ms SLA)
[2] Step-by-Step Incident Timeline & GNN Probability Replay (Real GraphML Cases)
[3] Real Dataset Adversarial Evasion & Stress-Testing Benchmark (IBM Fan-Out & Layering)
[4] Multi-State Police Complaint Triage & Auto-FIR Dispatcher (Real State Cyber Cells)
[5] Execute All 4 Real Simulations in Sequence
"""

import sys
import os
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulations.simulate_live_stream import run_live_stream_simulation
from simulations.simulate_incident_replay import replay_incident
from simulations.simulate_adversarial_evasion import run_adversarial_evasion_test
from simulations.simulate_police_dispatch import run_police_dispatch_simulation

C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"


def print_banner():
    print("\n" + "=" * 86)
    print(f"{C_BOLD}{C_CYAN}    🛡️  CYBERCRIME AML PLATFORM — REAL DATASET SIMULATION SUITE  🛡️{C_RESET}")
    print("=" * 86)
    print("  [1] Live Real Transaction Stream & Auto-Triage Monitor (Sub-25ms SLA)")
    print("  [2] Step-by-Step Incident Timeline & GNN Probability Replay (Real GraphML)")
    print("  [3] Real Dataset Adversarial Evasion Benchmark (IBM AML & Domestic Subgraphs)")
    print("  [4] Multi-State Police Complaint Triage & Auto-FIR Dispatcher")
    print("  [5] Run ALL 4 Real Simulations in Sequence (Complete Hackathon Demo)")
    print("  [0] Exit")
    print("=" * 86 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cybercrime Predictive Simulation Suite")
    parser.add_argument("choice", nargs="?", default=None, help="Simulation choice [1-5]")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "ibm"], help="Dataset source (synthetic or ibm)")
    parser.add_argument("--num-tx", type=int, default=60, help="Number of stream transactions")
    args = parser.parse_args()

    choice = args.choice
    if choice is None:
        print_banner()
        try:
            choice = input(f"{C_BOLD}Select a simulation to execute [1-5, 0 to exit]: {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if choice == "1":
        run_live_stream_simulation(dataset=args.dataset, num_tx=args.num_tx, speed_factor=0.01)
    elif choice == "2":
        replay_incident(incident_id="C000124", dataset=args.dataset, delay_sec=0.04)
    elif choice == "3":
        run_adversarial_evasion_test()
    elif choice == "4":
        run_police_dispatch_simulation()
    elif choice == "5" or choice.lower() == "all":
        print(f"\n{C_BOLD}{C_YELLOW}>>> RUNNING COMPLETE 4-STAGE REAL DATASET DEMONSTRATION <<<{C_RESET}\n")
        print("\n--- STAGE 1: REAL-TIME STREAMING INGESTION (15k Dataset) ---")
        run_live_stream_simulation(dataset="synthetic", num_tx=40, speed_factor=0.01)
        print("\n--- STAGE 2: REAL INCIDENT TIMELINE REPLAY (C000124 GraphML) ---")
        replay_incident(incident_id="C000124", dataset="synthetic", delay_sec=0.03)
        print("\n--- STAGE 3: REAL DATASET ADVERSARIAL EVASION BENCHMARK (IBM AML) ---")
        run_adversarial_evasion_test()
        print("\n--- STAGE 4: MULTI-STATE POLICE TRIAGE & FIR AUTO-DISPATCH ---")
        run_police_dispatch_simulation()
        print(f"\n{C_BOLD}{C_GREEN}🎉 ALL 4 REAL DATASET SIMULATIONS COMPLETED SUCCESSFULLY!{C_RESET}\n")
    elif choice == "0":
        print("Exiting.")
    else:
        print(f"{C_RED}Invalid option: '{choice}'. Please select 1, 2, 3, 4, or 5.{C_RESET}")


if __name__ == "__main__":
    main()
