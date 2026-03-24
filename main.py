"""
Experiment runner for FedCM-FSL.

Usage examples:
  # Federated FedProto, DS1, both k-shots
  python main.py --mode federated --method FedProto --scenario DS1

  # Centralized baseline, DS3, 5-shot only
  python main.py --mode centralized --scenario DS3 --k-shots 5

  # FedAvg, DS2, 1-shot, custom rounds
  python main.py --mode federated --method FedAvg --scenario DS2 --k-shots 1 --n-rounds 30

  # Run in background with nohup
  nohup python main.py --mode federated --method FedProto --scenario DS1 > logs/DS1_fedproto.log 2>&1 &
"""

import argparse
import sys
import os

# Ensure project root is on PYTHONPATH so 'config' and 'src' are importable
sys.path.insert(0, os.path.dirname(__file__))

import torch

from config import (
    BEN_ROOT_S1_PATH,
    BENR_ROOT_S2_PATH,
    RESULTS_DIR,
)
from src.runner import ExperimentRunner

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedCM-FSL experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["federated", "centralized"],
        required=True,
        help="Training mode",
    )
    parser.add_argument(
        "--dataset",
        choices=["BigEarthNet", "SEN12MS"],
        default="BigEarthNet",
        help="Dataset name",
    )
    parser.add_argument(
        "--method",
        choices=["FedProto", "FedAvg", "FedCMFSL", "FedProtoProj"],
        default="FedProto",
        help="Federated method (ignored for centralized mode)",
    )
    parser.add_argument(
        "--scenario",
        choices=["IID", "Non-IID"],
        required=True,
        help="Data distribution scenario",
    )
    parser.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="k-shot values to run (default: [1, 5])",
    )
    parser.add_argument(
        "--n-clients", type=int, default=5, help="Number of federated clients"
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=100,
        help="Number of federated rounds (federated mode only)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Episodes per round (federated) or total episodes (centralized)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="Validate every N rounds (federated mode only)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="/home/vscode/FedCM-FSL/data/bigearthnet/metadata_sampled.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--s2-root",
        type=str,
        default=BENR_ROOT_S2_PATH
        or "/kaggle/input/datasets/pervolarakis/bigearth-federated-few-shot/S2",
        help="Root directory for S2 patches",
    )
    parser.add_argument(
        "--s1-root",
        type=str,
        default=BEN_ROOT_S1_PATH
        or "/kaggle/input/datasets/pervolarakis/bigearth-federated-few-shot/S1",
        help="Root directory for S1 patches",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. 'cuda', 'cuda:1', 'cpu' (default: auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"Initializing Experiment | Mode: {args.mode} | Scenario: {args.scenario} | Device: {device}"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    try:
        runner = ExperimentRunner(args, device)
        results = runner.run()

        print("\n" + "=" * 60)
        print(f"{'EXPERIMENT COMPLETE':^60}")
        print("=" * 60)
        for label, result in results.items():
            test = result.get("test", {})
            print(f"ID: {label}")
            print(f"S2 Accuracy: {test.get('S2', {}).get('str', 'N/A')}")
            print(f"S1 Accuracy: {test.get('S1', {}).get('str', 'N/A')}")

    except Exception as e:
        print(f"Experiment failed with error: {e}")
        raise e


if __name__ == "__main__":
    main()
