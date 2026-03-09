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
import pandas as pd

from config import (
    BEN_PAPER_COUNTRIES,
    BEN_META_TRAIN_CLASSES,
    BEN_META_VAL_CLASSES,
    BEN_META_TEST_CLASSES,
    BEN_BAD_PATCHES,
    BEN_ROOT_S1_PATH,
    BENR_ROOT_S2_PATH,
    RESULTS_DIR,
)
from src.datasets.dataset_s2 import (
    BigEarthNetS2Dataset,
    S2ValTransform, S2SupportTransform, S2TrainTransform,
)
from src.datasets.dataset_s1 import (
    BigEarthNetS1Dataset,
    S1ValTransform, S1SupportTransform, S1TrainTransform,
)
from src.models.protonet import ResNet12
from src.federated_learning.partitioner import partition_by_scenario
from src.federated_learning.client import build_clients, FedProtoClient, FedAvgClient
from src.federated_learning.server import FedProtoServer, FedAvgServer
from src.cetralised_trainer import run_centralized
from src.federated_trainier import run_federated
from src.utils.save import save_result, load_all_results
from src.utils.visualiser import (
    plot_convergence,
    plot_proto_distance_over_rounds,
    plot_per_class_proto_distance,
    plot_tsne_prototypes,
)

# ── Constants ──────────────────────────────────────────────────────────────────
N_WAY   = 5
Q_QUERY = 15
PAPER_COUNTRIES  = BEN_PAPER_COUNTRIES
META_TRAIN_CLASSES = BEN_META_TRAIN_CLASSES
META_VAL_CLASSES   = BEN_META_VAL_CLASSES
META_TEST_CLASSES  = BEN_META_TEST_CLASSES
BAD_PATCHES        = BEN_BAD_PATCHES


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_metadata(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_df = pd.read_csv(csv_path)
    meta = meta_df[meta_df["country"].isin(PAPER_COUNTRIES)].reset_index(drop=True)

    train_df = meta[meta["primary_label"].isin(META_TRAIN_CLASSES)].reset_index(drop=True)

    val_df = meta[meta["primary_label"].isin(META_VAL_CLASSES)].reset_index(drop=True)
    val_df = val_df[
        ~val_df["patch_id"].isin(BAD_PATCHES["S2"]) &
        ~val_df["s1_name"].isin(BAD_PATCHES["S1"])
    ].reset_index(drop=True)

    test_df = meta[meta["primary_label"].isin(META_TEST_CLASSES)].reset_index(drop=True)

    assert not set(META_TRAIN_CLASSES) & set(META_VAL_CLASSES)
    assert not set(META_TRAIN_CLASSES) & set(META_TEST_CLASSES)
    assert not set(META_VAL_CLASSES)   & set(META_TEST_CLASSES)

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    return train_df, val_df, test_df


def build_val_test_datasets(val_df, test_df, s2_root, s1_root):
    val_datasets = {
        "S2": BigEarthNetS2Dataset(val_df, s2_root,
                                   support_transform=S2ValTransform(),
                                   query_transform=S2ValTransform()),
        "S1": BigEarthNetS1Dataset(val_df, s1_root,
                                   support_transform=S1ValTransform(),
                                   query_transform=S1ValTransform()),
    }
    test_datasets = {
        "S2": BigEarthNetS2Dataset(test_df, s2_root,
                                   support_transform=S2ValTransform(),
                                   query_transform=S2ValTransform()),
        "S1": BigEarthNetS1Dataset(test_df, s1_root,
                                   support_transform=S1ValTransform(),
                                   query_transform=S1ValTransform()),
    }
    return val_datasets, test_datasets


def build_centralized_train(scenario: str, train_df: pd.DataFrame, s2_root, s1_root):
    summer_df      = train_df[train_df["season"] == "Summer"].reset_index(drop=True)
    all_season_df  = train_df.reset_index(drop=True)
    summer_s2_half = summer_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
    summer_s1_half = summer_df.sample(frac=0.5, random_state=42).reset_index(drop=True)

    if scenario == "DS1":
        df_s2, df_s1 = summer_df, summer_df
    elif scenario == "DS3":
        df_s2, df_s1 = all_season_df, all_season_df
    elif scenario == "DS4":
        df_s2, df_s1 = summer_s2_half, summer_df
    elif scenario == "DS5":
        df_s2, df_s1 = summer_df, summer_s1_half
    else:
        raise ValueError(f"Centralized baseline not defined for {scenario}")

    train_s2 = BigEarthNetS2Dataset(df_s2, s2_root,
                                    support_transform=S2SupportTransform(),
                                    query_transform=S2TrainTransform())
    train_s1 = BigEarthNetS1Dataset(df_s1, s1_root,
                                    support_transform=S1SupportTransform(),
                                    query_transform=S1TrainTransform())
    print(f"[Centralized {scenario}]  S2: {len(train_s2):,}  S1: {len(train_s1):,}")
    return train_s2, train_s1


# ── Experiment runners ────────────────────────────────────────────────────────

def run_centralized_experiment(args, train_df, val_datasets, test_datasets, device):
    results = {}
    k_shots = args.k_shots if args.k_shots else [1, 5]
    for k_shot in k_shots:
        label = f"Centralized_{args.scenario}_{k_shot}shot"
        train_s2, train_s1 = build_centralized_train(
            args.scenario, train_df, args.s2_root, args.s1_root
        )
        result = run_centralized(
            label=label,
            train_s2=train_s2,
            train_s1=train_s1,
            val_datasets=val_datasets,
            test_datasets=test_datasets,
            n_episodes=args.n_episodes,
            k_shot=k_shot,
            q_query=Q_QUERY,
            n_way=N_WAY,
            device=device,
        )
        results[label] = result
    return results


def run_federated_experiment(args, train_df, val_datasets, test_datasets, device):
    results = {}
    k_shots = args.k_shots if args.k_shots else [1, 5]
    ClientClass = FedProtoClient if args.method == "FedProto" else FedAvgClient

    for k_shot in k_shots:
        label = f"{args.method}_{args.scenario}_{k_shot}shot"
        partitions = partition_by_scenario(
            train_df, scenario=args.scenario, n_clients=args.n_clients
        )
        s2_encoder = ResNet12(in_channels=10)
        s1_encoder = ResNet12(in_channels=2)

        use_split = args.method == "FedAvg"
        s2_clients, s1_clients = build_clients(
            partitions=partitions,
            s2_root=args.s2_root,
            s1_root=args.s1_root,
            ClientClass=ClientClass,
            s2_encoder=s2_encoder,
            s1_encoder=s1_encoder,
            split_encoder=use_split,
            device=device,
            n_way=N_WAY,
            k_shot=k_shot,
            q_query=Q_QUERY,
        )

        if args.method == "FedProto":
            server = FedProtoServer(s2_clients=s2_clients, s1_clients=s1_clients, lam=0.1)
            shared_body = None
        else:
            shared_body = ResNet12(in_channels=10).to(device)
            server = FedAvgServer(
                shared_body=shared_body,
                s2_clients=s2_clients,
                s1_clients=s1_clients,
            )

        result = run_federated(
            label=label,
            server=server,
            s2_clients=s2_clients,
            s1_clients=s1_clients,
            shared_body=shared_body,
            val_datasets=val_datasets,
            test_datasets=test_datasets,
            n_rounds=args.n_rounds,
            n_episodes=args.n_episodes,
            k_shot=k_shot,
            q_query=Q_QUERY,
            n_way=N_WAY,
            device=device,
            track_protos=(args.method == "FedProto"),
            val_every=args.val_every,
        )
        results[label] = result
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(results: dict, scenario: str, method: str):
    for k_shot in [1, 5]:
        plot_convergence(results, modality="avg", k_shot=k_shot, scenario=scenario)

    label_1shot = f"{method}_{scenario}_1shot"
    label_5shot = f"{method}_{scenario}_5shot"

    if label_1shot in results and label_5shot in results:
        if method == "FedProto":
            plot_proto_distance_over_rounds(
                results,
                label_naive=label_1shot,
                label_ours=label_5shot,
            )
            plot_per_class_proto_distance(
                result_naive=results[label_1shot],
                result_ours=results[label_5shot],
                class_names={i: c for i, c in enumerate(META_TRAIN_CLASSES)},
            )
            plot_tsne_prototypes(
                result_naive=results[label_1shot],
                result_ours=results[label_5shot],
                class_names={i: c for i, c in enumerate(META_TRAIN_CLASSES)},
            )


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="FedCM-FSL experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode", choices=["federated", "centralized"], required=True,
        help="Training mode"
    )
    parser.add_argument(
        "--method", choices=["FedProto", "FedAvg"], default="FedProto",
        help="Federated method (ignored for centralized mode)"
    )
    parser.add_argument(
        "--scenario", choices=["DS1", "DS2", "DS3", "DS4", "DS5"], required=True,
        help="Data distribution scenario"
    )
    parser.add_argument(
        "--k-shots", type=int, nargs="+", default=None, metavar="K",
        help="k-shot values to run (default: [1, 5])"
    )
    parser.add_argument(
        "--n-clients", type=int, default=5,
        help="Number of federated clients"
    )
    parser.add_argument(
        "--n-rounds", type=int, default=20,
        help="Number of federated rounds (federated mode only)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=50,
        help="Episodes per round (federated) or total episodes (centralized)"
    )
    parser.add_argument(
        "--val-every", type=int, default=1,
        help="Validate every N rounds (federated mode only)"
    )
    parser.add_argument(
        "--metadata-csv", type=str,
        default="/home/vscode/FedCM-FSL/data/bigearthnet/metadata_sampled.csv",
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--s2-root", type=str,
        default=BENR_ROOT_S2_PATH or "/kaggle/input/datasets/pervolarakis/bigearth-federated-few-shot/S2",
        help="Root directory for S2 patches"
    )
    parser.add_argument(
        "--s1-root", type=str,
        default=BEN_ROOT_S1_PATH or "/kaggle/input/datasets/pervolarakis/bigearth-federated-few-shot/S1",
        help="Root directory for S1 patches"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plotting after training"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override, e.g. 'cuda', 'cuda:1', 'cpu' (default: auto-detect)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output dirs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Data
    train_df, val_df, test_df = load_metadata(args.metadata_csv)
    val_datasets, test_datasets = build_val_test_datasets(
        val_df, test_df, args.s2_root, args.s1_root
    )

    # Run
    if args.mode == "centralized":
        results = run_centralized_experiment(
            args, train_df, val_datasets, test_datasets, device
        )
    else:
        results = run_federated_experiment(
            args, train_df, val_datasets, test_datasets, device
        )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, result in results.items():
        test = result.get("test", {})
        s2 = test.get("S2", {})
        s1 = test.get("S1", {})
        print(f"{label}")
        print(f"  S2 test: {s2.get('str', 'N/A')}")
        print(f"  S1 test: {s1.get('str', 'N/A')}")

    # Plots
    if not args.no_plots:
        plot_results(results, scenario=args.scenario, method=args.method)


if __name__ == "__main__":
    main()
