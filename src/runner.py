import torch
import pandas as pd
from config import (
    BEN_COUNTRIES,
    BEN_META_TRAIN_CLASSES,
    BEN_META_VAL_CLASSES,
    BEN_BAD_PATCHES,
    BEN_META_TEST_CLASSES,
    BEN_Q_QUERY,
    BEN_N_WAY,
)
from src.datasets.dataset_s1 import (
    BigEarthNetS1Dataset,
    S1ValTransform,
    S1SupportTransform,
    S1TrainTransform,
)
from src.datasets.dataset_s2 import (
    BigEarthNetS2Dataset,
    S2ValTransform,
    S2SupportTransform,
    S2TrainTransform,
)
from src.cetralised_trainer import run_centralized
from src.federated_learning.factory import FederatedFactory
from src.federated_trainier import run_federated
from src.federated_learning.partitioner import partition_by_scenario
from src.models.protonet import ResNet12
from src.federated_learning.client import build_clients


class ExperimentRunner:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.train_df, self.val_ds, self.test_ds = self._setup_data()

    def run(self):
        if self.args.mode == "centralized":
            return self._run_centralized()
        return self._run_federated()

    def _setup_data(self):
        if self.args.dataset == "BigEarthNet":
            meta_df = pd.read_csv(self.args.metadata_csv)
            meta = meta_df[meta_df["country"].isin(BEN_COUNTRIES)].reset_index(
                drop=True
            )

            train_df = meta[
                meta["primary_label"].isin(BEN_META_TRAIN_CLASSES)
            ].reset_index(drop=True)

            val_df = meta[meta["primary_label"].isin(BEN_META_VAL_CLASSES)].reset_index(
                drop=True
            )
            val_df = val_df[
                ~val_df["patch_id"].isin(BEN_BAD_PATCHES["S2"])
                & ~val_df["s1_name"].isin(BEN_BAD_PATCHES["S1"])
            ].reset_index(drop=True)

            test_df = meta[
                meta["primary_label"].isin(BEN_META_TEST_CLASSES)
            ].reset_index(drop=True)

            assert not set(BEN_META_TRAIN_CLASSES) & set(BEN_META_VAL_CLASSES)
            assert not set(BEN_META_TRAIN_CLASSES) & set(BEN_META_TEST_CLASSES)
            assert not set(BEN_META_VAL_CLASSES) & set(BEN_META_TEST_CLASSES)

            print(
                f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}"
            )

            val_datasets = {
                "S2": BigEarthNetS2Dataset(
                    val_df,
                    self.args.s2_root,
                    support_transform=S2ValTransform(),
                    query_transform=S2ValTransform(),
                ),
                "S1": BigEarthNetS1Dataset(
                    val_df,
                    self.args.s1_root,
                    support_transform=S1ValTransform(),
                    query_transform=S1ValTransform(),
                ),
            }
            test_datasets = {
                "S2": BigEarthNetS2Dataset(
                    test_df,
                    self.args.s2_root,
                    support_transform=S2ValTransform(),
                    query_transform=S2ValTransform(),
                ),
                "S1": BigEarthNetS1Dataset(
                    test_df,
                    self.args.s1_root,
                    support_transform=S1ValTransform(),
                    query_transform=S1ValTransform(),
                ),
            }
            return train_df, val_datasets, test_datasets

    def _run_centralised(self):
        results = {}
        k_shots = self.args.k_shots if self.args.k_shots else [1, 5]
        for k_shot in k_shots:
            label = f"Centralized_{self.args.scenario}_{k_shot}shot"
            if self.args.dataset == "BigEarthNet":
                summer_df = self.train_df[
                    self.train_df["season"] == "Summer"
                ].reset_index(drop=True)
                all_season_df = self.train_df.reset_index(drop=True)
                summer_s2_half = summer_df.sample(
                    frac=0.5, random_state=42
                ).reset_index(drop=True)
                summer_s1_half = summer_df.sample(
                    frac=0.5, random_state=42
                ).reset_index(drop=True)

                if self.args.scenario == "DS1":
                    df_s2, df_s1 = summer_df, summer_df
                elif self.args.scenario == "DS3":
                    df_s2, df_s1 = all_season_df, all_season_df
                elif self.args.scenario == "DS4":
                    df_s2, df_s1 = summer_s2_half, summer_df
                elif self.args.scenario == "DS5":
                    df_s2, df_s1 = summer_df, summer_s1_half
                else:
                    raise ValueError(
                        f"Centralized baseline not defined for {self.args.scenario }"
                    )

                train_s2 = BigEarthNetS2Dataset(
                    df_s2,
                    self.args.s2_root,
                    support_transform=S2SupportTransform(),
                    query_transform=S2TrainTransform(),
                )
                train_s1 = BigEarthNetS1Dataset(
                    df_s1,
                    self.args.s1_root,
                    support_transform=S1SupportTransform(),
                    query_transform=S1TrainTransform(),
                )
                print(
                    f"[Centralized {self.args.scenario}]  S2: {len(train_s2):,}  S1: {len(train_s1):,}"
                )

                result = run_centralized(
                    label=label,
                    train_s2=train_s2,
                    train_s1=train_s1,
                    val_datasets=self.val_ds,
                    test_datasets=self.test_ds,
                    n_episodes=self.args.n_episodes,
                    k_shot=k_shot,
                    q_query=BEN_Q_QUERY,
                    n_way=BEN_N_WAY,
                    device=self.device,
                )
                results[label] = result

    def _run_federated(self):
        results = {}
        k_shots = self.args.k_shots if self.args.k_shots else [1, 5]
        _, ClientClass, _ = FederatedFactory.get_components(
            self.args, [], [], self.device
        )
        for k_shot in k_shots:
            label = f"{self.args.method}_{self.args.scenario}_{k_shot}shot"
            partitions = partition_by_scenario(
                self.train_df, self.args.scenario, self.args.n_clients
            )

            s2_encoder = ResNet12(in_channels=10)
            s1_encoder = ResNet12(in_channels=2)

            use_split = self.args.method == "FedAvg"
            s2_clients, s1_clients = build_clients(
                partitions=partitions,
                s2_root=self.args.s2_root,
                s1_root=self.args.s1_root,
                ClientClass=ClientClass,
                s2_encoder=s2_encoder,
                s1_encoder=s1_encoder,
                split_encoder=use_split,
                device=self.device,
                n_way=BEN_N_WAY,
                k_shot=k_shot,
                q_query=BEN_Q_QUERY,
            )

            server, _, shared_body = FederatedFactory.get_components(
                self.args, s2_clients, s1_clients, self.device
            )

            result = run_federated(
                label=label,
                server=server,
                s2_clients=s2_clients,
                s1_clients=s1_clients,
                shared_body=shared_body,
                val_datasets=self.val_ds,
                test_datasets=self.test_ds,
                n_rounds=self.args.n_rounds,
                n_episodes=self.args.n_episodes,
                k_shot=k_shot,
                q_query=BEN_Q_QUERY,
                n_way=BEN_N_WAY,
                device=self.device,
                track_protos=(self.args.method == "FedProto"),
                val_every=self.args.val_every,
            )
            results[label] = result
        return results
