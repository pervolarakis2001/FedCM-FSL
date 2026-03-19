import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import pandas as pd
import ast
import rasterio
from skimage.transform import resize
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import pickle, copy, time
import numpy as np
import torch
import torch.nn.functional as F

from src.models.protonet import ProtoNet, SplitEncoder
from src.utils.episode_sampler import EpisodeSampler
from src.datasets.dataset_s1 import BigEarthNetS1Dataset
from src.datasets.dataset_s2 import BigEarthNetS2Dataset


class BaseEpisodicClient:
    def __init__(
        self,
        client_id,
        dataset,
        model,
        device,
        n_way=5,
        k_shot=5,
        q_query=15,
        lr=1e-3,
        modality="S2",
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.n_way = n_way

        # Episodic sampler is now standard for all clients
        self.sampler = EpisodeSampler(
            dataset, n_way=n_way, k_shot=k_shot, q_query=q_query, is_train=True
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=5e-4
        )

        self.modality = modality
        self._accumulated_protos = {}  # Initialize here once for all subclasses

    def get_weights(self):
        """Standard for FedAvg and others that share model parameters."""
        if hasattr(self.model.encoder, "get_shared_weights"):
            return self.model.encoder.get_shared_weights()
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, global_weights):
        """Dynamically loads weights based on the encoder type."""
        if hasattr(self.model.encoder, "load_shared_weights"):
            self.model.encoder.load_shared_weights(global_weights)
        else:
            self.model.encoder.load_state_dict(global_weights)

    def local_train(self, n_episodes=100, **kwargs):
        self.model.train()
        losses, accs = [], []
        proto_accumulator = {}  # cls_idx -> list of proto tensors
        proto_counts = {}

        for s_x, s_y, q_x, q_y, true_classes in self.sampler.episodes(n_episodes):
            s_x, s_y = s_x.to(self.device), s_y.to(self.device)
            q_x, q_y = q_x.to(self.device), q_y.to(self.device)

            self.optimizer.zero_grad()
            loss, acc, local_protos = self._compute_loss(
                s_x, s_y, q_x, q_y, true_classes=true_classes, **kwargs
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            # Accumulate running average
            if local_protos is not None:
                for i, true_class in enumerate(true_classes):
                    cls = true_class.item()
                    proto = local_protos[i].detach().cpu()
                    if cls not in proto_accumulator:
                        proto_accumulator[cls] = proto
                        proto_counts[cls] = 1
                    else:
                        n = proto_counts[cls]
                        proto_accumulator[cls] = (
                            proto_accumulator[cls] * n + proto
                        ) / (n + 1)
                        proto_counts[cls] += 1

            losses.append(loss.item())
            accs.append(acc)

        self._accumulated_protos = proto_accumulator  # Save for extract_prototypes
        return np.mean(losses), np.mean(accs)

    def _compute_loss(self, s_x, s_y, q_x, q_y, **kwargs):
        raise NotImplementedError("Subclasses must define how to compute the loss.")

    def extract_prototypes(self):
        return self._accumulated_protos


class FedAvgClient(BaseEpisodicClient):
    def _compute_loss(self, s_x, s_y, q_x, q_y, true_classes=None, **kwargs):
        # Standard FedAvg ignores prototype regularization
        loss, acc, _ = self.model.train_episode(
            s_x,
            s_y,
            q_x,
            q_y,
            self.n_way,
            true_classes=true_classes,
            global_protos=None,
            lam1=0.0,
            lam2=0.0,
            method="base",
            temperature=0.0,
        )
        return loss, acc, None


class LocalOnlyClient(BaseEpisodicClient):
    """
    Baseline client that trains purely on local data.
    No prototype sharing, no weight aggregation, no server communication.
    """

    def _compute_loss(self, s_x, s_y, q_x, q_y, true_classes=None, **kwargs):
        loss, acc, _ = self.model.train_episode(
            s_x,
            s_y,
            q_x,
            q_y,
            self.n_way,
            true_classes=true_classes,
            global_protos=None,
            lam1=0.0,
            lam2=0.0,
            method="base",
            temperature=0.0,
        )
        return loss, acc, None  # No protos to accumulate


class FedProtoClient(BaseEpisodicClient):
    def local_train(self, n_episodes=100, global_protos=None, lam=0.1, **kwargs):
        return super().local_train(
            n_episodes, global_protos=global_protos, lam=lam, **kwargs
        )

    def _compute_loss(
        self,
        s_x,
        s_y,
        q_x,
        q_y,
        true_classes=None,
        global_protos=None,
        lam=0.1,
        **kwargs,
    ):
        loss, acc, local_protos = self.model.train_episode(
            s_x,
            s_y,
            q_x,
            q_y,
            self.n_way,
            true_classes=true_classes,
            global_protos=global_protos,
            lam1=lam,
            lam2=0.0,
            method="fed_proto",
            temperature=0.0,
        )
        return loss, acc, local_protos


class FedCMFSLClient(BaseEpisodicClient):
    def local_train(
        self,
        n_episodes=100,
        global_protos=None,
        lam1=0.1,
        lam2=0.1,
        temperature=0.07,
        **kwargs,
    ):
        return super().local_train(
            n_episodes,
            global_protos=global_protos,
            lam1=lam1,
            lam2=lam2,
            temperature=temperature,
            **kwargs,
        )

    def _compute_loss(
        self,
        s_x,
        s_y,
        q_x,
        q_y,
        true_classes=None,
        global_protos=None,
        lam1=0.1,
        lam2=0.1,
        temperature=0.07,
        **kwargs,
    ):
        # Changed method to "proposed" and correctly passed lam1, lam2, and temperature
        loss, acc, local_protos = self.model.train_episode(
            s_x,
            s_y,
            q_x,
            q_y,
            self.n_way,
            true_classes=true_classes,
            global_protos=global_protos,
            lam1=lam1,
            lam2=lam2,
            method="ours",
            temperature=temperature,
        )
        return loss, acc, local_protos


def min_samples_per_class(df: pd.DataFrame, min_required: int = None) -> int:
    counts = df.groupby("primary_label").size()
    if min_required is not None:
        # Return count of viable classes, not the minimum
        return (counts >= min_required).sum()
    return counts.min()


def build_clients(
    partitions,
    s2_root,
    s1_root,
    s2_encoder,
    s1_encoder,
    ClientClass,
    device,
    split_encoder=False,
    n_way=5,
    k_shot=1,
    q_query=15,
    lr=1e-3,
):
    min_required = k_shot + q_query
    s2_clients, s1_clients = [], []
    for p in partitions:
        clean_df = filter_sparse_classes(p.df, min_samples=k_shot + q_query)
        if clean_df["primary_label"].nunique() < n_way:
            print(f"  ✗ Skipping client {p.client_id}")
            continue
        if p.has_s2:
            if split_encoder:
                enc = SplitEncoder(
                    in_channels=10, shared_body=copy.deepcopy(s2_encoder)
                )
            else:
                enc = copy.deepcopy(s2_encoder)
            model = ProtoNet(enc, feat_dim=512)
            s2_clients.append(
                ClientClass(
                    client_id=f"client{p.client_id}_S2",
                    dataset=BigEarthNetS2Dataset(clean_df, s2_root),
                    model=model,
                    device=device,
                    n_way=n_way,
                    k_shot=k_shot,
                    q_query=q_query,
                    lr=lr,
                    modality="S2",
                )
            )
        if p.has_s1:
            if split_encoder:
                enc = SplitEncoder(in_channels=2, shared_body=copy.deepcopy(s1_encoder))
            else:
                enc = copy.deepcopy(s1_encoder)
            model = ProtoNet(enc, feat_dim=512)
            s1_clients.append(
                ClientClass(
                    client_id=f"client{p.client_id}_S1",
                    dataset=BigEarthNetS1Dataset(clean_df, s1_root),
                    model=model,
                    device=device,
                    n_way=n_way,
                    k_shot=k_shot,
                    q_query=q_query,
                    lr=lr,
                    modality="S1",
                )
            )
    print(f"Built {len(s2_clients)} S2 clients, {len(s1_clients)} S1 clients")
    return s2_clients, s1_clients


def filter_sparse_classes(
    df: pd.DataFrame,
    label_col: str = "primary_label",
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    Removes classes that have fewer than `min_samples` rows.

    Parameters
    ----------
    df          : DataFrame to filter (a client's partition).
    label_col   : Column holding the class label.
    min_samples : Minimum number of samples a class must have to be kept.

    Returns
    -------
    Filtered DataFrame with sparse classes removed.
    """
    counts = df[label_col].value_counts()
    valid_classes = counts[counts >= min_samples].index
    dropped = counts[counts < min_samples].index.tolist()

    if dropped:
        print(
            f"  Dropping {len(dropped)} sparse class(es) "
            f"(< {min_samples} samples): {dropped}"
        )

    return df[df[label_col].isin(valid_classes)].reset_index(drop=True)
