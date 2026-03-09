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

from src.models.protonet import ProtoNet,  SplitEncoder
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
        self.model = model  # kept on CPU; moved to device only during local_train
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
        """The base training loop. Subclasses can override or pass specific kwargs."""
        self.model.to(self.device)
        self.model.train()
        losses, accs = [], []

        for s_x, s_y, q_x, q_y, true_classes in self.sampler.episodes(n_episodes):
            s_x, s_y = s_x.to(self.device), s_y.to(self.device)
            q_x, q_y = q_x.to(self.device), q_y.to(self.device)

            self.optimizer.zero_grad()

            # We defer the actual forward/loss logic to a method that subclasses can override
            loss, acc = self._compute_loss(
                s_x, s_y, q_x, q_y, true_classes=true_classes, **kwargs
            )

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            accs.append(acc)

        self.model.cpu()
        torch.cuda.empty_cache()
        return np.mean(losses), np.mean(accs)

    def _compute_loss(self, s_x, s_y, q_x, q_y, **kwargs):
        raise NotImplementedError("Subclasses must define how to compute the loss.")


class FedAvgClient(BaseEpisodicClient):
    def _compute_loss(self, s_x, s_y, q_x, q_y, true_classes=None, **kwargs):
        # Standard FedAvg ignores prototype regularization (lam=0.0)
        loss, acc, _ = self.model.train_episode(
            s_x, s_y, q_x, q_y, self.n_way, 
            true_classes=true_classes, global_protos=None, lam=0.0
        )
        return loss, acc
    
class FedProtoClient(BaseEpisodicClient):
    def _compute_loss(self, s_x, s_y, q_x, q_y, true_classes=None, global_protos=None, lam=0.1, **kwargs):
        # FedProto utilizes the global prototypes and lambda
        loss, acc, _ = self.model.train_episode(
            s_x, s_y, q_x, q_y, self.n_way,
            true_classes=true_classes, global_protos=global_protos, lam=lam
        )
        return loss, acc

    @torch.no_grad()
    def extract_prototypes(self):
        """Extracts class prototypes. Run this after local_train."""
        self.model.to(self.device)
        self.model.eval()
        prototypes = {}
        for class_idx, indices in self.dataset.class_images.items():
            all_features = []
            for i in range(0, len(indices), 32):
                batch_indices = indices[i:i + 32]
                imgs = torch.stack([self.dataset[idx][0] for idx in batch_indices]).to(self.device)
                features = self.model.encode(imgs)
                all_features.append(features)

            all_features = torch.cat(all_features, dim=0)
            prototypes[class_idx] = all_features.mean(dim=0).cpu()

        self.model.cpu()
        torch.cuda.empty_cache()
        return prototypes
    

def min_samples_per_class(df: pd.DataFrame, min_required: int = None) -> int:
    counts = df.groupby("primary_label").size()
    if min_required is not None:
        # Return count of viable classes, not the minimum
        return (counts >= min_required).sum()
    return counts.min()

def filter_sparse_classes(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Drop classes that don't have enough samples to fill one episode."""
    counts = df.groupby("primary_label").size()
    viable_classes = counts[counts >= min_samples].index
    dropped = set(counts.index) - set(viable_classes)
    if dropped:
        print(f"  ⚠ Dropping {len(dropped)} sparse classes: {dropped}")
    return df[df["primary_label"].isin(viable_classes)].reset_index(drop=True)
    
    
def build_clients(partitions, s2_root, s1_root, s2_encoder, s1_encoder,
                  ClientClass, device, split_encoder=False,
                  n_way=5, k_shot=1, q_query=15, lr=1e-3):
    min_required = k_shot + q_query
    s2_clients, s1_clients = [], []
    for p in partitions:
        clean_df = filter_sparse_classes(p.df, min_samples=min_required)
        if clean_df["primary_label"].nunique() < n_way:
            print(f" Skipping client {p.client_id}")
            continue
        if p.has_s2:
            if split_encoder:
                enc = SplitEncoder(in_channels=10, shared_body=copy.deepcopy(s2_encoder))
            else:
                enc = copy.deepcopy(s2_encoder)  
            model = ProtoNet(enc, feat_dim=512)
            s2_clients.append(ClientClass(
                client_id=f"client{p.client_id}_S2",
                dataset=BigEarthNetS2Dataset(clean_df, s2_root),
                model=model, device=device,
                n_way=n_way, k_shot=k_shot, q_query=q_query, lr=lr, modality="S2",
            ))
        if p.has_s1:
            if split_encoder:
                enc = SplitEncoder(in_channels=2, shared_body=copy.deepcopy(s1_encoder))
            else:
                enc = copy.deepcopy(s1_encoder) 
            model = ProtoNet(enc, feat_dim=512)
            s1_clients.append(ClientClass(
                client_id=f"client{p.client_id}_S1",
                dataset=BigEarthNetS1Dataset(clean_df, s1_root),
                model=model, device=device,
                n_way=n_way, k_shot=k_shot, q_query=q_query, lr=lr, modality="S1",
            ))
    print(f"Built {len(s2_clients)} S2 clients, {len(s1_clients)} S1 clients")
    return s2_clients, s1_clients