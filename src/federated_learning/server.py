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
import collections


def average_weights(weight_list: list[dict]) -> dict:
    """
    FedAvg aggregation. Simple mean over all client weight dicts.
    All clients share the same architecture (same keys), so no coverage
    issue here — but we guard against an empty list defensively.
    """
    if not weight_list:
        raise ValueError("average_weights received an empty list.")

    avg = {}
    for key in weight_list[0].keys():
        # Stack tensors from all clients and take the mean
        stacked = torch.stack([w[key].float() for w in weight_list], dim=0)
        avg[key] = stacked.mean(dim=0)

    return avg


def average_prototypes(proto_list: list[dict]) -> dict:
    """
    FedProto aggregation across clients with NON-IID class coverage.

    Each client dict is:  { global_class_idx (int) -> prototype (cpu tensor) }

    Not every client will have seen every class in their local episodes,
    so we must aggregate per-class only over the clients that actually
    contributed a prototype for that class — not divide by total clients.

    Returns: { global_class_idx -> aggregated prototype tensor }
    """
    if not proto_list:
        raise ValueError("average_prototypes received an empty list.")

    # Accumulate sum and count separately per class
    proto_sum = {}  # class_idx -> sum tensor
    proto_count = {}  # class_idx -> int  (how many clients contributed)

    for client_protos in proto_list:
        for class_idx, proto in client_protos.items():
            proto = proto.float().cpu()

            if class_idx not in proto_sum:
                proto_sum[class_idx] = torch.zeros_like(proto)
                proto_count[class_idx] = 0

            proto_sum[class_idx] += proto
            proto_count[class_idx] += 1

    # Divide each class sum by only the number of clients that saw it
    global_protos = {
        class_idx: proto_sum[class_idx] / proto_count[class_idx]
        for class_idx in proto_sum
    }

    return global_protos


def cpga_aggregation(
    s2_protos: dict,
    s1_protos: dict,
    bank_s2: dict = None,
    bank_s1: dict = None,
    eps: float = 1e-8,
):
    shared = sorted(set(s2_protos.keys()) & set(s1_protos.keys()))
    N = len(shared)

    # normalise to unit sphere
    P2 = torch.stack([s2_protos[c] for c in shared])
    P1 = torch.stack([s1_protos[c] for c in shared])
    z2 = F.normalize(P2, dim=-1)
    z1 = F.normalize(P1, dim=-1)

    # adjacency matrices
    A2 = torch.mm(z2, z2.T)
    A1 = torch.mm(z1, z1.T)

    # node consistency
    cons = torch.zeros(N)
    for i in range(N):  # BUG 1 FIX: range(N) not range N
        mask = torch.ones(N, dtype=torch.bool)
        mask[i] = False
        cons[i] = F.cosine_similarity(
            A2[i, mask].unsqueeze(0), A1[i, mask].unsqueeze(0)
        ).clamp(-1, 1)
    cons = (cons + 1) / 2

    # temporal confidence
    conf2 = compute_temporal_confidence(s2_protos, bank_s2, shared)
    conf1 = compute_temporal_confidence(s1_protos, bank_s1, shared)

    # sharpen weights
    alpha = 1.0 + (1.0 - cons)
    w2 = conf2.pow(alpha)
    w1 = conf1.pow(alpha)
    total = w2 + w1 + eps
    w2 = w2 / total
    w1 = w1 / total

    # shared classes
    global_protos = {}
    for i, cls in enumerate(shared):
        z_combined = w2[i] * z2[i] + w1[i] * z1[i]
        global_protos[cls] = F.normalize(z_combined, dim=-1)

    # optical-only classes
    for cls in set(s2_protos.keys()) - set(shared):  # BUG 2 FIX: removed redundant if
        global_protos[cls] = F.normalize(s2_protos[cls], dim=-1)

    # SAR-only classes                              # BUG 3 FIX: was using wrong variable cls
    for cls in set(s1_protos.keys()) - set(shared):
        global_protos[cls] = F.normalize(s1_protos[cls], dim=-1)

    return global_protos, cons, w2, w1


def compute_temporal_confidence(
    current_protos: dict,
    bank: dict,
    classes: list,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Confidence = cosine similarity between current prototype
    and historical mean from bank.
    No bank → neutral confidence = 0.5.
    """
    conf = torch.full((len(classes),), 0.5)

    if bank is None:
        return conf

    for i, cls in enumerate(classes):
        if cls not in bank or len(bank[cls]) == 0:
            continue

        history = torch.stack(list(bank[cls]))
        hist_mean = F.normalize(history.mean(0), dim=-1)
        current = F.normalize(current_protos[cls], dim=-1)

        sim = F.cosine_similarity(current.unsqueeze(0), hist_mean.unsqueeze(0)).item()

        conf[i] = (sim + 1) / 2  # [-1,1] → [0,1]

    return conf


class BaseServer:
    def __init__(self, s2_clients, s1_clients, fraction=1.0):
        self.s2_clients = s2_clients
        self.s1_clients = s1_clients
        self.clients = s2_clients + s1_clients
        self.fraction = fraction

    def train_round(self, n_episodes=100):
        num_to_select = max(1, int(self.fraction * len(self.clients)))
        selected_clients = random.sample(self.clients, num_to_select)

        self._broadcast(selected_clients)
        updates, round_losses, round_accs = self._collect_updates(
            selected_clients, n_episodes
        )
        self._aggregate(updates)

        avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0
        avg_acc = sum(round_accs) / len(round_accs) if round_accs else 0
        return avg_loss, avg_acc

    def _broadcast(self, clients):
        raise NotImplementedError

    def _collect_updates(self, clients, n_episodes):
        raise NotImplementedError

    def _aggregate(self, updates):
        raise NotImplementedError


class FedAvgServer(BaseServer):
    def __init__(self, shared_body: nn.Module, s2_clients, s1_clients, fraction=1.0):
        super().__init__(s2_clients + s1_clients, fraction)
        self.global_model = shared_body

    def _broadcast(self, clients):
        global_weights = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        for client in clients:
            client.set_weights(global_weights)

    def _collect_updates(self, clients, n_episodes):
        updates, losses, accs = [], [], []
        for client in clients:
            loss, acc = client.local_train(n_episodes=n_episodes)
            updates.append(client.get_weights())
            losses.append(loss)
            accs.append(acc)
        return updates, losses, accs

    def _aggregate(self, updates):
        # Use the average_weights helper function we defined earlier
        self.global_model.load_state_dict(average_weights(updates))


class FedProtoServer(BaseServer):
    def __init__(
        self,
        s2_clients,
        s1_clients,
        fraction=1.0,
        lam=0.1,
        aggregate_mode="cross_modal",
    ):
        """
        aggregate_mode:
            "cross_modal"  — average S1+S2 prototypes together (baseline, will collapse)
            "per_modality" — average within modality only (no cross-modal transfer)
        """
        super().__init__(s2_clients, s1_clients, fraction)
        self.lam = lam
        self.aggregate_mode = aggregate_mode
        self.global_protos = {}
        self.global_protos_s2 = {}
        self.global_protos_s1 = {}

    def _broadcast(self, clients):
        # FedProto doesn't share model weights
        pass

    def _get_protos_for_client(self, client):
        """Each client gets the right prototypes based on aggregate mode."""
        if self.aggregate_mode == "per_modality":
            if client.modality == "S2":
                return self.global_protos_s2 or None
            else:
                return self.global_protos_s1 or None
        # cross_modal: everyone gets the same (mixed) prototypes
        return self.global_protos or None

    def _collect_updates(self, clients, n_episodes):
        updates, losses, accs = [], [], []
        for client in clients:
            loss, acc = client.local_train(
                n_episodes=n_episodes,
                global_protos=self._get_protos_for_client(client),
                lam=self.lam,
            )
            protos = client.extract_prototypes()
            updates.append((protos, client.modality))
            losses.append(loss)
            accs.append(acc)
        return updates, losses, accs

    def _aggregate(self, updates):
        if self.aggregate_mode == "cross_modal":
            # Average everything together regardless of modality
            all_protos = [p for p, _ in updates]
            self.global_protos = average_prototypes(all_protos)

        elif self.aggregate_mode == "per_modality":
            s2_protos = [p for p, m in updates if m == "S2"]
            s1_protos = [p for p, m in updates if m == "S1"]
            if s2_protos:
                self.global_protos_s2 = average_prototypes(s2_protos)
            if s1_protos:
                self.global_protos_s1 = average_prototypes(s1_protos)


class FedProtoProjServer(BaseServer):
    def __init__(self, s2_clients, s1_clients, fraction=1.0, lam=0.1):
        super().__init__(s2_clients, s1_clients, fraction)
        self.lam = lam
        self.global_protos = {}
        self.global_proj_weights = None

    def _broadcast(self, clients):
        """FedAvg the projection head so both modalities project to same space."""
        if self.global_proj_weights is not None:
            for client in clients:
                client.set_projection_weights(self.global_proj_weights)

    def _collect_updates(self, clients, n_episodes):
        updates, losses, accs = [], [], []
        proj_weights = []

        for client in clients:
            loss, acc = client.local_train(
                n_episodes=n_episodes,
                global_protos=self.global_protos or None,
                lam=self.lam,
            )
            # Projected prototypes for prototype aggregation
            updates.append(client.extract_prototypes())
            # Projection head weights for FedAvg
            proj_weights.append(client.get_projection_weights())
            losses.append(loss)
            accs.append(acc)

        self._proj_weights = proj_weights
        return updates, losses, accs

    def _aggregate(self, updates):
        # Average prototypes in projected space (cross-modal, same space now)
        self.global_protos = average_prototypes(updates)
        # FedAvg the projection head weights
        self.global_proj_weights = average_weights(self._proj_weights)


class FedCMFSLServer(BaseServer):
    def __init__(self, s2_clients, s1_clients, fraction=1.0, lam=0.1, metric="Euclidean"):
        super().__init__(s2_clients, s1_clients, fraction)
        self.metric = metric

        # 1. Dynamically find ALL unique class IDs from all clients
        all_unique_classes = set()
        for c in self.clients:
            # This assumes your dataset object has a .classes or similar attribute
            # If not, you can use unique values from the sampler's valid_class_keys
            all_unique_classes.update(c.sampler.class_keys)

        # 2. Sort them to ensure consistent matrix indexing
        self.sorted_classes = sorted(list(all_unique_classes))
        self.num_classes = len(self.sorted_classes)

        # 3. Create the mapping: ClassID -> Matrix Row/Col Index
        self.class_to_idx = {cls_id: i for i, cls_id in enumerate(self.sorted_classes)}

        self.lam = lam
        self.global_D = torch.zeros((self.num_classes, self.num_classes))
        self.obs_mask = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.bool
        )

    def _broadcast(self, clients):
        # Our method doesn't share model weights
        pass

    def _collect_updates(self, clients, n_episodes):
        updates, losses, accs = [], [], []

        for client in clients:
            loss, acc = client.local_train(
                n_episodes=n_episodes,
                global_D=self.global_D,
                obs_mask=self.obs_mask,
                class_to_idx=self.class_to_idx,
                lam=self.lam,
                metric=self.metric,
            )
            local_D, local_classes = client.extract_distance_matrix(metric=self.metric)

            updates.append((local_D, local_classes))
            losses.append(loss)
            accs.append(acc)

        return updates, losses, accs

    def _aggregate(self, updates):
        """Combine heterogeneous local matrices into the global consensus."""
        # Ensure these are on CPU to match incoming .cpu() clones if you prefer,
        # or just move incoming data to match the server's device.
        new_global_D = torch.zeros((self.num_classes, self.num_classes))
        counts = torch.zeros((self.num_classes, self.num_classes))

        for local_D, local_classes in updates:
            # Move local_D to CPU to match new_global_D
            local_D_cpu = local_D.cpu()

            for i, class_i in enumerate(local_classes):
                idx_i = self.class_to_idx[class_i]
                for j, class_j in enumerate(local_classes):
                    idx_j = self.class_to_idx[class_j]

                    # Now both are on CPU
                    new_global_D[idx_i, idx_j] += local_D_cpu[i, j]
                    counts[idx_i, idx_j] += 1

        mask = counts > 0
        new_global_D[mask] = new_global_D[mask] / counts[mask]

        self.global_D = new_global_D
        self.obs_mask = mask
