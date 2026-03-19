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
    def __init__(self, clients, fraction=1.0):
        self.clients = clients
        self.fraction = fraction  # Percentage of clients to select each round

    def train_round(self, n_episodes=100):
        """The universal federated learning loop."""
        # Select a subset of clients
        num_to_select = max(1, int(self.fraction * len(self.clients)))
        selected_clients = random.sample(self.clients, num_to_select)

        # Broadcast global state (implemented by subclasses)
        self._broadcast(selected_clients)

        # Train locally and collect updates (implemented by subclasses)
        updates, round_losses, round_accs = self._collect_updates(
            selected_clients, n_episodes
        )

        # Aggregate updates (implemented by subclasses)
        self._aggregate(updates)

        avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0
        return avg_loss

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
    def __init__(self, s2_clients, s1_clients, fraction=1.0, lam=0.1):
        super().__init__(s2_clients + s1_clients, fraction)
        self.lam = lam
        self.global_protos = {}

    def _broadcast(self, clients):
        # FedProto doesn't set weights, it just passes the prototypes during training.
        pass

    def _collect_updates(self, clients, n_episodes):
        updates, losses, accs = [], [], []
        for client in clients:
            # Pass global prototypes into the training loop for regularization
            loss, acc = client.local_train(
                n_episodes=n_episodes,
                global_protos=self.global_protos if self.global_protos else None,
                lam=self.lam,
            )
            # Collect newly generated prototypes
            updates.append(client.extract_prototypes())
            losses.append(loss)
            accs.append(acc)
        return updates, losses, accs

    def _aggregate(self, updates):
        # Use the average_prototypes helper function we defined earlier
        self.global_protos = average_prototypes(updates)


class FedCMFSLServer(BaseServer):
    def __init__(
        self,
        s2_clients,
        s1_clients,
        fraction=1.0,
        lam1=0.1,
        lam2=0.1,
        temperature=0.07,
        bank_max_history=10,
    ):
        super().__init__(s2_clients + s1_clients, fraction)
        self.s2_clients = s2_clients
        self.s1_clients = s1_clients
        self.lam1 = lam1
        self.lam2 = lam2
        self.temperature = temperature

        self.global_protos = {}

        # ── BANK lives here ───────────────────────────────────────────────
        # bank_s2[cls] = deque of past normalized optical prototypes
        # bank_s1[cls] = deque of past normalized SAR prototypes
        self.bank_s2 = {}
        self.bank_s1 = {}
        self.bank_max_history = bank_max_history  # sliding window size

    # ── helpers ──────────────────────────────────────────────────────────

    def _update_bank(self, bank: dict, protos: dict):
        """Push current round's normalized prototypes into the sliding window."""
        for cls, proto in protos.items():
            if cls not in bank:
                bank[cls] = collections.deque(maxlen=self.bank_max_history)
            bank[cls].append(F.normalize(proto.detach(), dim=-1))

    def _split_updates(self, updates, clients):
        """Separate prototype dicts by modality."""
        s2_updates = [u for u, c in zip(updates, clients) if c in self.s2_clients]
        s1_updates = [u for u, c in zip(updates, clients) if c in self.s1_clients]
        return s2_updates, s1_updates

    # ── BaseServer interface ──────────────────────────────────────────────

    def _broadcast(self, clients):
        pass  # global_protos injected at train time, nothing to push

    def _collect_updates(self, clients, n_episodes):
        updates, losses, accs = [], [], []
        for client in clients:
            loss, acc = client.local_train(
                n_episodes=n_episodes,
                global_protos=self.global_protos if self.global_protos else None,
                lam1=self.lam1,
                lam2=self.lam2,
                temperature=self.temperature,
            )
            updates.append(client.extract_prototypes())
            losses.append(loss)
            accs.append(acc)
        return updates, losses, accs

    def _aggregate(self, updates):
        raise NotImplementedError("Use train_round override")

    def train_round(self, n_episodes=100):
        """Override to pass selected_clients into _aggregate."""
        num_to_select = max(1, int(self.fraction * len(self.clients)))
        selected = random.sample(self.clients, num_to_select)

        self._broadcast(selected)
        updates, losses, accs = self._collect_updates(selected, n_episodes)
        self._aggregate_cpga(updates, selected)

        return sum(losses) / len(losses) if losses else 0.0

    def _aggregate_cpga(self, updates, selected_clients):
        # ── 1. split updates by modality ─────────────────────────────────
        s2_updates, s1_updates = self._split_updates(updates, selected_clients)

        # ── 2. intra-modal mean ───────────────────────────────────────────
        s2_protos = average_prototypes(s2_updates)  # {cls: tensor}
        s1_protos = average_prototypes(s1_updates)

        if not s2_protos or not s1_protos:
            # one modality completely absent this round — skip bank update
            return

        # ── 3. CPGA with current bank
        global_protos, cons, w2, w1 = cpga_aggregation(
            s2_protos,
            s1_protos,
            bank_s2=self.bank_s2,
            bank_s1=self.bank_s1,
        )
        self.global_protos = global_protos

        self._update_bank(self.bank_s2, s2_protos)
        self._update_bank(self.bank_s1, s1_protos)
