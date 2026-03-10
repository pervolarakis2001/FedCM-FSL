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


class BaseServer:
    def __init__(self, clients, fraction=1.0):
        self.clients = clients
        self.fraction = fraction  # Percentage of clients to select each round

    def train_round(self, n_episodes=100):
        """The universal federated learning loop."""
        num_to_select = max(1, int(self.fraction * len(self.clients)))
        selected_clients = random.sample(self.clients, num_to_select)

        self._broadcast(selected_clients)

        # Train locally and collect updates
        updates, round_losses, round_accs = self._collect_updates(
            selected_clients, n_episodes
        )

        # Aggregate updates
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
