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
from src.models.protonet import ProtoNet, SplitEncoder
from src.utils.episode_sampler import EpisodeSampler
from src.datasets.dataset_s1 import BigEarthNetS1Dataset
from src.datasets.dataset_s2 import BigEarthNetS2Dataset
import pickle, copy, time


def build_eval_encoders(shared_body, device, s2_clients=None, s1_clients=None):
    """
    Returns a dictionary mapping modality to a LIST of personalized models.
    """
    encoders = {"S2": [], "S1": []}

    if shared_body is not None:
        # Assuming you want to wrap them in ProtoNet
        if s2_clients:
            for client in s2_clients:
                enc = SplitEncoder(
                    in_channels=10, shared_body=copy.deepcopy(shared_body)
                )
                enc.load_state_dict(client.model.encoder.state_dict())
                encoders["S2"].append(ProtoNet(enc, feat_dim=512).to(device).eval())
        if s1_clients:
            for client in s1_clients:
                enc = SplitEncoder(
                    in_channels=2, shared_body=copy.deepcopy(shared_body)
                )
                enc.load_state_dict(client.model.encoder.state_dict())
                encoders["S1"].append(ProtoNet(enc, feat_dim=512).to(device).eval())
    else:
        # Pure deepcopy of the personalized models
        if s2_clients:
            encoders["S2"] = [
                copy.deepcopy(client.model).to(device).eval() for client in s2_clients
            ]
        if s1_clients:
            encoders["S1"] = [
                copy.deepcopy(client.model).to(device).eval() for client in s1_clients
            ]

    # Clean up empty lists
    return {k: v for k, v in encoders.items() if len(v) > 0}


import numpy as np
import torch


@torch.no_grad()
def evaluate_with_ci(
    eval_encoders: dict,  # Dict mapping modality to a LIST of models
    datasets: dict,  # Global unpartitioned datasets
    device,
    n_episodes: int = 600,
    k_shot: int = 1,
    q_query: int = 12,
    n_way: int = 5,
    seed: int = 0,
) -> dict:
    """
    Returns:
    {
        modality: {
            "mean": float,
            "ci": float,
            "str": string,
            "per_client": [
                {"client_idx": int, "mean": float, "ci": float, "str": string}, ...
            ]
        }
    }
    """
    results = {}

    for modality, models_list in eval_encoders.items():
        if modality not in datasets:
            continue

        dataset = datasets[modality]

        # Use your updated EpisodeSampler (the one with the valid_class_keys fix)
        sampler = EpisodeSampler(
            dataset,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            seed=seed,
            is_train=False,
        )

        # Trackers
        network_accs = []  # The average performance of the network per episode
        client_accs = [
            [] for _ in range(len(models_list))
        ]  # Tracks all 600 accs for EACH client

        for s_x, s_y, q_x, q_y, _ in sampler.episodes(n_episodes):
            s_x, s_y = s_x.to(device), s_y.to(device)
            q_x, q_y = q_x.to(device), q_y.to(device)

            episode_client_accs = []

            # Evaluate EVERY personalized model on this exact same episode
            for client_idx, model in enumerate(models_list):
                logits, _, _ = model(s_x, s_y, q_x, n_way=n_way)
                acc = (logits.argmax(1) == q_y).float().mean().item() * 100

                episode_client_accs.append(acc)
                client_accs[client_idx].append(acc)  # Store it for this specific client

            # The network's performance on this episode is the average of all clients
            network_accs.append(np.mean(episode_client_accs))

        # Calculate final CI across the 600 episodes for the ENTIRE network
        net_mean = float(np.mean(network_accs))
        net_ci = float(1.96 * np.std(network_accs) / np.sqrt(len(network_accs)))

        # Calculate final CI for EACH individual client
        per_client_results = []
        for client_idx, accs in enumerate(client_accs):
            c_mean = float(np.mean(accs))
            c_ci = float(1.96 * np.std(accs) / np.sqrt(len(accs)))
            per_client_results.append(
                {
                    "client_idx": client_idx,
                    "mean": c_mean,
                    "ci": c_ci,
                    "str": f"{c_mean:.2f} ± {c_ci:.2f}",
                }
            )

        results[modality] = {
            "mean": net_mean,
            "ci": net_ci,
            "str": f"{net_mean:.2f} ± {net_ci:.2f}",
            "per_client": per_client_results,
        }

    return results
