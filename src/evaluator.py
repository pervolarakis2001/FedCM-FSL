
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
from src.models.protonet import ProtoNet,  SplitEncoder
from src.utils.episode_sampler import EpisodeSampler
from src.datasets.dataset_s1 import BigEarthNetS1Dataset
from src.datasets.dataset_s2 import BigEarthNetS2Dataset
import  pickle, copy, time

def build_eval_encoders(shared_body, device, s2_clients=None, s1_clients=None):
    encoders = {}
    if shared_body is not None:
        if s2_clients:
            enc = SplitEncoder(in_channels=10, shared_body=copy.deepcopy(shared_body))
            enc.load_state_dict(s2_clients[0].model.encoder.state_dict())
            encoders["S2"] = ProtoNet(enc, feat_dim=512).to(device).eval()
        if s1_clients:
            enc = SplitEncoder(in_channels=2, shared_body=copy.deepcopy(shared_body))
            enc.load_state_dict(s1_clients[0].model.encoder.state_dict())
            encoders["S1"] = ProtoNet(enc, feat_dim=512).to(device).eval()
    else:
        if s2_clients:
            encoders["S2"] = copy.deepcopy(s2_clients[0].model).to(device).eval()
        if s1_clients:
            encoders["S1"] = copy.deepcopy(s1_clients[0].model).to(device).eval()
    return encoders


@torch.no_grad()
def evaluate_with_ci(
    eval_encoders: dict,
    datasets: dict,
    device,
    n_episodes: int = 600,
    k_shot: int = 1,
    q_query: int = 12,
    n_way: int = 5,
    seed: int = 0,
) -> dict:
    """
    Returns {modality: {"mean": float, "ci": float}} over n_episodes.
    Use n_episodes=200 for val, 600 for final test.
    """
    results = {}
    for modality, model in eval_encoders.items():
        if modality not in datasets:
            continue
        model.eval()
        dataset = datasets[modality]
        sampler = EpisodeSampler(
            dataset, n_way=n_way, k_shot=k_shot, q_query=q_query, seed=seed, is_train=False
        )
        accs = []
        for s_x, s_y, q_x, q_y, _ in sampler.episodes(n_episodes):
            s_x, s_y = s_x.to(device), s_y.to(device)
            q_x, q_y = q_x.to(device), q_y.to(device)
            logits, _ = model(s_x, s_y, q_x, n_way=n_way)
            acc = (logits.argmax(1) == q_y).float().mean().item() * 100
            accs.append(acc)

        mean = float(np.mean(accs))
        ci = float(1.96 * np.std(accs) / np.sqrt(len(accs)))
        results[modality] = {"mean": mean, "ci": ci, "str": f"{mean:.2f} ± {ci:.2f}"}
    return results


@torch.no_grad()
def extract_modal_prototypes(clients: list, device) -> dict:
    """
    Collect per-class prototypes from all clients of the same modality.
    Returns {class_idx: mean_prototype_tensor (cpu)}.
    """
    proto_sum = {}
    proto_count = {}
    for client in clients:
        client.model.eval()
        for cls_idx, indices in client.dataset.class_images.items():
            all_feats = []
            for i in range(0, len(indices), 32):
                batch = indices[i : i + 32]
                imgs = torch.stack([client.dataset[j][0] for j in batch]).to(device)
                feats = client.model.encode(imgs).cpu()
                all_feats.append(feats)
            proto = torch.cat(all_feats).mean(0)

            if cls_idx not in proto_sum:
                proto_sum[cls_idx] = torch.zeros_like(proto)
                proto_count[cls_idx] = 0
            proto_sum[cls_idx] += proto
            proto_count[cls_idx] += 1

    return {k: proto_sum[k] / proto_count[k] for k in proto_sum}


def compute_inter_modal_distance(protos_s2: dict, protos_s1: dict) -> dict:
    """
    Per-class L2 and cosine distance between S2 and S1 prototypes.
    Returns {"avg_l2", "avg_cosine", "per_class": {cls: {"l2", "cosine"}}}.
    """
    common = set(protos_s2.keys()) & set(protos_s1.keys())
    per_class = {}
    for cls in common:
        p2 = protos_s2[cls].float()
        p1 = protos_s1[cls].float()
        l2 = torch.norm(p2 - p1).item()
        cos = F.cosine_similarity(p2.unsqueeze(0), p1.unsqueeze(0)).item()
        per_class[cls] = {"l2": l2, "cosine": cos}

    avg_l2 = float(np.mean([v["l2"] for v in per_class.values()]))
    avg_cos = float(np.mean([v["cosine"] for v in per_class.values()]))
    return {"avg_l2": avg_l2, "avg_cosine": avg_cos, "per_class": per_class}
