
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import numpy as np
import torch
import torch.nn.functional as F
from src.models.protonet import ProtoNet,  SplitEncoder, ResNet12
from src.utils.episode_sampler import EpisodeSampler
from src.datasets.dataset_s1 import BigEarthNetS1Dataset
from src.datasets.dataset_s2 import BigEarthNetS2Dataset
from src.evaluator import evaluate_with_ci
from src.utils.save import load_checkpoint, save_checkpoint, save_result
from config import RESULTS_DIR
PATIENCE_CENTRAL  = 200 

def run_centralized(
    label: str,
    train_s2,  # BigEarthNetS2Dataset — full training set
    train_s1,  # BigEarthNetS1Dataset
    val_datasets: dict,
    test_datasets: dict,
    n_episodes: int = 5000,  # 50 rounds × 100 eps equivalent
    k_shot: int = 1,
    q_query: int = 12,
    n_way: int = 5,
    lr: float = 1e-3,
    device: torch.device = None,
    val_every: int = 100,  # validate every N episodes
) -> dict:
    """
    Single-machine ProtoNet — no federation, no SplitEncoder.
    Upper bound for all federated experiments.
    """
    device = device or torch.device("cpu")

    # Plain encoders — no stem needed
    model_s2 = ProtoNet(ResNet12(in_channels=10), feat_dim=512).to(device)
    model_s1 = ProtoNet(ResNet12(in_channels=2), feat_dim=512).to(device)
    opt_s2 = torch.optim.Adam(model_s2.parameters(), lr=lr, weight_decay=5e-4)
    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=lr, weight_decay=5e-4)

    ckpt = load_checkpoint(label)
    start_ep = 0
    best_acc = 0.0
    best_s2 = None
    best_s1 = None

    history = {
        "episode": [],
        "loss_S2": [],
        "loss_S1": [],
        "val_S2": [],
        "val_S1": [],
        "val_avg": [],
    }

    if ckpt is not None:
        start_ep = ckpt["round"]
        history = ckpt["history"]
        best_acc = ckpt["best_acc"]
        model_s2.load_state_dict(ckpt["model_state"]["S2"])
        model_s1.load_state_dict(ckpt["model_state"]["S1"])
        best_s2 = ckpt["best_state"]["S2"]
        best_s1 = ckpt["best_state"]["S1"]

    sampler_s2 = EpisodeSampler(
        train_s2, n_way=n_way, k_shot=k_shot, q_query=q_query, seed=42, is_train=True
    )
    sampler_s1 = EpisodeSampler(
        train_s1, n_way=n_way, k_shot=k_shot, q_query=q_query, seed=42, is_train=True
    )

    ep_losses_s2, ep_losses_s1 = [], []
    no_improve = 0 
    
    for ep, ((sx2, sy2, qx2, qy2, _), (sx1, sy1, qx1, qy1, _)) in enumerate(
        zip(
            sampler_s2.episodes(n_episodes - start_ep),
            sampler_s1.episodes(n_episodes - start_ep),
        ),
        start=start_ep,
    ):
        model_s2.train()
        model_s1.train()

        # S2 step
        sx2, sy2 = sx2.to(device), sy2.to(device)
        qx2, qy2 = qx2.to(device), qy2.to(device)
        opt_s2.zero_grad()
        loss_s2, _, _ = model_s2.train_episode(sx2, sy2, qx2, qy2, n_way)
        loss_s2.backward()
        opt_s2.step()

        # S1 step
        sx1, sy1 = sx1.to(device), sy1.to(device)
        qx1, qy1 = qx1.to(device), qy1.to(device)
        opt_s1.zero_grad()
        loss_s1, _, _ = model_s1.train_episode(sx1, sy1, qx1, qy1, n_way)
        loss_s1.backward()
        opt_s1.step()

        ep_losses_s2.append(loss_s2.item())
        ep_losses_s1.append(loss_s1.item())

        # Validate and log every val_every episodes
        if (ep + 1) % val_every == 0:
            eval_enc = {"S2": model_s2, "S1": model_s1}
            val_m = evaluate_with_ci(
                eval_enc,
                val_datasets,
                device,
                n_episodes=200,
                k_shot=k_shot,
                q_query=q_query,
                n_way=n_way,
            )
            s2_m = val_m.get("S2", {"mean": 0})
            s1_m = val_m.get("S1", {"mean": 0})
            avg = (s2_m["mean"] + s1_m["mean"]) / 2

            history["episode"].append(ep + 1)
            history["loss_S2"].append(float(np.mean(ep_losses_s2[-val_every:])))
            history["loss_S1"].append(float(np.mean(ep_losses_s1[-val_every:])))
            history["val_S2"].append(s2_m["mean"])
            history["val_S1"].append(s1_m["mean"])
            history["val_avg"].append(avg)

            if avg > best_acc:
                best_acc = avg
                no_improve = 0
                best_s2 = {k: v.cpu().clone() for k, v in model_s2.state_dict().items()}
                best_s1 = {k: v.cpu().clone() for k, v in model_s1.state_dict().items()}
            else:
                no_improve += val_every 

            if no_improve >= PATIENCE_CENTRAL:
                print(f"Early stop at ep {ep+1} — "
                      f"no improvement for {PATIENCE_CENTRAL} episodes  "
                      f"(best={best_acc:.2f}%)")
                break

            print(
                f"[{label}] Ep {ep+1:>5}/{n_episodes}  "
                f"S2={s2_m['mean']:.1f}%  S1={s1_m['mean']:.1f}%  "
                f"best={best_acc:.1f}%"
            )

            save_checkpoint(
                label,
                ep + 1,
                model_state={"S2": best_s2, "S1": best_s1},
                history=history,
                best_acc=best_acc,
                best_state={"S2": best_s2, "S1": best_s1},
            )

    # Final test with best model
    model_s2.load_state_dict(best_s2)
    model_s1.load_state_dict(best_s1)
    test_m = evaluate_with_ci(
        {"S2": model_s2, "S1": model_s1},
        test_datasets,
        device,
        n_episodes=600,
        k_shot=k_shot,
        q_query=q_query,
        n_way=n_way,
    )

    result = {
        "label": label,
        "history": history,
        "best_val": best_acc,
        "test": test_m,
    }
    print(f"Test Results for {label}: {test_m}")
    save_result(label, result, results_dir=RESULTS_DIR)
    return result
