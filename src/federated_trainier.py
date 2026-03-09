
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from src.models.protonet import ProtoNet,  SplitEncoder, ResNet12
from src.utils.episode_sampler import EpisodeSampler
from src.datasets.dataset_s1 import BigEarthNetS1Dataset
from src.datasets.dataset_s2 import BigEarthNetS2Dataset
from src.evaluator import evaluate_with_ci, build_eval_encoders, extract_modal_prototypes, compute_inter_modal_distance
from src.utils.save import load_checkpoint, save_checkpoint, save_result
from config import RESULTS_DIR

PATIENCE_FEDERATED = 15 
def run_federated(
    label: str,
    server,
    s2_clients: list,
    s1_clients: list,
    val_datasets: dict,
    test_datasets: dict,
    shared_body=None,        # None for FedProto, nn.Module for FedAvg
    n_rounds: int = 50,
    n_episodes: int = 100,
    k_shot: int = 1,
    q_query: int = 12,
    n_way: int = 5,
    device: torch.device = None,
    track_protos: bool = False,
    val_every: int = 1,
) -> dict:

    device = device or torch.device("cpu")
    is_fedproto = shared_body is None  

    ckpt = load_checkpoint(label)
    start = 1
    best_acc = 0.0
    best_state = None
    no_improve = 0

    history = {
        "round": [], "loss": [],
        "val_S2": [], "val_S1": [], "val_avg": [],
        "val_S2_ci": [], "val_S1_ci": [],
        "proto_l2": [], "proto_cos": [],
        "proto_s2": {}, "proto_s1": {},
    }

    # ── Resume from checkpoint
    if ckpt is not None:
        start      = ckpt["round"] + 1
        history    = ckpt["history"]
        best_acc   = ckpt["best_acc"]
        best_state = ckpt["best_state"]
        if not is_fedproto:
            shared_body.load_state_dict(ckpt["model_state"])
        else:
            # Restore best client weights from checkpoint
            if best_state is not None:
                server.global_protos = best_state["global_protos"]
                for c in s2_clients:
                    c.model.load_state_dict(best_state["s2_client_states"][c.client_id])
                for c in s1_clients:
                    c.model.load_state_dict(best_state["s1_client_states"][c.client_id])

    # ── Training loop
    for r in range(start, n_rounds + 1):
        t0 = time.time()
        avg_loss = server.train_round(n_episodes=n_episodes)
        elapsed  = time.time() - t0

        history["round"].append(r)
        history["loss"].append(avg_loss)

        # ── Validation
        if r % val_every == 0:
            eval_enc = build_eval_encoders(shared_body, device, s2_clients, s1_clients)
            val_m    = evaluate_with_ci(
                eval_enc, val_datasets, device,
                n_episodes=200, k_shot=k_shot, q_query=q_query, n_way=n_way,
            )
            s2_m = val_m.get("S2", {"mean": 0, "ci": 0})
            s1_m = val_m.get("S1", {"mean": 0, "ci": 0})
            avg  = (s2_m["mean"] + s1_m["mean"]) / 2

            history["val_S2"].append(s2_m["mean"])
            history["val_S1"].append(s1_m["mean"])
            history["val_avg"].append(avg)
            history["val_S2_ci"].append(s2_m["ci"])
            history["val_S1_ci"].append(s1_m["ci"])

            if avg > best_acc:
                best_acc   = avg
                no_improve = 0
                # ── Save best state: FedAvg vs FedProto
                if not is_fedproto:
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in shared_body.state_dict().items()
                    }
                else:
                    best_state = {
                        "global_protos": server.global_protos,
                        "s2_client_states": {
                            c.client_id: {k: v.cpu().clone() for k, v in c.model.state_dict().items()}
                            for c in s2_clients
                        },
                        "s1_client_states": {
                            c.client_id: {k: v.cpu().clone() for k, v in c.model.state_dict().items()}
                            for c in s1_clients
                        },
                    }
            else:
                no_improve += val_every
        else:
            history["val_S2"].append(None)
            history["val_S1"].append(None)
            history["val_avg"].append(None)
            history["val_S2_ci"].append(None)
            history["val_S1_ci"].append(None)

        # ── Prototype tracking
        if track_protos:
            ps2  = extract_modal_prototypes(s2_clients, device)
            ps1  = extract_modal_prototypes(s1_clients, device)
            dist = compute_inter_modal_distance(ps2, ps1)
            history["proto_l2"].append(dist["avg_l2"])
            history["proto_cos"].append(dist["avg_cosine"])
            if r % 10 == 0 or r == n_rounds:
                history["proto_s2"][r] = ps2
                history["proto_s1"][r] = ps1
        else:
            history["proto_l2"].append(None)
            history["proto_cos"].append(None)

        # ── Print
        s2_val = history["val_S2"][-1]
        s1_val = history["val_S1"][-1]
        print(
            f"[{label}] Round {r:>3}/{n_rounds}  "
            f"loss={avg_loss:.4f}  "
            f"S2={f'{s2_val:.1f}%' if s2_val is not None else 'skip'}  "
            f"S1={f'{s1_val:.1f}%' if s1_val is not None else 'skip'}  "
            f"best={best_acc:.1f}%  ({elapsed:.0f}s)"
        )

        # ── Checkpoint
        save_checkpoint(
            label, r,
            model_state=(
                {k: v.cpu() for k, v in shared_body.state_dict().items()}
                if not is_fedproto else {}
            ),
            history=history,
            best_acc=best_acc,
            best_state=best_state,
        )

    # ── Final test with BEST model
    print(f"\n[{label}] Loading best model (val_avg={best_acc:.2f}%) for test eval")
    if not is_fedproto:
        shared_body.load_state_dict(best_state)
    else:
        server.global_protos = best_state["global_protos"]
        for c in s2_clients:
            c.model.load_state_dict(best_state["s2_client_states"][c.client_id])
        for c in s1_clients:
            c.model.load_state_dict(best_state["s1_client_states"][c.client_id])

    eval_enc = build_eval_encoders(shared_body, device, s2_clients, s1_clients)
    test_m   = evaluate_with_ci(
        eval_enc, test_datasets, device,
        n_episodes=600, k_shot=k_shot, q_query=q_query, n_way=n_way,
    )

    result = {
        "label": label, "history": history,
        "best_val": best_acc, "best_state": best_state, "test": test_m,
    }
    print(f"Test Results for {label}: {test_m}")
    save_result(label, result)
    return result

