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
from src.models.protonet import ProtoNet, SplitEncoder, ResNet12
from src.utils.episode_sampler import EpisodeSampler
from src.datasets.dataset_s1 import BigEarthNetS1Dataset
from src.datasets.dataset_s2 import BigEarthNetS2Dataset
from src.evaluator import (
    evaluate_with_ci,
    build_eval_encoders,
)
from src.utils.save import load_checkpoint, save_checkpoint, save_result
from config import RESULTS_DIR

PATIENCE_FEDERATED = 100
from scipy.stats import spearmanr


def _compute_analysis_metrics(server, s2_clients, s1_clients, device):
    """Compute all collapse/structural metrics for current round."""
    margins = {"S2": 0, "S1": 0}
    norms = {"S2": 0, "S1": 0}
    rank_corr = {}
    proj_corr = {}
    distance_matrices = {"S2": {}, "S1": {}}
    per_pair = {"S2": {}, "S1": {}}

    for modality, clients in [("S2", s2_clients), ("S1", s1_clients)]:
        if not clients:
            continue

        mod_margins = []
        mod_norms = []
        mod_rank_corrs = []
        mod_proj_corrs = []

        for client in clients:
            loader = client._get_full_dataloader()

            # Distance matrix
            D, classes = client.model.get_local_distance_matrix(loader, device)
            n = D.shape[0]

            # Store for heatmap snapshots
            distance_matrices[modality][client.client_id] = {
                "D": D.cpu().clone(),
                "classes": classes,
            }

            # Inter-class margin
            if n > 1:
                mask = ~torch.eye(n, dtype=torch.bool)
                mod_margins.append(D[mask].mean().item())

            # Per-pair distances
            pair_dists = {}
            for i in range(n):
                for j in range(i + 1, n):
                    pair_key = (classes[i], classes[j])
                    pair_dists[pair_key] = D[i, j].item()
            per_pair[modality][client.client_id] = pair_dists

            # Proto norms
            protos = client.model.get_native_prototypes(loader, device)
            proto_norm = torch.stack(list(protos.values())).norm(dim=1).mean().item()
            mod_norms.append(proto_norm)

            # Rank correlation with global D (RPT only)
            if hasattr(server, "global_D") and server.global_D is not None:
                gi = [
                    server.class_to_idx[c] for c in classes if c in server.class_to_idx
                ]
                if len(gi) >= 2:

                    gi_t = torch.tensor(gi)
                    D_global_sub = server.global_D[gi_t][:, gi_t]

                    idx = torch.triu_indices(len(gi), len(gi), offset=1)
                    local_vals = D[: len(gi), : len(gi)][idx[0], idx[1]].cpu().numpy()
                    global_vals = D_global_sub[idx[0], idx[1]].cpu().numpy()

                    if len(local_vals) > 1:
                        rho, _ = spearmanr(local_vals, global_vals)
                        if not np.isnan(rho):
                            mod_rank_corrs.append(rho)

            # Projection analysis: native vs projected distance structure
            if hasattr(client.model, "use_projection") and client.model.use_projection:
                proj_protos = client.model.get_projected_prototypes(loader, device)
                proj_list = [proj_protos[c] for c in classes if c in proj_protos]
                if len(proj_list) >= 2:
                    proj_stack = torch.stack(proj_list)
                    D_proj = torch.cdist(
                        proj_stack.unsqueeze(0), proj_stack.unsqueeze(0)
                    ).squeeze(0)
                    n_proj = min(D.shape[0], D_proj.shape[0])
                    idx = torch.triu_indices(n_proj, n_proj, offset=1)
                    native_vals = D[:n_proj, :n_proj][idx[0], idx[1]].cpu().numpy()
                    proj_vals = D_proj[idx[0], idx[1]].detach().cpu().numpy()
                    if len(native_vals) > 1:
                        rho, _ = spearmanr(native_vals, proj_vals)
                        if not np.isnan(rho):
                            mod_proj_corrs.append(rho)

        margins[modality] = np.mean(mod_margins) if mod_margins else 0
        norms[modality] = np.mean(mod_norms) if mod_norms else 0
        if mod_rank_corrs:
            rank_corr[modality] = np.mean(mod_rank_corrs)
        if mod_proj_corrs:
            proj_corr[modality] = np.mean(mod_proj_corrs)

    # Add global D snapshot if available
    if hasattr(server, "global_D") and server.global_D is not None:
        distance_matrices["global"] = {
            "D": server.global_D.cpu().clone(),
            "class_to_idx": dict(server.class_to_idx),
        }

    return {
        "margins": margins,
        "norms": norms,
        "rank_corr": rank_corr,
        "proj_corr": proj_corr,
        "distance_matrices": distance_matrices,
        "per_pair": per_pair,
    }


def _build_eval_samplers(s2_clients, s1_clients, val_datasets, n_way, k_shot, q_query):
    """Build evaluation episode samplers from validation datasets."""
    eval_samplers = {}
    if "S2" in val_datasets and s2_clients:
        eval_samplers["S2"] = [
            (
                client,
                EpisodeSampler(
                    val_datasets["S2"],
                    n_way=n_way,
                    k_shot=k_shot,
                    q_query=q_query,
                    is_train=False,
                ),
            )
            for client in s2_clients
        ]
    if "S1" in val_datasets and s1_clients:
        eval_samplers["S1"] = [
            (
                client,
                EpisodeSampler(
                    val_datasets["S1"],
                    n_way=n_way,
                    k_shot=k_shot,
                    q_query=q_query,
                    is_train=False,
                ),
            )
            for client in s1_clients
        ]
    return eval_samplers


def _snapshot_state(server, s2_clients, s1_clients, shared_body, is_fedproto):
    """Snapshot current state for best model tracking."""
    state = {
        "s2_client_states": {
            c.client_id: {k: v.clone().cpu() for k, v in c.model.state_dict().items()}
            for c in s2_clients
        },
        "s1_client_states": {
            c.client_id: {k: v.clone().cpu() for k, v in c.model.state_dict().items()}
            for c in s1_clients
        },
    }

    if is_fedproto:
        if hasattr(server, "global_protos"):
            state["global_protos"] = {
                k: v.clone().cpu() for k, v in server.global_protos.items()
            }
        if hasattr(server, "global_D") and server.global_D is not None:
            state["global_D"] = server.global_D.clone().cpu()
            state["obs_mask"] = server.obs_mask.clone().cpu()
            state["class_to_idx"] = dict(server.class_to_idx)
        if (
            hasattr(server, "global_proj_weights")
            and server.global_proj_weights is not None
        ):
            state["global_proj_weights"] = {
                k: v.clone().cpu() for k, v in server.global_proj_weights.items()
            }
    else:
        state["shared_body"] = {
            k: v.clone().cpu() for k, v in shared_body.state_dict().items()
        }

    return state


def _load_client_states(
    server, s2_clients, s1_clients, state, shared_body, is_fedproto
):
    """Restore client and server state from snapshot."""
    if state is None:
        return

    if not is_fedproto and "shared_body" in state:
        shared_body.load_state_dict(state["shared_body"])
        return

    for c in s2_clients:
        if c.client_id in state.get("s2_client_states", {}):
            c.model.load_state_dict(state["s2_client_states"][c.client_id], strict=False)
    for c in s1_clients:
        if c.client_id in state.get("s1_client_states", {}):
            c.model.load_state_dict(state["s1_client_states"][c.client_id], strict=False)

    if "global_protos" in state and hasattr(server, "global_protos"):
        server.global_protos = state["global_protos"]
    if "global_D" in state and hasattr(server, "global_D"):
        server.global_D = state["global_D"]
        server.obs_mask = state["obs_mask"]
        server.class_to_idx = state["class_to_idx"]
    if "global_proj_weights" in state and hasattr(server, "global_proj_weights"):
        server.global_proj_weights = state["global_proj_weights"]


PATIENCE_FEDERATED = 100


def run_federated(
    label: str,
    server,
    s2_clients: list,
    s1_clients: list,
    val_datasets: dict,
    test_datasets: dict,
    shared_body=None,
    n_rounds: int = 50,
    n_episodes: int = 10,
    k_shot: int = 1,
    q_query: int = 15,
    n_way: int = 5,
    device: torch.device = None,
    val_every: int = 4,
) -> dict:

    device = device or torch.device("cpu")
    is_fedproto = shared_body is None

    # Build eval samplers once (reused every eval round)
    eval_samplers = _build_eval_samplers(
        s2_clients, s1_clients, val_datasets, n_way, k_shot, q_query
    )

    ckpt = load_checkpoint(label)
    start = 1
    best_acc = 0.0
    best_state = None
    no_improve = 0

    history = {
        "round": [],
        "loss": [],
        "train_acc": [],
        "val_S2": [],
        "val_S1": [],
        "val_avg": [],
        "val_S2_ci": [],
        "val_S1_ci": [],
        "val_S2_per_client": [],
        "val_S1_per_client": [],
        # Analysis metrics
        "inter_class_margin_S2": [],
        "inter_class_margin_S1": [],
        "proto_norms_S2": [],
        "proto_norms_S1": [],
        "accuracy_gap": [],
        "rank_correlation_S2": [],
        "rank_correlation_S1": [],
        # Distance matrix snapshots for heatmaps
        "distance_matrix_snapshots": {},
        # Per-class-pair analysis
        "per_pair_distances": {},
    }

    # Resume from checkpoint
    if ckpt is not None:
        start = ckpt["round"] + 1
        history = ckpt["history"]
        best_acc = ckpt["best_acc"]
        best_state = ckpt["best_state"]
        no_improve = ckpt.get("no_improve", 0)
        _load_client_states(
            server, s2_clients, s1_clients, best_state, shared_body, is_fedproto
        )

    if best_state is None:
        best_state = _snapshot_state(
            server, s2_clients, s1_clients, shared_body, is_fedproto
        )

    # Training loop
    for r in range(start, n_rounds + 1):
        t0 = time.time()
        avg_loss, avg_acc = server.train_round(n_episodes=n_episodes)
        elapsed = time.time() - t0

        history["round"].append(r)
        history["loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)

        # Evaluation and analysis
        if r % val_every == 0 or r == 1:

            # --- Standard validation accuracy ---
            eval_enc = build_eval_encoders(shared_body, device, s2_clients, s1_clients)
            val_m = evaluate_with_ci(
                eval_enc,
                val_datasets,
                device,
                n_episodes=200,
                k_shot=k_shot,
                q_query=q_query,
                n_way=n_way,
                seed=r,
            )

            s2_m = val_m.get("S2", {"mean": 0, "ci": 0, "per_client": []})
            s1_m = val_m.get("S1", {"mean": 0, "ci": 0, "per_client": []})
            avg = (s2_m["mean"] + s1_m["mean"]) / 2

            history["val_S2"].append(s2_m["mean"])
            history["val_S1"].append(s1_m["mean"])
            history["val_avg"].append(avg)
            history["val_S2_ci"].append(s2_m["ci"])
            history["val_S1_ci"].append(s1_m["ci"])
            history["val_S2_per_client"].append(s2_m.get("per_client", []))
            history["val_S1_per_client"].append(s1_m.get("per_client", []))

            # --- Collapse analysis metrics ---
            analysis = _compute_analysis_metrics(server, s2_clients, s1_clients, device)

            history["inter_class_margin_S2"].append(analysis["margins"]["S2"])
            history["inter_class_margin_S1"].append(analysis["margins"]["S1"])
            history["proto_norms_S2"].append(analysis["norms"]["S2"])
            history["proto_norms_S1"].append(analysis["norms"]["S1"])
            history["accuracy_gap"].append(abs(s2_m["mean"] - s1_m["mean"]))
            history["rank_correlation_S2"].append(analysis["rank_corr"].get("S2", None))
            history["rank_correlation_S1"].append(analysis["rank_corr"].get("S1", None))

            # --- Distance matrix snapshots at key rounds ---
            if r in [1, 5, 10, 25, 50] or r == n_rounds:
                history["distance_matrix_snapshots"][r] = analysis["distance_matrices"]
                history["per_pair_distances"][r] = analysis["per_pair"]

            # --- Print analysis ---
            gap = history["accuracy_gap"][-1]
            m_s2 = analysis["margins"]["S2"]
            m_s1 = analysis["margins"]["S1"]
            n_s2 = analysis["norms"]["S2"]
            n_s1 = analysis["norms"]["S1"]

            print(
                f"[{label}] Round {r:>3}/{n_rounds}  "
                f"loss={avg_loss:.4f}  "
                f"S2={s2_m['mean']:.1f}%  S1={s1_m['mean']:.1f}%  "
                f"gap={gap:.1f}%  "
                f"margin(S2={m_s2:.3f},S1={m_s1:.3f})  "
                f"norm(S2={n_s2:.1f},S1={n_s1:.1f})  "
                f"best={best_acc:.1f}%  ({elapsed:.0f}s)"
            )

            # --- Best model tracking ---
            if avg > best_acc:
                best_acc = avg
                no_improve = 0
                best_state = _snapshot_state(
                    server, s2_clients, s1_clients, shared_body, is_fedproto
                )
            else:
                no_improve += val_every

            if no_improve >= PATIENCE_FEDERATED:
                print(f"[{label}] Early stopping at round {r}")
                break

        else:
            # Non-eval round — append None to keep lists aligned
            for key in [
                "val_S2",
                "val_S1",
                "val_avg",
                "val_S2_ci",
                "val_S1_ci",
                "val_S2_per_client",
                "val_S1_per_client",
                "inter_class_margin_S2",
                "inter_class_margin_S1",
                "proto_norms_S2",
                "proto_norms_S1",
                "accuracy_gap",
                "rank_correlation_S2",
                "rank_correlation_S1",
            ]:
                history[key].append(None)

            print(
                f"[{label}] Round {r:>3}/{n_rounds}  "
                f"loss={avg_loss:.4f}  train_acc={avg_acc:.1f}%  ({elapsed:.0f}s)"
            )

        # Checkpoint
        save_checkpoint(
            label,
            r,
            model_state=(
                {k: v.cpu() for k, v in shared_body.state_dict().items()}
                if not is_fedproto
                else {}
            ),
            history=history,
            best_acc=best_acc,
            best_state=best_state,
            no_improve=no_improve,
        )

    # --- Final test with best model ---
    print(f"\n[{label}] Loading best model (val_avg={best_acc:.2f}%) for final test")
    _load_client_states(
        server, s2_clients, s1_clients, best_state, shared_body, is_fedproto
    )

    eval_enc = build_eval_encoders(shared_body, device, s2_clients, s1_clients)
    test_m = evaluate_with_ci(
        eval_enc,
        test_datasets,
        device,
        n_episodes=600,
        k_shot=k_shot,
        q_query=q_query,
        n_way=n_way,
        seed=42,
    )

    # Final analysis snapshot on best model
    final_analysis = _compute_analysis_metrics(server, s2_clients, s1_clients, device)

    result = {
        "label": label,
        "history": history,
        "best_val": best_acc,
        "best_state": best_state,
        "test": test_m,
        "final_analysis": final_analysis,
    }
    print(f"[{label}] Test → {test_m}")
    save_result(label, result)
    return result
