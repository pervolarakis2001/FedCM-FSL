import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import numpy as np
import os
import shutil
from pathlib import Path

from config import FIGURES_PATH

COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
SAVE_DIR = Path(FIGURES_PATH)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def plot_convergence(
    results: dict,  # {label: result_dict}
    modality: str = "S2",  # "S2", "S1", or "avg"
    k_shot: int = 1,
    scenario: str = "DS1",
    title: str = None,
    save_name: str = None,
):
    """
    x = communication round, y = val accuracy.
    Filters results to matching scenario + k_shot automatically.
    Call once per scenario — all methods appear as separate lines.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    metric = f"val_{modality}" if modality != "avg" else "val_avg"
    tag = f"{scenario}_{k_shot}shot"

    plotted = 0
    for label, res in results.items():
        if tag not in label:
            continue
        h = res["history"]
        rounds = h["round"]
        vals = h[metric]

        # Remove None placeholders (rounds where we skipped validation)
        pairs = [(r, v) for r, v in zip(rounds, vals) if v is not None]
        if not pairs:
            continue
        rs, vs = zip(*pairs)

        method = label.split("_")[0]  # FedAvg / FedProto / Ours etc.
        ax.plot(rs, vs, linewidth=2, marker="o", markersize=3, label=method)
        plotted += 1

    if plotted == 0:
        print(f"No results matched tag='{tag}' and modality='{modality}'")
        plt.close()
        return

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel(f"Validation Accuracy % ({modality})", fontsize=12)
    ax.set_title(
        title or f"Convergence — {scenario} {k_shot}-shot ({modality})", fontsize=13
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = save_name or f"convergence_{scenario}_{k_shot}shot_{modality}.pdf"
    plt.savefig(SAVE_DIR / fname, bbox_inches="tight")
    plt.savefig(SAVE_DIR / fname.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.show()
    print(f"  ✓  Saved → {SAVE_DIR / fname}")


def plot_convergence_grid(
    results: dict,
    scenarios: list = ("DS1", "DS3", "DS4", "DS5"),
    k_shot: int = 1,
    modality: str = "avg",
):
    """
    2×2 grid — one subplot per scenario.
    All methods overlaid in each subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metric = f"val_{modality}" if modality != "avg" else "val_avg"

    method_colors = {}
    color_idx = 0

    for ax, scenario in zip(axes, scenarios):
        tag = f"{scenario}_{k_shot}shot"
        for label, res in results.items():
            if tag not in label:
                continue
            method = label.split("_")[0]
            if method not in method_colors:
                method_colors[method] = COLORS[color_idx % len(COLORS)]
                color_idx += 1

            h = res["history"]
            pairs = [(r, v) for r, v in zip(h["round"], h[metric]) if v is not None]
            if not pairs:
                continue
            rs, vs = zip(*pairs)
            ax.plot(
                rs,
                vs,
                linewidth=2,
                color=method_colors[method],
                label=method,
                marker="o",
                markersize=2,
            )

        ax.set_title(f"{scenario}  ({k_shot}-shot)", fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy %")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle(f"Convergence — {k_shot}-shot  ({modality})", fontsize=13)
    plt.tight_layout()
    fname = f"convergence_grid_{k_shot}shot_{modality}.pdf"
    plt.savefig(SAVE_DIR / fname, bbox_inches="tight")
    plt.savefig(SAVE_DIR / fname.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.show()
    print(f"  ✓  Saved → {SAVE_DIR / fname}")


# ── B. Inter-modal L2 distance over rounds ───────────────────────────────────


def plot_proto_distance_over_rounds(
    results: dict,
    label_naive: str,  # e.g. "FedProto_DS1_multimodal_1shot"
    label_ours: str,  # e.g. "Ours_DS1_multimodal_1shot"
    save_name: str = "proto_dist_over_rounds.pdf",
):
    """
    Two y-axes: left = L2 distance, right = accuracy.
    Shows that as your method aligns prototypes, accuracy rises.
    Naive FedProto: distance stays high, accuracy plateaus lower.
    """
    res_n = results.get(label_naive)
    res_o = results.get(label_ours)

    if res_n is None or res_o is None:
        print(f"Missing results for {label_naive} or {label_ours}")
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    def clean(rounds, vals):
        return zip(*[(r, v) for r, v in zip(rounds, vals) if v is not None])

    h_n = res_n["history"]
    h_o = res_o["history"]

    # L2 distance — left axis
    if any(v is not None for v in h_n["proto_l2"]):
        rn, ln = clean(h_n["round"], h_n["proto_l2"])
        ax1.plot(list(rn), list(ln), "r--", linewidth=2, label="FedProto (naive) — L2")
    if any(v is not None for v in h_o["proto_l2"]):
        ro, lo = clean(h_o["round"], h_o["proto_l2"])
        ax1.plot(list(ro), list(lo), "b-", linewidth=2, label="Ours — L2")

    # Accuracy — right axis (dashed, lighter)
    rn2, an2 = clean(h_n["round"], h_n["val_avg"])
    ro2, ao2 = clean(h_o["round"], h_o["val_avg"])
    ax2.plot(
        list(rn2), list(an2), "r:", linewidth=1.5, alpha=0.6, label="FedProto — Acc"
    )
    ax2.plot(list(ro2), list(ao2), "b:", linewidth=1.5, alpha=0.6, label="Ours — Acc")

    ax1.set_xlabel("Communication Round", fontsize=12)
    ax1.set_ylabel("Inter-Modal Prototype L2 Distance", fontsize=11)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11, color="gray")
    ax1.set_title("Cross-Modal Prototype Alignment over Rounds", fontsize=13)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="center right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / save_name, bbox_inches="tight")
    plt.savefig(
        SAVE_DIR / save_name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150
    )
    plt.show()
    print(f"  ✓  Saved → {SAVE_DIR / save_name}")


# ── C. Per-class prototype distance bar chart ─────────────────────────────────


def plot_per_class_proto_distance(
    result_naive: dict,
    result_ours: dict,
    class_names: dict,  # {class_idx (int): str}
    round_idx: int = -1,  # which saved round to use (-1 = last saved)
    save_name: str = "per_class_proto_dist.pdf",
):
    """
    Bar chart: per-class L2 distance for naive FedProto vs Ours.
    Uses the prototype tensors saved inside history["proto_s2/s1"].
    """

    def get_round_protos(result, round_idx):
        h = result["history"]
        keys = sorted(h["proto_s2"].keys())
        if not keys:
            return None, None
        r = keys[round_idx]
        return h["proto_s2"][r], h["proto_s1"][r]

    ps2_n, ps1_n = get_round_protos(result_naive, round_idx)
    ps2_o, ps1_o = get_round_protos(result_ours, round_idx)

    if ps2_n is None or ps2_o is None:
        print("Prototype tensors not found. " "Make sure track_protos=True was set.")
        return

    common = sorted(
        set(ps2_n.keys()) & set(ps1_n.keys()) & set(ps2_o.keys()) & set(ps1_o.keys())
    )

    names = [class_names.get(c, str(c)) for c in common]
    l2_n = [torch.norm(ps2_n[c].float() - ps1_n[c].float()).item() for c in common]
    l2_o = [torch.norm(ps2_o[c].float() - ps1_o[c].float()).item() for c in common]

    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - w / 2,
        l2_n,
        w,
        color="tomato",
        alpha=0.85,
        label="Naive FedProto",
        edgecolor="white",
    )
    ax.bar(
        x + w / 2,
        l2_o,
        w,
        color="steelblue",
        alpha=0.85,
        label="Ours (Cross-Modal)",
        edgecolor="white",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("S1–S2 Prototype L2 Distance", fontsize=11)
    ax.set_title("Per-Class Inter-Modal Prototype Distance (Final Round)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / save_name, bbox_inches="tight")
    plt.savefig(
        SAVE_DIR / save_name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150
    )
    plt.show()
    print(f"  ✓  Saved → {SAVE_DIR / save_name}")


# ── D. t-SNE of prototypes ────────────────────────────────────────────────────


def plot_tsne_prototypes(
    result_naive: dict,
    result_ours: dict,
    class_names: dict,  # {class_idx (int): str}
    round_idx: int = -1,
    save_name: str = "tsne_prototypes.pdf",
    perplexity: int = 4,
):
    """
    Side-by-side t-SNE of S1 and S2 prototypes.
    ○ = S2,  ✕ = S1,  same color = same class.
    Dashed lines connect S1/S2 pair of the same class.
    Short lines = good cross-modal alignment.
    """

    def get_protos(result, round_idx):
        h = result["history"]
        keys = sorted(h["proto_s2"].keys())
        if not keys:
            return None, None
        r = keys[round_idx]
        return h["proto_s2"][r], h["proto_s1"][r]

    ps2_n, ps1_n = get_protos(result_naive, round_idx)
    ps2_o, ps1_o = get_protos(result_ours, round_idx)

    if ps2_n is None:
        print("No prototype tensors saved. Set track_protos=True.")
        return

    n_classes = len(class_names)
    cmap = plt.cm.tab10(np.linspace(0, 1, n_classes))
    cls_list = sorted(class_names.keys())
    color_map = {c: cmap[i] for i, c in enumerate(cls_list)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, (ps2, ps1), title in zip(
        axes,
        [(ps2_n, ps1_n), (ps2_o, ps1_o)],
        ["Naive FedProto", "Cross-Modal FedProto (Ours)"],
    ):
        common = sorted(set(ps2.keys()) & set(ps1.keys()))

        # Stack vectors for t-SNE
        vecs, labels, mods = [], [], []
        for cls in common:
            vecs.append(ps2[cls].float().numpy())
            labels.append(cls)
            mods.append("S2")
            vecs.append(ps1[cls].float().numpy())
            labels.append(cls)
            mods.append("S1")

        X = np.stack(vecs)
        perp = min(perplexity, len(X) - 1)
        X2d = TSNE(
            n_components=2, perplexity=perp, random_state=42, n_iter=2000
        ).fit_transform(X)

        # Draw dashed connection lines first (behind points)
        for i in range(0, len(X2d), 2):  # every pair (S2, S1)
            s2_xy = X2d[i]
            s1_xy = X2d[i + 1]
            cls = labels[i]
            ax.plot(
                [s2_xy[0], s1_xy[0]],
                [s2_xy[1], s1_xy[1]],
                color=color_map[cls],
                linewidth=1.5,
                linestyle="--",
                alpha=0.5,
                zorder=2,
            )

        # Draw points on top
        for i, (x, y) in enumerate(X2d):
            cls = labels[i]
            marker = "o" if mods[i] == "S2" else "X"
            ax.scatter(
                x,
                y,
                color=color_map[cls],
                marker=marker,
                s=250,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

    # Shared legend
    class_handles = [
        mpatches.Patch(color=color_map[c], label=class_names.get(c, str(c)))
        for c in cls_list
        if c in color_map
    ]
    modal_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markersize=9,
            linestyle="None",
            label="S2",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="gray",
            markersize=9,
            linestyle="None",
            label="S1",
        ),
        Line2D(
            [0], [0], color="gray", linewidth=1.5, linestyle="--", label="S1–S2 pair"
        ),
    ]
    fig.legend(
        handles=class_handles + modal_handles,
        loc="center right",
        bbox_to_anchor=(1.13, 0.5),
        fontsize=8,
        framealpha=0.9,
    )

    plt.suptitle(
        "Prototype Space (t-SNE)\n"
        "Shorter dashed lines = better cross-modal alignment",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(SAVE_DIR / save_name, bbox_inches="tight")
    plt.savefig(
        SAVE_DIR / save_name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150
    )
    plt.show()
    print(f"  ✓  Saved → {SAVE_DIR / save_name}")
