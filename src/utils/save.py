from pathlib import Path
import pandas as pd
import pickle, copy, time
from config import CHECKPOINTS_PATH, FIGURES_PATH, RESULTS_DIR

CKPT_DIR = Path(CHECKPOINTS_PATH)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _experiment_dir(base_dir: Path, label: str) -> Path:
    """Return (and create) a subdirectory named after the experiment label."""
    exp_dir = base_dir / label
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# ──────────────────────────────────────────────
# Checkpoints
# ──────────────────────────────────────────────


def save_checkpoint(
    label: str,
    round_idx: int,
    model_state: dict,
    history: dict,
    best_acc: float,
    best_state: dict,
    no_improve: int = 0,
    **extra,
):
    """Save mid-run state so a crash doesn't lose everything."""
    path = CKPT_DIR / f"{label}_ckpt.pt"
    payload = {
        "label": label,
        "round": round_idx,
        "model_state": model_state,
        "history": history,
        "best_acc": best_acc,
        "best_state": best_state,
        "no_improve": no_improve,
        **extra,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(label: str):
    """Return checkpoint dict, or None if not found."""
    path = CKPT_DIR / f"{label}_ckpt.pt"
    if path.exists():
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        print(f"  ↩  Resuming '{label}' from round {ckpt['round']}")
        return ckpt
    return None


# Per-experiment results
def save_result(label: str, result: dict, results_dir: str = None):
    """
    Persist a result dict into its own experiment subfolder.

    Layout:  <results_dir>/<label>/<label>.pkl
    """
    base = Path(results_dir) if results_dir else RESULTS_DIR
    exp_dir = _experiment_dir(base, label)
    path = exp_dir / f"{label}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"Result saved → {path}")


def load_result(label: str, results_dir: str = None):
    """Load a single experiment result, or return None if absent."""
    base = Path(results_dir) if results_dir else RESULTS_DIR
    path = base / label / f"{label}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_all_results(prefix: str = "", results_dir: str = None) -> dict:
    """
    Load every saved result from all experiment subfolders,
    optionally filtered by a label prefix.

    Returns a dict  { label: result_dict }
    """
    base = Path(results_dir) if results_dir else RESULTS_DIR
    all_res = {}

    for pkl_path in sorted(base.glob("*/*.pkl")):
        label = pkl_path.stem
        if prefix and not label.startswith(prefix):
            continue
        with open(pkl_path, "rb") as f:
            all_res[label] = pickle.load(f)

    print(f"Loaded {len(all_res)} results from {base}")
    return all_res
