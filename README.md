# FedCM-FSL

Federated few-shot learning on multi-modal satellite imagery (Sentinel-1 SAR + Sentinel-2 optical) using the BigEarthNet dataset.

Supports two federation strategies (**FedProto**, **FedAvg**) across five data distribution scenarios (DS1–DS5), evaluated with Prototypical Networks under 1-shot and 5-shot settings.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended; falls back to CPU)
- Kaggle account (to download the dataset)

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd FedCM-FSL
```

### 2. Create and activate a virtual environment

```bash
python -m venv fedcm_env
source fedcm_env/bin/activate
```

Or with conda:

```bash
conda create -n fedcm_env python=3.10 -y
conda activate fedcm_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The `requirements.txt` includes the PyTorch CUDA 12.4 index. If you need a different CUDA version, edit the `--extra-index-url` line before installing.

### 4. Configure paths

Copy the example config and edit it:

```bash
cp config.py.example config.py
```

Open `config.py` and set the paths to your data:

```python
BEN_ROOT_S1_PATH  = "/path/to/your/S1"
BENR_ROOT_S2_PATH = "/path/to/your/S2"
```

Output directories (`output/figures`, `output/checkpoints`, `output/results`) are created automatically on first run.

### 5. Download the dataset

Ensure your Kaggle API credentials are at `~/.kaggle/kaggle.json`, then run:

```bash
python kaggle_downloader.py
```

This downloads `pervolarakis/bigearth-federated-few-shot` and prints the local path. Update `BEN_ROOT_S1_PATH` and `BENR_ROOT_S2_PATH` in `config.py` accordingly.

---

## Running Experiments

All experiments are launched from the project root via `main.py`.

### Basic usage

```bash
python main.py --mode <federated|centralized> --scenario <DS1|DS2|DS3|DS4|DS5> [options]
```

### Examples

```bash
# FedProto, DS1, both 1-shot and 5-shot
python main.py --mode federated --method FedProto --scenario DS1

# FedAvg, DS2, 5-shot only
python main.py --mode federated --method FedAvg --scenario DS2 --k-shots 5

# Centralized baseline, DS3
python main.py --mode centralized --scenario DS3

# Custom rounds / episodes
python main.py --mode federated --method FedProto --scenario DS1 --n-rounds 50 --n-episodes 100

# Skip plots (useful on headless servers)
python main.py --mode federated --method FedProto --scenario DS1 --no-plots
```

### Background execution with nohup

```bash
mkdir -p logs
nohup python -u main.py --mode federated --method FedProto --scenario DS1 \
  > logs/DS1_fedproto.log 2>&1 &
echo "PID: $!"

# Monitor live output
tail -f logs/DS1_fedproto.log
```

### All CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | required | `federated` or `centralized` |
| `--method` | `FedProto` | `FedProto` or `FedAvg` (federated only) |
| `--scenario` | required | `DS1`–`DS5` |
| `--k-shots` | `1 5` | One or more k-shot values, e.g. `--k-shots 1 5` |
| `--n-clients` | `5` | Number of federated clients |
| `--n-rounds` | `20` | Federated communication rounds |
| `--n-episodes` | `50` | Episodes per round (federated) or total (centralized) |
| `--val-every` | `1` | Validate every N rounds |
| `--metadata-csv` | see config | Path to `metadata_sampled.csv` |
| `--s2-root` | see config | Root directory for S2 patches |
| `--s1-root` | see config | Root directory for S1 patches |
| `--device` | auto | Override device, e.g. `cuda`, `cuda:1`, `cpu` |
| `--no-plots` | off | Skip figure generation after training |

---

## Data Distribution Scenarios

| Scenario | Description |
|----------|-------------|
| DS1 | Summer data, IID split across 5 clients (one per country) |
| DS2 | Summer data, restricted to 5 paper countries |
| DS3 | All seasons, IID split |
| DS4 | Summer data, half the clients missing S2 (optical) modality |
| DS5 | Summer data, half the clients missing S1 (SAR) modality |

---

## Project Structure

```
FedCM-FSL/
├── main.py                          # Experiment runner / CLI entry point
├── config.py                        # Paths and dataset constants (git-ignored)
├── config.py.example                # Config template
├── requirements.txt
├── kaggle_downloader.py             # Dataset download helper
│
├── src/
│   ├── cetralised_trainer.py        # Centralized ProtoNet baseline
│   ├── federated_trainier.py        # Federated training loop
│   ├── evaluator.py                 # Evaluation with confidence intervals
│   │
│   ├── models/
│   │   └── protonet.py              # ResNet12, ProtoNet, SplitEncoder
│   │
│   ├── datasets/
│   │   ├── dataset_s1.py            # BigEarthNet S1 (SAR, 2-band)
│   │   └── dataset_s2.py            # BigEarthNet S2 (optical, 10-band)
│   │
│   ├── federated_learning/
│   │   ├── client.py                # FedAvgClient, FedProtoClient
│   │   ├── server.py                # FedAvgServer, FedProtoServer
│   │   └── partitioner.py           # DS1–DS5 data partitioning logic
│   │
│   └── utils/
│       ├── episode_sampler.py       # Few-shot episode generator
│       ├── loader.py                # S1/S2 patch loading from TIFF
│       ├── save.py                  # Checkpointing and result serialisation
│       └── visualiser.py            # Convergence, prototype, t-SNE plots
│
├── output/                          # Auto-created: figures, checkpoints, results
└── logs/                            # nohup log files
```

---

## Outputs

| Path | Contents |
|------|----------|
| `output/results/` | Per-experiment result dicts (pickle) |
| `output/checkpoints/` | Per-round checkpoints — interrupted runs resume automatically |
| `output/figures/` | Convergence curves, prototype distance plots, t-SNE visualisations |
