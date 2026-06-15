# pLaSDI

**Physics-Informed Latent-Space Dynamics Identification**

Maintained by **Jeongwoo Nam**  
High Energy Density Physics Laboratory, Gwangju Institute of Science and Technology

> **This project is built upon GPLaSDI, developed by Lawrence Livermore National Laboratory (LLNL).**  
> GPLaSDI is a Gaussian Process-based Interpretable Latent Space Dynamics Identification framework via deep autoencoders.

---

Physics-informed neural network system for modeling atomic population dynamics in plasma physics. Compresses high-dimensional atomic state distribution data (1583 dimensions) into a low-dimensional latent space (3-10 dimensions) using an autoencoder, then learns interpretable governing equations via SINDyC(DMDc).

---

## Overview

LaSDIc combines three core components:

1. **Autoencoder** — Compresses 1583-dimensional population data into a 3-dimensional latent space and reconstructs it back
2. **SINDyC** — Learns dynamics equations in latent space: `dZ/dt = a + A·Z + B·U` (lstsq) or `dZ/dt = a(U) + A(U)·Z` (adaptive)
3. **Physics-informed losses** — Enforces physical constraints (charge state distribution, mean charge, fraction conservation, rate equations)

The system operates on temperature (T) and density (n) as control inputs to predict atomic population evolution across different plasma conditions.

---

## Requirements

### Python Version

- **Python >= 3.8**

### Dependencies

| Package | Version |
|---------|---------|
| `torch` | >= 2.3.0 |
| `numpy` | >= 1.26.4 |
| `scikit-learn` | >= 1.4.2 |
| `scipy` | >= 1.13.1 |
| `pyyaml` | >= 6.0 |
| `matplotlib` | >= 3.8.4 |
| `argparse` | >= 1.4.0 |
| `h5py` | latest |

Install all dependencies at once:

```bash
pip install "torch>=2.3.0" "numpy>=1.26.4" "scikit-learn>=1.4.2" "scipy>=1.13.1" \
            "pyyaml>=6.0" "matplotlib>=3.8.4" "argparse>=1.4.0" h5py
```

---

### `src/` — Core Package (14 files)

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization and public API exports |
| `common.py` | **NEW** — Shared utilities: unified activation dict (ACT_DICT), FD operator cache, common time derivative function |
| `checkpoint_manager.py` | **NEW** — BestTracker + CheckpointManager for RAM-buffered best model + log management |
| `autoencoder.py` | Autoencoder model (Encoder + Decoder with configurable layers and activation) |
| `sindyc.py` | SINDyC lstsq mode — global constant coefficients via least-squares |
| `sindyc_adaptive.py` | SINDyC Adaptive mode — CoefNet generates U-dependent coefficients with Hurwitz stability |
| `sindy_utils.py` | SINDy loss calculator, Hurwitz penalty functions, coefficient splitting utilities |
| `trainer.py` | Main training loop — orchestrates AE + SINDy + physics losses with mini-batching |
| `train_utils.py` | LR scheduling, checkpointing, logging, ramp-up weight functions, timing profiler |
| `atomic_physics.py` | CSD, Zbar, ion fraction computation + rate equation losses (all autograd-compatible) |
| `scaling.py` | Population scaler (log-space W-transform) + control variable scaler + PyTorch helpers |
| `data_utils.py` | Data loading, segment slicing, train/val splitting, mini-batch iteration, steady-state data |
| `fd.py` | SBP (Summation-By-Parts) finite difference operators for time derivative computation |
| `visualization.py` | Plotting utilities for training curves, heatmaps, and analysis |

### Root Directory (6 scripts + 2 data files)

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters — data, model, SINDy, training, Hurwitz, saving (with disk flush interval) |
| `train.py` | Training entry point — loads config, creates trainer, runs training loop |
| `evaluate.py` | Comprehensive evaluation — auto-loads config from model directory, 15+ visualization sections |
| `rollout.py` | SINDy rollout evaluation — forward simulation analysis |
| `save_outputs.py` | Saves predictions to `.npz` files — auto-loads config from model directory |
| `view_truth.py` | Ground truth data visualization |
| `conftest.py` | **NEW** — pytest path configuration |
| `pyproject.toml` | **NEW** — Dependency management and pytest settings |
| `name_total.txt` | All 1583 atomic energy level names (one per line). Used for CSD/Zbar charge parsing |
| `name_bound.txt` | Subset of names for CSD/Zbar calculation (ground states per charge, e.g., Sn_0_0, Sn_1_0, ...) |

### `tests/` — Unit Tests (NEW)

| File | Description |
|------|-------------|
| `test_scaling.py` | W↔fraction roundtrip, fraction sum=1, range validation (9 tests) |
| `test_sindy.py` | split_coefs consistency, Hurwitz guarantee, calibrate (7 tests) |
| `test_physics.py` | CSD sum=1, Zbar range, FD accuracy, PhysicsLoss (12 tests) |

### Data Files (external, configured via `config.py`)

```
data_storage/
├── lithography_4ns/
│   ├── density_population_seg0.txt ... seg184.txt    # (nt × nx) population time series
│   └── historyfile_seg0.txt ... seg184.txt           # (nt × 2) control variables [T, n]
└── steady_data/
    ├── steady_population.txt                          # Steady-state populations
    └── steady_history.txt                             # Steady-state conditions [T, n]
```

---

## Architecture

```
Population (1583-dim)  →  Encoder  →  Z (3-dim)  →  Decoder  →  N̂ (1583-dim)
                                        ↓
                                   SINDyC learns
                                dZ/dt = a + A·Z + B·U
```

### Scaling Pipeline

Raw population data goes through a log-space transformation before entering the autoencoder:

```
Raw Population (N) → Normalize (fraction) → Log transform → Min-Max → W ∈ [0, 1]
```

The decoder outputs sigmoid-activated values in W-space, which are then inverse-transformed back to physical fractions for physics loss computation.

### SINDy Modes

**lstsq mode** (`use_adaptive=False`): Solves `dZ/dt = a + A·Z + B·U` via `torch.linalg.lstsq` each epoch. Coefficients (a, A, B) are time/condition-invariant constants.

**Adaptive mode** (`use_adaptive=True`): A small neural network (CoefNet) maps control inputs U(t) to time-varying coefficients a(U) and A(U). Hurwitz stability is structurally guaranteed via `A = -(P·Pᵀ + S) - ε·I`, where S is anti-symmetric (optional, `adaptive_symmetric=False`). -> we didn't discuss about this mode in the paper

---

## Configuration Guide

All settings are in `config.py`. Key sections:

### DataConfig

- `data_dir` / `history_dir` — paths to population and history files
- `nx = 1583` — number of atomic energy levels (must match `name_total.txt` line count)
- `dt = 0.01e-9` — simulation time step in seconds
- `pop_lim = 1e-50` — population floor for numerical stability
- `steady_enable` — whether to use steady-state data for equilibrium loss

### ModelConfig

- `latent_dim = 3` — latent space dimension
- `hidden = [800, 400, 200, 100, 20]` — encoder hidden layers (decoder uses reverse)
- `activation = "mish"` — activation function

### SINDyConfig

- `weight` — SINDy fitting loss weight (dZ/dt residual MSE)
- `coef_weight` — coefficient regularization weight (L2 norm) -> in the paper, this value is 0
- `fd_type = "sbp12"` — finite difference stencil accuracy
- `dt_eff = 1e-3` — effective dt for SINDy (numerical scaling, not physical dt)
- `use_adaptive` — `True` for CoefNet mode, `False` for lstsq mode
- `adaptive_symmetric` — `True`: symmetric A (real eigenvalues), `False`: asymmetric A (allows oscillatory dynamics)
- `adaptive_eps` — Hurwitz stability margin ε

### TrainConfig

- `epochs`, `batch_size`, `lr` — standard training hyperparameters
- `w_rec`, `w_fse`, `w_frac`, `w_ion`, `w_zbar` — AE + physics loss weights
- `w_rate_W/N/CSD/Zbar` — rate equation loss weights (default 0, optional)
- `steady_weight` — steady-state loss weight
- `noise = 0.1` — Gaussian noise σ injected into encoder output during training
- `lr_scheduler` — `"cosine"` (recommended), `"one_cycle"`, or `"none"`
- `ramp_config` — per-loss ramp-up schedule (T: completion epoch, mode: `"exp_slow"`, `"linear"`, `"cosine"`)
- `loss_mode` — per-loss operation mode dictionary (see [Loss Mode System](#loss-mode-system) below)

### HurwitzConfig

- `enable` — whether to compute Hurwitz penalty (only for lstsq mode; adaptive is structurally stable)
- `gate_enable` — block saving if model is unstable

---

## Quick Start

### 1. Prepare Data

Ensure population and history files are in the configured directory. Each case `N` needs:
- `density_population_seg{N}.txt` — (nt × nx) population matrix
- `historyfile_seg{N}.txt` — (nt × 2) control variables [T, density]

### 2. Configure

Edit `config.py`:

```python
cfg = create_default_config()

# Data paths
cfg.data.data_dir = "path/to/your/data"
cfg.data.history_dir = "path/to/your/data"

# Training
cfg.train.epochs = 30000
cfg.train.lr = 1e-5
cfg.train.batch_size = 30

# Output
cfg.save.save_root = "./runs/my_experiment"
```

### 3. Train

```bash
python train.py
```

Or in VSCode, open `train.py` and run cells with `Shift+Enter`.

### 4. Evaluate

Edit `evaluate.py` top section:

```python
CONFIG_DIR = "./runs/my_experiment"
SHOW_PLOTS = True
```

Then run:

```bash
python evaluate.py
```

### 5. Output Files

```
runs/my_experiment/
├── model_train.pt              # Best model checkpoint (AE + SINDy + optimizer)
├── config.py                   # Config snapshot at training start
├── config_resume_*.py          # Config snapshots for each resume session (if any)
├── train_metrics.csv           # Per-epoch metrics (36 columns, flushed every disk_save_every epochs)
├── train_metrics.npz           # Metrics as numpy arrays
├── train_loss.npy              # Training loss curve
├── training.log                # Console output log
└── train_sindy_coefs.npz       # SINDy coefficients
```

---

## Loss Functions

LaSDIc uses a multi-objective loss with independent weighting and ramp-up scheduling:

| Category | Loss | Description |
|----------|------|-------------|
| **Reconstruction** | `recW` | W-space MSE between input and AE output |
| | `FSE` | Fraction Sum Error — penalizes deviation from sum=1 |
| | `frac` | Fraction MSE — primary reconstruction metric |
| | `ion` | Ion distribution match (charge state populations) |
| | `zbar` | Mean charge (Zbar) match |
| **Dynamics** | `sindy` | dZ/dt residual MSE in latent space |
| | `coef` | Coefficient regularization (L2 norm) |
| **Stability** | `hurwitz` | Penalty for positive eigenvalues (lstsq only) |
| **Equilibrium** | `steady` | Steady-state Z* prediction accuracy |
| **Rate Equations** | `rate_W/N/CSD/Zbar` | Time derivative consistency in physical space |

Each loss weight can be independently ramped up from 0 to its target value over a configurable number of epochs using exponential, linear, or cosine schedules.

### Loss Mode System

Each loss term can be independently set to one of three operation modes via `loss_mode` in `TrainConfig`:

| Mode | Compute | Gradient | In `loss_total` | Epoch Log | Tag |
|------|---------|----------|-----------------|-----------|-----|
| `"train"` | Yes | Yes | Yes | value shown | (none) |
| `"monitor"` | Yes (`no_grad`) | No | No | value shown | `[M]` |
| `"off"` | No | No | No | `0.00e+00` | `[X]` |

**`"train"`** — The loss is computed with gradient tracking and included in backpropagation. This is the standard training behavior.

**`"monitor"`** — The loss is computed inside `torch.no_grad()` for logging/monitoring purposes only. The value appears in epoch logs but does not affect model weights. Useful for tracking metrics without paying the backprop cost.

**`"off"`** — The loss is not computed at all. No GPU time is spent. The value is reported as zero in logs. Use this to completely disable expensive loss terms that are not needed.

Example configuration:

```python
loss_mode={
    "rec":      "train",      # Always train reconstruction
    "fse":      "train",
    "frac":     "train",
    "ion":      "train",
    "zbar":     "monitor",    # Track Zbar but don't backprop
    "sindy":    "train",
    "coef":     "train",
    "hurwitz":  "monitor",    # Monitor eigenvalues only
    "steady":   "train",
    "rate_W":   "off",        # Skip entirely
    "rate_N":   "off",
    "rate_CSD": "monitor",    # Track CSD rate for diagnostics
    "rate_Zbar":"off",
}
```

---


## Author

**Jeongwoo Nam**  
M.S. / Ph.D. Integrated Student  
High Energy Density Physics Laboratory  
Department of Physics and Photon Science  
Gwangju Institute of Science and Technology

123 Cheomdan-gwagiro, Buk-gu, Gwangju, 61005, Republic of Korea

Email: jeongwoo.nam@gm.gist.ac.kr (primary) | njwo1342@gmail.com

---

## Acknowledgement

This project is based on GPLaSDI, originally developed at Lawrence Livermore National Laboratory (LLNL).  
Original GPLaSDI citation:

> Bonneville, C., Choi, Y., Ghosh, D., & Belof, J. L. (2023). *GPLaSDI: Gaussian Process-based Interpretable Latent Space Dynamics Identification through Deep Autoencoder.* arXiv preprint.
