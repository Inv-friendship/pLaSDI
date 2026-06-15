#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaSDIc Noise Evaluation
=======================

Response-letter style robustness test.

Metrics are aligned with the manuscript:
  1. Fractional population RMSE over all states
  2. CSD RMSE over all charge states
  3. Mean charge state relative error (%)
  4. Conservation deviation: mean |sum_i n_i - 1|

Noise targets:
  - control: T(t), rho(t)
  - population: all population snapshots used as model input
  - initial: only n(0) per trajectory
  - both: control + population snapshots
"""

#%%
# =============================================================================
# CONFIG - edit here
# =============================================================================

CONFIG_DIR = "./runs/dim3"
BEST_TYPE = "val"  # "train" or "val"

# For response letter, validation/test split is usually the cleanest report.
# You may set ["train", "val"] or ["all"] when needed.
EVALUATE_SPLITS = ["val"]

NOISE_TARGETS = ["control", "population", "initial", "both"]
NOISE_LEVELS = [0.005, 0.01, 0.05, 0.10]  # 0.5%, 1%, 5%, 10%
INCLUDE_STRESS_20 = True                   # include 20% stress test in CSV/table
STRESS_LEVELS = [0.20]

NOISE_TRIALS = 5
NOISE_SEED = 42

EVALUATE_MODELS = ["Input", "SINDy"]
PLOT_MODEL = "SINDy"
PLOT_INCLUDE_STRESS_20 = False  # response-letter figure usually stops at 10%

# Linear lstsq SINDy is evaluated many times in a noise study.  The original
# scipy odeint path is accurate but slow; RK4 with piecewise-constant controls
# is much faster and sufficient for robustness sweeps.
FAST_LINEAR_ROLLOUT = True

SAVE_CSV = True
SAVE_FIGURE = True
OUTPUT_DIR = None  # None -> CONFIG_DIR/noise_eval

BATCH_SIZE = 8192


#%%
# =============================================================================
# Imports
# =============================================================================

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import create_default_config
from src.autoencoder import Autoencoder
from src.atomic_physics import AtomicPhysics
from src.data_utils import (
    align_controls,
    build_segment_slices,
    guess_history_path,
    load_history_file,
    load_or_build_pops,
    load_state_names,
    split_train_val_random_segments,
)
from src.scaling import ControlScaler, PopulationScaler, TorchScaleHelper

print("[Init] Imports complete")


#%%
# =============================================================================
# Helpers
# =============================================================================

def load_config(config_dir: str):
    config_path = Path(config_dir) / "config.py"
    if config_path.exists():
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("saved_config", str(config_path))
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.create_default_config()
        cfg.save.save_root = config_dir
        cfg.__post_init__()
        print(f"[Config] Loaded from {config_path}")
    else:
        cfg = create_default_config()
        cfg.save.save_root = config_dir
        cfg.__post_init__()
        print("[Config] No saved config found, using default config")
    return cfg


def flatten_decoder_output(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x[:, 0, 0, :]
    if x.dim() == 3:
        return x[:, 0, :]
    return x


def iter_chunks(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def add_relative_noise(arr: np.ndarray, level: float, rng: np.random.Generator,
                       floor: float) -> np.ndarray:
    if level <= 0:
        return arr.copy()
    noisy = arr * (1.0 + rng.normal(loc=0.0, scale=level, size=arr.shape))
    return np.clip(noisy, floor, None)


def metric_values(truth_frac: np.ndarray, pred_frac: np.ndarray,
                  ap: AtomicPhysics) -> dict:
    diff = pred_frac - truth_frac
    n_rmse = float(np.sqrt(np.mean(diff ** 2)))

    pred_sum = np.sum(pred_frac, axis=1)
    conservation = float(np.mean(np.abs(pred_sum - 1.0)))
    conservation_max = float(np.max(np.abs(pred_sum - 1.0)))

    out = {
        "n_rmse": n_rmse,
        "conservation": conservation,
        "conservation_max": conservation_max,
    }

    if ap.ion_available:
        truth_csd = ap.compute_csd_numpy(truth_frac)
        pred_csd = ap.compute_csd_numpy(pred_frac)
        out["csd_rmse"] = float(np.sqrt(np.mean((pred_csd - truth_csd) ** 2)))

        truth_qbar = ap.compute_zbar_numpy(truth_frac)
        pred_qbar = ap.compute_zbar_numpy(pred_frac)
        denom = np.maximum(np.abs(truth_qbar), 1e-30)
        out["qbar_rel_err_pct"] = float(np.mean(np.abs(pred_qbar - truth_qbar) / denom) * 100.0)
        out["qbar_rmse"] = float(np.sqrt(np.mean((pred_qbar - truth_qbar) ** 2)))
    else:
        out["csd_rmse"] = np.nan
        out["qbar_rel_err_pct"] = np.nan
        out["qbar_rmse"] = np.nan

    return out


def write_csv(path: Path, rows: list, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list) -> list:
    key_fields = ["split", "target", "level", "level_pct", "model"]
    metrics = ["n_rmse", "csd_rmse", "qbar_rel_err_pct", "qbar_rmse", "conservation", "conservation_max"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in key_fields)].append(row)

    summary = []
    for key, group in grouped.items():
        out = dict(zip(key_fields, key))
        out["trials"] = len(group)
        for m in metrics:
            vals = np.array([float(g[m]) for g in group], dtype=float)
            out[f"{m}_mean"] = float(np.nanmean(vals))
            out[f"{m}_std"] = float(np.nanstd(vals))
        summary.append(out)

    def sort_key(r):
        target_order = {"none": 0, "control": 1, "population": 2, "initial": 3, "both": 4}
        model_order = {"Input": 0, "AE": 1, "SINDy": 2}
        return (
            r["split"],
            target_order.get(r["target"], 99),
            float(r["level"]),
            model_order.get(r["model"], 99),
        )

    return sorted(summary, key=sort_key)


def fmt_pm(mean, std, fmt=".3e"):
    return f"{format(mean, fmt)} +/- {format(std, fmt)}"


def print_response_table(summary: list, split: str, model: str):
    rows = [r for r in summary if r["split"] == split and r["model"] == model]
    if not rows:
        return

    print("\n" + "=" * 118)
    print(f"[Response Table] split={split}, model={model}")
    print("=" * 118)
    print(
        f"{'Target':<12} {'Level':>8} {'n RMSE':>23} {'CSD RMSE':>23} "
        f"{'qbar rel err (%)':>23} {'Conservation':>23}"
    )
    print("-" * 118)
    for r in rows:
        print(
            f"{r['target']:<12} {float(r['level_pct']):>7.1f}% "
            f"{fmt_pm(r['n_rmse_mean'], r['n_rmse_std']):>23} "
            f"{fmt_pm(r['csd_rmse_mean'], r['csd_rmse_std']):>23} "
            f"{fmt_pm(r['qbar_rel_err_pct_mean'], r['qbar_rel_err_pct_std'], '.3f'):>23} "
            f"{fmt_pm(r['conservation_mean'], r['conservation_std']):>23}"
        )


def make_noise_figure(summary: list, split: str, model: str, out_path: Path):
    metric_specs = [
        ("n_rmse", "Fractional population RMSE"),
        ("csd_rmse", "CSD RMSE"),
        ("qbar_rel_err_pct", "Mean charge rel. error (%)"),
        ("conservation", "Conservation deviation"),
    ]
    targets = [t for t in NOISE_TARGETS if t != "none"]
    colors = {
        "control": "tab:blue",
        "population": "tab:orange",
        "initial": "tab:green",
        "both": "tab:red",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160, constrained_layout=True)
    axes = axes.ravel()

    baseline = {
        m: next(
            (r for r in summary if r["split"] == split and r["model"] == model
             and r["target"] == "none" and abs(float(r["level"])) < 1e-15),
            None,
        )
        for m, _ in metric_specs
    }

    max_plot_level = 0.20 if PLOT_INCLUDE_STRESS_20 else 0.10

    for ax, (metric, title) in zip(axes, metric_specs):
        for target in targets:
            rows = [
                r for r in summary
                if r["split"] == split and r["model"] == model and r["target"] == target
                and float(r["level"]) <= max_plot_level + 1e-15
            ]
            rows = sorted(rows, key=lambda r: float(r["level"]))
            if not rows:
                continue

            x = [0.0]
            y = [baseline[metric][f"{metric}_mean"] if baseline[metric] else np.nan]
            s = [baseline[metric][f"{metric}_std"] if baseline[metric] else 0.0]
            for r in rows:
                x.append(float(r["level_pct"]))
                y.append(float(r[f"{metric}_mean"]))
                s.append(float(r[f"{metric}_std"]))

            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
            s = np.array(s, dtype=float)
            ax.plot(x, y, marker="o", label=target, color=colors.get(target))
            ax.fill_between(x, y - s, y + s, alpha=0.18, color=colors.get(target))

        ax.set_title(title)
        ax.set_xlabel("Noise level (%)")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Noise robustness ({split}, {model})", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


#%%
# =============================================================================
# Load Data
# =============================================================================

cfg = load_config(CONFIG_DIR)
device = cfg.get_device()
dtype = cfg.dtype
torch.set_default_dtype(dtype)

out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else Path(CONFIG_DIR) / "noise_eval"
out_dir.mkdir(parents=True, exist_ok=True)

all_levels = list(NOISE_LEVELS) + (list(STRESS_LEVELS) if INCLUDE_STRESS_20 else [])
all_levels = sorted(set(float(x) for x in all_levels))

print(f"[Device] {device}, dtype={dtype}")
print(f"[Output] {out_dir}")
print(f"[Noise] levels={[100*x for x in all_levels]}%, trials={NOISE_TRIALS}")

print("\n[Data] Loading population and control data...")
pops = load_or_build_pops(cfg.data_files, cfg.data.nx, cfg.data.data_dir)
pops = [np.asarray(p, dtype=np.float64) for p in pops]
pop_clean = np.concatenate(pops, axis=0)

tmp = np.sum(pop_clean, axis=1, keepdims=True)
pop_clean = pop_clean / tmp + cfg.data.pop_lim
pop_clean = pop_clean * tmp
nA_all = np.sum(pop_clean, axis=1, keepdims=True)
truth_frac_all = pop_clean / (nA_all + 1e-300)

pop_scaler = PopulationScaler(eps=cfg.data.pop_lim, normalize=True)
W_clean_all = pop_scaler.fit_transform(pop_clean, axis=1)
scale_helper = TorchScaleHelper(pop_scaler, dtype)

U_segments = []
for i, f in enumerate(cfg.data_files):
    hpath = cfg.history_files[i] if i < len(cfg.history_files) else guess_history_path(f)
    if hpath is None or not os.path.exists(hpath):
        raise FileNotFoundError(f"History not found for {f}")
    t_h, U_h = load_history_file(hpath)
    L = pops[i].shape[0]
    t_seg = np.arange(L) * cfg.data.dt
    if len(t_h) != L:
        U_h = align_controls(t_h, U_h, t_seg)
    U_segments.append(U_h.astype(np.float64))

U_raw_clean_all = np.concatenate(U_segments, axis=0)
ctrl_scaler = ControlScaler(eps=1e-300)
U_clean_all = ctrl_scaler.fit_transform(U_raw_clean_all)
mu = U_clean_all.shape[1]

segment_slices = build_segment_slices(pops)
(train_idx, val_idx, train_slices, val_slices,
 train_seg_ids, val_seg_ids) = split_train_val_random_segments(
    segment_slices, cfg.train.val_ratio, seed=cfg.seed
)

split_seg_ids = {
    "train": train_seg_ids,
    "val": val_seg_ids,
    "all": list(range(len(segment_slices))),
}

for split_name in EVALUATE_SPLITS:
    if split_name not in split_seg_ids:
        raise ValueError(f"Unknown split: {split_name}. Use train, val, or all.")

state_names = load_state_names(cfg.data.names_file, cfg.data.nx)
ap = AtomicPhysics(state_names, cfg.data.nx, dtype)

print(f"[Data] Loaded {len(segment_slices)} segments, {W_clean_all.shape[0]} timesteps, mu={mu}")
print(f"       train segments={len(train_seg_ids)}, val segments={len(val_seg_ids)}")


#%%
# =============================================================================
# Load Model
# =============================================================================

print("\n[Model] Loading...")
ae = Autoencoder(
    nx=cfg.data.nx,
    latent_dim=cfg.model.latent_dim,
    hidden_units=cfg.model.hidden,
    activation=cfg.model.activation,
).to(device, dtype=dtype)

ckpt_path = cfg.ckpt_val_best_path if BEST_TYPE == "val" else cfg.ckpt_train_best_path
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
ae.load_state_dict(ckpt["model_state"])
ae.eval()
print(f"[Model] AE loaded from {ckpt_path} (epoch={ckpt.get('epoch', '?')})")

dt_eff = cfg.sindy.dt_eff if cfg.sindy.dt_eff else cfg.data.dt
use_adaptive_sindy = cfg.sindy.use_adaptive
sindy_model = None
ld = None
coef_vec_np = None
coef_a = None
coef_A = None
coef_B = None

if use_adaptive_sindy:
    from src.sindyc_adaptive import AdaptiveSINDyC
    sindy_model = AdaptiveSINDyC(
        nz=cfg.model.latent_dim,
        mu=mu,
        hidden_dims=cfg.sindy.adaptive_hidden,
        activation=cfg.sindy.adaptive_activation,
        fd_type=cfg.sindy.fd_type,
        eps=cfg.sindy.adaptive_eps,
        symmetric=cfg.sindy.adaptive_symmetric,
        head_gain=cfg.sindy.adaptive_head_gain,
    ).to(device, dtype=dtype)
    if "sindy_model_state" not in ckpt:
        raise RuntimeError("Adaptive SINDy checkpoint has no 'sindy_model_state'")
    sindy_model.load_state_dict(ckpt["sindy_model_state"])
    sindy_model.eval()
    print("[Model] AdaptiveSINDyC loaded")
else:
    from src.sindyc import SINDyC
    ld = SINDyC(
        dim=cfg.model.latent_dim,
        nt=max(len(train_idx), 10),
        fd_type=cfg.sindy.fd_type,
        use_global_coefs=cfg.sindy.use_global_coefs,
    )
    ld._set_mu(mu)
    print("[Model] SINDyC lstsq ready")

    precomputed_path = Path(CONFIG_DIR) / "precomputed.npz"
    coef_key = "sindy_coef_vec_val" if BEST_TYPE == "val" else "sindy_coef_vec_train"
    if precomputed_path.exists():
        pc = np.load(str(precomputed_path), allow_pickle=True)
        if coef_key in pc:
            coef_vec_np = pc[coef_key].reshape(-1)
            print(f"[SINDy] Coefficients loaded from precomputed.npz ({coef_key})")
        elif "sindy_coef_vec_train" in pc:
            coef_vec_np = pc["sindy_coef_vec_train"].reshape(-1)
            print("[SINDy] Coefficients loaded from precomputed.npz (train fallback)")

    if coef_vec_np is None:
        print("[SINDy] Calibrating lstsq coefficients from clean train segments...")
        Z_train_list, U_train_list = [], []
        for sid in train_seg_ids:
            sl = segment_slices[sid]
            with torch.no_grad():
                X_t = torch.tensor(W_clean_all[sl][:, None, None, :], dtype=dtype, device=device)
                Z_t = flatten_decoder_output(ae.encoder(X_t))
            Z_train_list.append(Z_t.cpu().numpy())
            U_train_list.append(U_clean_all[sl])
        Z_train = np.vstack(Z_train_list)
        U_train = np.vstack(U_train_list)
        with torch.no_grad():
            coef_vec = ld.calibrate(
                torch.tensor(Z_train, dtype=dtype, device=device),
                torch.tensor(U_train, dtype=dtype, device=device),
                float(dt_eff),
                compute_loss=False,
                numpy=False,
            )
        coef_vec_np = coef_vec.detach().cpu().numpy().reshape(-1)

    C = coef_vec_np.reshape(-1, cfg.model.latent_dim)
    coef_a = C[0, :]
    coef_A = C[1:1 + cfg.model.latent_dim, :]       # z @ A
    coef_B = C[1 + cfg.model.latent_dim:, :]        # u @ B


#%%
# =============================================================================
# Prediction helpers
# =============================================================================

def W_to_fraction_np(W: np.ndarray) -> np.ndarray:
    out = []
    with torch.no_grad():
        for ch in iter_chunks(W.shape[0], BATCH_SIZE):
            W_t = torch.tensor(W[ch], dtype=dtype, device=device)
            F_t = scale_helper.W_to_fraction(W_t.unsqueeze(1).unsqueeze(1))
            out.append(F_t.cpu().numpy().reshape(-1, cfg.data.nx))
    return np.vstack(out)


def ae_predict_fraction(W: np.ndarray):
    W_pred_parts, F_pred_parts, Z_parts = [], [], []
    with torch.no_grad():
        for ch in iter_chunks(W.shape[0], BATCH_SIZE):
            X_t = torch.tensor(W[ch, None, None, :], dtype=dtype, device=device)
            Z_t = flatten_decoder_output(ae.encoder(X_t))
            W_pred_t = flatten_decoder_output(ae.decoder(Z_t))
            F_pred_t = scale_helper.W_to_fraction(W_pred_t.unsqueeze(1).unsqueeze(1))
            W_pred_parts.append(W_pred_t.cpu().numpy())
            F_pred_parts.append(F_pred_t.cpu().numpy().reshape(-1, cfg.data.nx))
            Z_parts.append(Z_t.cpu().numpy())
    return np.vstack(F_pred_parts), np.vstack(W_pred_parts), np.vstack(Z_parts)


def encode_first(W0: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        X_t = torch.tensor(W0[None, None, None, :], dtype=dtype, device=device)
        Z_t = flatten_decoder_output(ae.encoder(X_t))
    return Z_t.cpu().numpy()[0]


def encode_batch(W: np.ndarray) -> np.ndarray:
    Z_parts = []
    with torch.no_grad():
        for ch in iter_chunks(W.shape[0], BATCH_SIZE):
            X_t = torch.tensor(W[ch, None, None, :], dtype=dtype, device=device)
            Z_t = flatten_decoder_output(ae.encoder(X_t))
            Z_parts.append(Z_t.cpu().numpy())
    return np.vstack(Z_parts)


def decode_Z_to_fraction(Z: np.ndarray) -> np.ndarray:
    F_parts = []
    with torch.no_grad():
        for ch in iter_chunks(Z.shape[0], BATCH_SIZE):
            Z_t = torch.tensor(Z[ch], dtype=dtype, device=device)
            W_pred_t = flatten_decoder_output(ae.decoder(Z_t))
            F_pred_t = scale_helper.W_to_fraction(W_pred_t.unsqueeze(1).unsqueeze(1))
            F_parts.append(F_pred_t.cpu().numpy().reshape(-1, cfg.data.nx))
    return np.vstack(F_parts)


def sindy_rollout_Z(z0: np.ndarray, U_seg: np.ndarray, L: int) -> np.ndarray:
    if use_adaptive_sindy:
        U_seg_t = torch.tensor(U_seg, dtype=dtype, device=device)
        t_grid = np.linspace(0.0, (L - 1) * dt_eff, L)

        def rhs(t, z):
            t_idx = int(min(max(t / dt_eff, 0.0), L - 1))
            with torch.no_grad():
                a_t, A_t = sindy_model.get_coefficients_batch(U_seg_t[t_idx:t_idx + 1])
            a_np = a_t.squeeze(0).cpu().numpy()
            A_np = A_t.squeeze(0).cpu().numpy()
            return a_np + z @ A_np.T

        sol = solve_ivp(rhs, [0.0, (L - 1) * dt_eff], z0, t_eval=t_grid, method="RK45")
        Z_pred = sol.y.T
    else:
        if FAST_LINEAR_ROLLOUT:
            Z_pred = np.zeros((L, cfg.model.latent_dim), dtype=np.float64)
            Z_pred[0] = z0

            def rhs(z, u):
                return coef_a + z @ coef_A + u @ coef_B

            h = float(dt_eff)
            for k in range(L - 1):
                u = U_seg[k]
                z = Z_pred[k]
                k1 = rhs(z, u)
                k2 = rhs(z + 0.5 * h * k1, u)
                k3 = rhs(z + 0.5 * h * k2, u)
                k4 = rhs(z + h * k3, u)
                Z_pred[k + 1] = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            t_grid = np.linspace(0.0, (L - 1) * dt_eff, L)
            Z_pred = ld.simulate(coef_vec_np, z0, t_grid, U=U_seg)
    return Z_pred


def make_noisy_segment(sid: int, target: str, level: float, rng: np.random.Generator):
    sl = segment_slices[sid]
    pop_seg = pop_clean[sl].copy()
    U_raw_seg = U_raw_clean_all[sl].copy()

    if target in ("population", "both") and level > 0:
        pop_seg = add_relative_noise(pop_seg, level, rng, floor=cfg.data.pop_lim)
    elif target == "initial" and level > 0:
        pop_seg[0:1] = add_relative_noise(pop_seg[0:1], level, rng, floor=cfg.data.pop_lim)

    if target in ("control", "both") and level > 0:
        U_raw_seg = add_relative_noise(U_raw_seg, level, rng, floor=1e-300)

    W_seg = pop_scaler.transform(pop_seg)
    U_seg = ctrl_scaler.transform(U_raw_seg)
    return W_seg, U_seg


def evaluate_one(split: str, target: str, level: float, trial: int):
    rng = np.random.default_rng(NOISE_SEED + 1_000_000 * trial + int(level * 1e6) + hash(target) % 100_000)
    seg_ids = split_seg_ids[split]

    truth_parts = []
    W_parts = []
    U_parts = []
    lengths = []

    for sid in seg_ids:
        sl = segment_slices[sid]
        L = sl.stop - sl.start
        lengths.append(L)
        truth_parts.append(truth_frac_all[sl])

        W_seg, U_seg = make_noisy_segment(sid, target, level, rng)
        W_parts.append(W_seg)
        U_parts.append(U_seg)

    W_eval = np.vstack(W_parts)
    U_eval = np.vstack(U_parts)
    truth_frac = np.vstack(truth_parts)

    pred_parts = {}

    if "Input" in EVALUATE_MODELS:
        pred_parts["Input"] = W_to_fraction_np(W_eval)

    Z_eval = None
    Z_initials = None
    if "AE" in EVALUATE_MODELS:
        F_ae, _, Z_eval = ae_predict_fraction(W_eval)
        pred_parts["AE"] = F_ae
    elif "SINDy" in EVALUATE_MODELS:
        W0 = np.vstack([w[0:1] for w in W_parts])
        Z_initials = encode_batch(W0)

    if "SINDy" in EVALUATE_MODELS:
        sindy_Z_parts = []
        offset = 0
        for i, L in enumerate(lengths):
            z0 = Z_eval[offset] if Z_eval is not None else Z_initials[i]
            U_seg = U_eval[offset:offset + L]
            sindy_Z_parts.append(sindy_rollout_Z(z0, U_seg, L))
            offset += L
        pred_parts["SINDy"] = decode_Z_to_fraction(np.vstack(sindy_Z_parts))

    rows = []
    for model in EVALUATE_MODELS:
        pred_frac = pred_parts[model]
        metric = metric_values(truth_frac, pred_frac, ap)
        row = {
            "split": split,
            "target": target,
            "level": level,
            "level_pct": 100.0 * level,
            "trial": trial,
            "model": model,
        }
        row.update(metric)
        rows.append(row)
    return rows


#%%
# =============================================================================
# Run
# =============================================================================

jobs = [("none", 0.0, 0)]
for target in NOISE_TARGETS:
    for level in all_levels:
        for trial in range(NOISE_TRIALS):
            jobs.append((target, level, trial))

print("\n" + "=" * 80)
print("[Run] Noise evaluation")
print("=" * 80)
print(f"Jobs={len(jobs)}, splits={EVALUATE_SPLITS}, models={EVALUATE_MODELS}, best={BEST_TYPE}")

raw_rows = []
for j, (target, level, trial) in enumerate(jobs, start=1):
    print(f"[{j:03d}/{len(jobs):03d}] target={target:<10} level={100*level:5.1f}% trial={trial}")
    for split in EVALUATE_SPLITS:
        raw_rows.extend(evaluate_one(split, target, level, trial))

summary_rows = summarize_rows(raw_rows)

raw_fields = [
    "split", "target", "level", "level_pct", "trial", "model",
    "n_rmse", "csd_rmse", "qbar_rel_err_pct", "qbar_rmse",
    "conservation", "conservation_max",
]
summary_fields = [
    "split", "target", "level", "level_pct", "model", "trials",
    "n_rmse_mean", "n_rmse_std",
    "csd_rmse_mean", "csd_rmse_std",
    "qbar_rel_err_pct_mean", "qbar_rel_err_pct_std",
    "qbar_rmse_mean", "qbar_rmse_std",
    "conservation_mean", "conservation_std",
    "conservation_max_mean", "conservation_max_std",
]

raw_path = out_dir / f"noise_response_metrics_{BEST_TYPE}.csv"
summary_path = out_dir / f"noise_response_summary_{BEST_TYPE}.csv"
write_csv(raw_path, raw_rows, raw_fields)
write_csv(summary_path, summary_rows, summary_fields)

print(f"\n[Saved] Raw metrics:     {raw_path}")
print(f"[Saved] Summary metrics: {summary_path}")

for split in EVALUATE_SPLITS:
    if PLOT_MODEL in EVALUATE_MODELS:
        print_response_table(summary_rows, split, PLOT_MODEL)
    for model in ("Input", "AE"):
        if model in EVALUATE_MODELS:
            print_response_table(summary_rows, split, model)

if SAVE_FIGURE and PLOT_MODEL in EVALUATE_MODELS:
    for split in EVALUATE_SPLITS:
        fig_path = out_dir / f"noise_response_{split}_{PLOT_MODEL}_{BEST_TYPE}.png"
        make_noise_figure(summary_rows, split, PLOT_MODEL, fig_path)
        print(f"[Saved] Figure: {fig_path}")

print("\n[Done] Noise evaluation complete")

# %%
