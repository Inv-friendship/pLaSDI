# -*- coding: utf-8 -*-
"""
LaSDIc Evaluation v2.2 - Enhanced Edition
==========================================
Comprehensive evaluation script based on v2.1.

Additional features:
- Visualize weight ramp-up curves.
- Visualize learning-rate curves.
- Detailed loss breakdown by category.

Usage:
    1. Open this file in VSCode.
    2. Edit the CONFIG section below.
    3. Execute cells with Shift+Enter or run the whole file.
"""

#%%
# =============================================================================
# CONFIG - edit only this section
# =============================================================================

# Path settings
CONFIG_DIR = "./runs/dim3"  # Model checkpoint directory

# Best model selection: "train" or "val"
BEST_TYPE = "train"          # Best model type used for evaluation

# Visualization options
SHOW_PLOTS = True           # Show plots
SAVE_PLOTS = False          # Save plots
DPI = 150                   # Resolution

# Scale settings
CSD_SCALE = 'linear'        # 'linear' or 'log' (CSD heatmap)
FRACTION_SCALE = 'log'      # 'linear' or 'log' (Fraction heatmap)
POPULATION_SCALE = 'linear' # 'linear' or 'log' (Population heatmap)

# Segment settings specified by case number, e.g. seg4.txt -> 4
# None means the first 3; a list selects those case numbers
SELECTED_CASE_NUMBERS = None    # Example: [4, 10, 25] or None

# Visualization section settings (True/False)
VIS_SECTIONS = {
    'training_curves': True,       # Training curves
    'weight_curves': True,         # Weight ramp-up curves (NEW)
    'lr_curve': True,              # Learning-rate curves (NEW)
    'comprehensive_metrics': True, # Comprehensive metrics (Train/Val)
    'state_by_state': True,        # State-by-State RMSE/MRE
    'heatmaps': True,              # Heatmaps (W, Frac, Pop, CSD, Error)
    'line_plots': True,            # Line plots (snapshots)
    'physics': True,               # Zbar and ion analysis
    'latent': True,                # Z space
    'sindy': True,                 # SINDy analysis (lstsq mode)
    'adaptive_sindy': True,        # Adaptive SINDy analysis (CoefNet mode)
    'steady': True,                # Steady-state (AE + Equilibrium)
    'steady_rollout': True,        # Steady-state rollout (time extension)
    'benchmark': True,             # Inference performance
    'rate_equation': True,         # Rate equation analysis
}

# Adaptive SINDy visualization settings
ADAPTIVE_EXTRAP_FACTOR = 1.5     # Extrapolation expansion ratio relative to the training range
ADAPTIVE_GRID_RESOLUTION = 50   # Heatmap resolution

# Steady-state rollout settings
ROLLOUT_CASE_NUMBER = None       # If None, use the first case; otherwise use the specified case
ROLLOUT_EXTEND_NS = 5.0          # Extension time (ns)

# Activated metrics threshold (None uses config values)
FRAC_THRESHOLD = None            # Example: 1e-5 (fraction activation threshold)
CSD_THRESHOLD = None             # Example: 1e-5 (CSD activation threshold)

#%%
# =============================================================================
# Setup & Imports
# =============================================================================

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Non-blocking plot: continue in terminal without closing figure windows
def show_plot():
    """Show non-blocking plots when SHOW_PLOTS is True; otherwise close them."""
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.05)
    else:
        plt.close()
from matplotlib.colors import LogNorm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Project modules
from config import LaSDIcConfig, create_default_config
from src.scaling import PopulationScaler, ControlScaler, TorchScaleHelper
from src.atomic_physics import AtomicPhysics
from src.data_utils import *
from src.train_utils import csv_to_npy, metrics_csv_to_npz
from src import visualization as viz

print("✅ Imports complete")

#%%
# =============================================================================
# Load Configuration
# =============================================================================

# Prefer the config.py backed up inside CONFIG_DIR when available
_config_in_dir = Path(CONFIG_DIR) / "config.py"
if _config_in_dir.exists():
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("saved_config", str(_config_in_dir))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = _mod.create_default_config()
    cfg.save.save_root = CONFIG_DIR
    cfg.__post_init__()
    print(f"✅ Config loaded from {_config_in_dir}")
else:
    cfg = create_default_config()
    cfg.save.save_root = CONFIG_DIR
    cfg.__post_init__()
    print(f"⚠️  No config.py in {CONFIG_DIR}, using default config")

device = cfg.get_device()
dtype = cfg.dtype
torch.set_default_dtype(dtype)

if SAVE_PLOTS:
    plot_dir = Path(CONFIG_DIR) / "plots"
    plot_dir.mkdir(exist_ok=True)
else:
    plot_dir = None

print(f"   Device: {device}")

#%%
# =============================================================================
# Load Data
# =============================================================================

print("[Data] Loading...")

# Population
pops = load_or_build_pops(cfg.data_files, cfg.data.nx, cfg.data.data_dir)
pops = [np.asarray(p, dtype=np.float64) for p in pops]
pop = np.concatenate(pops, axis=0)

tmp = np.sum(pop, axis=1, keepdims=True)
pop = pop / tmp + cfg.data.pop_lim
pop = pop * tmp

# Scaler
pop_scaler = PopulationScaler(eps=cfg.data.pop_lim, normalize=True)
W_all = pop_scaler.fit_transform(pop, axis=1)
W_all = np.expand_dims(W_all, axis=1)
X_frames = torch.tensor(W_all[:, None, :, :], dtype=dtype)

nt_total = pop.shape[0]
time_axis = np.arange(nt_total) * cfg.data.dt
nA_all = np.sum(pop, axis=1, keepdims=True)

# Control
U_segments = []
for i, f in enumerate(cfg.data_files):
    hpath = cfg.history_files[i] if i < len(cfg.history_files) else guess_history_path(f)
    if not os.path.exists(hpath):
        raise FileNotFoundError(f"History not found: {f}")
    t_h, U_h = load_history_file(hpath)
    L = pops[i].shape[0]
    t_seg = np.arange(L) * cfg.data.dt
    if len(t_h) != L:
        U_h = align_controls(t_h, U_h, t_seg)
    U_segments.append(U_h.astype(np.float64))

U_all_raw = np.concatenate(U_segments, axis=0)
ctrl_scaler = ControlScaler(eps=1e-300)
U_all = ctrl_scaler.fit_transform(U_all_raw)
mu = U_all_raw.shape[1]

scale_helper = TorchScaleHelper(pop_scaler, dtype)

# Segments
segment_slices = build_segment_slices(pops)
case_numbers = cfg.case_numbers

(train_idx, val_idx, train_slices, val_slices,
 train_seg_ids, val_seg_ids) = split_train_val_random_segments(
    segment_slices, cfg.train.val_ratio, seed=cfg.seed
)

# Atomic physics
state_names = load_state_names(cfg.data.names_file, cfg.data.nx)
ap = AtomicPhysics(state_names, cfg.data.nx, dtype)

# Steady-state
steady_data = None
if cfg.data.steady_enable:
    steady_data = SteadyStateData(
        cfg.steady_pop_hist_pairs, pop_scaler, ctrl_scaler,
        random_pick=cfg.data.steady_random_pick,
        num_samples=cfg.data.steady_num_samples,
        seed=cfg.data.steady_random_seed,
        pop_lim=cfg.data.pop_lim
    )

print(f"✅ Data loaded: {nt_total} timesteps, {len(segment_slices)} segments")
print(f"\n{'='*60}")
print(" Train/Val Split")
print(f"{'='*60}")
print(f"Train segments ({len(train_seg_ids)}):")
train_cases = [case_numbers[i] for i in train_seg_ids]
print(f"  Cases: {train_cases}")
print(f"\nVal segments ({len(val_seg_ids)}):")
val_cases = [case_numbers[i] for i in val_seg_ids]
print(f"  Cases: {val_cases}")
print(f"{'='*60}\n")

#%%
# =============================================================================
# Load Model
# =============================================================================

print("[Model] Loading...")

from src.autoencoder import Autoencoder
from src.sindyc import SINDyC

ae = Autoencoder(
    nx=cfg.data.nx,
    latent_dim=cfg.model.latent_dim,
    hidden_units=cfg.model.hidden,
    activation=cfg.model.activation
).to(device, dtype=dtype)

# Select best model path according to BEST_TYPE
if BEST_TYPE == "val":
    _ckpt_path = cfg.ckpt_val_best_path
    _ckpt_label = "Val-best"
else:
    _ckpt_path = cfg.ckpt_train_best_path
    _ckpt_label = "Train-best"

if os.path.exists(_ckpt_path):
    ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
    ae.load_state_dict(ckpt['model_state'])
    best_epoch = ckpt.get('epoch', '?')
    ckpt_best_type = ckpt.get('best_type', 'unknown')
    print(f"✅ Model loaded from {_ckpt_path}")
    print(f"   {_ckpt_label}: epoch {best_epoch}, stored best_type={ckpt_best_type}")
else:
    raise FileNotFoundError(f"Checkpoint not found: {_ckpt_path}")

dt_eff = cfg.sindy.dt_eff if cfg.sindy.dt_eff else cfg.data.dt

# Load SINDy model (adaptive or lstsq)
use_adaptive_sindy = cfg.sindy.use_adaptive
sindy_model = None

if use_adaptive_sindy:
    from src.sindyc_adaptive import AdaptiveSINDyC
    
    sindy_model = AdaptiveSINDyC(
        nz=cfg.model.latent_dim,
        mu=2,  # T, density
        hidden_dims=cfg.sindy.adaptive_hidden,
        activation=cfg.sindy.adaptive_activation,
        fd_type=cfg.sindy.fd_type,
        eps=cfg.sindy.adaptive_eps,
        symmetric=cfg.sindy.adaptive_symmetric
    ).to(device, dtype=dtype)
    
    if 'sindy_model_state' in ckpt:
        sindy_model.load_state_dict(ckpt['sindy_model_state'])
        print("✅ AdaptiveSINDyC loaded")
    else:
        print("⚠️ No sindy_model_state in checkpoint, using random init")
    
    sindy_model.eval()
    ld = None
else:
    from src.sindyc import SINDyC
    ld = SINDyC(
        dim=cfg.model.latent_dim,
        nt=len(train_idx),
        fd_type=cfg.sindy.fd_type,
        use_global_coefs=cfg.sindy.use_global_coefs
    )
    ld._set_mu(mu)

ae.eval()
print("✅ Model ready")

#%%
# =============================================================================
# Compute AE Reconstruction
# =============================================================================

print("[AE] Computing reconstruction...")

with torch.no_grad():
    X_dev = X_frames.to(device=device, dtype=dtype)
    Z_truth = ae.encoder(X_dev)
    if Z_truth.dim() == 4:
        Z_truth = Z_truth[:, 0, 0, :]
    elif Z_truth.dim() == 3:
        Z_truth = Z_truth[:, 0, :]
    
    W_recon = ae.decoder(Z_truth)
    if W_recon.dim() == 4:
        W_recon = W_recon[:, 0, 0, :]
    elif W_recon.dim() == 3:
        W_recon = W_recon[:, 0, :]
    
    truth_W = X_dev[:, 0, 0, :].cpu().numpy()
    pred_W_ae = W_recon.cpu().numpy()
    
    truth_frac = scale_helper.W_to_fraction(X_dev).cpu().numpy().reshape(-1, cfg.data.nx)
    pred_frac_ae = scale_helper.W_to_fraction(
        W_recon.unsqueeze(1).unsqueeze(1)
    ).cpu().numpy().reshape(-1, cfg.data.nx)
    
    Z_truth_np = Z_truth.cpu().numpy()

truth_pop = truth_frac * nA_all
pred_pop_ae = pred_frac_ae * nA_all

print("✅ AE reconstruction complete")

#%%
# =============================================================================
# Compute SINDy global coefficients (always use TRAIN data only)
# =============================================================================

print("[SINDy] Computing global coefficients...")
print(f"  Using TRAIN segments ONLY: {train_seg_ids}")
print(f"  (SELECTED_CASE_NUMBERS is for visualization only)")

Z_train_list, U_train_list = [], []
for i in train_seg_ids:
    sl = segment_slices[i]
    Z_train_list.append(Z_truth_np[sl.start:sl.stop])
    U_train_list.append(U_all[sl.start:sl.stop])

Z_train = np.vstack(Z_train_list)
U_train = np.vstack(U_train_list)

print(f"  Train data: {Z_train.shape[0]} timesteps")

# Default settings (unused in adaptive mode)
coef_vec_np = None
a_global, A_global, B_global = None, None, None
is_hurwitz = True
max_real, min_real = -1.0, -1.0

# Check precomputed file
_precomputed_path = Path(CONFIG_DIR) / "precomputed.npz"
_has_precomputed = _precomputed_path.exists()

if use_adaptive_sindy:
    print(f"  Mode: Adaptive SINDy (CoefNet)")
    print(f"  A(U), a(U) are computed dynamically - skipping lstsq calibration")
    is_hurwitz = True
    max_real = -cfg.sindy.adaptive_eps
    min_real = -cfg.sindy.adaptive_eps
else:
    # Try loading from precomputed.npz
    _coef_key = 'sindy_coef_vec_val' if BEST_TYPE == 'val' else 'sindy_coef_vec_train'
    _coef_loaded = False
    
    if _has_precomputed:
        _pc = np.load(str(_precomputed_path), allow_pickle=True)
        if _coef_key in _pc:
            coef_vec_np = _pc[_coef_key]
            _coef_loaded = True
            print(f"  Coefficients loaded from precomputed.npz ({_coef_key})")
        elif 'sindy_coef_vec_train' in _pc:
            coef_vec_np = _pc['sindy_coef_vec_train']
            _coef_loaded = True
            print(f"  Coefficients loaded from precomputed.npz (sindy_coef_vec_train, fallback)")
    
    if not _coef_loaded:
        # Fallback: recompute lstsq
        print(f"  Calibrating lstsq from scratch...")
        with torch.no_grad():
            Z_t = torch.tensor(Z_train, dtype=dtype, device=device)
            U_t = torch.tensor(U_train, dtype=dtype, device=device)
            coef_vec = ld.calibrate(Z_t, U_t, float(dt_eff), compute_loss=False, numpy=False)
            coef_vec_np = coef_vec.detach().cpu().numpy().reshape(-1)
    
    # Split coefficients
    def split_coefs(coef_vec, nz, mu):
        if coef_vec.ndim == 1:
            C = coef_vec.reshape(-1, nz)
        else:
            C = coef_vec
        a = C[0, :]
        A = C[1:1+nz, :].T
        B = C[1+nz:1+nz+mu, :].T if mu > 0 else np.zeros((nz, 0))
        return a, A, B
    
    a_global, A_global, B_global = split_coefs(coef_vec_np, cfg.model.latent_dim, mu)
    
    # Eigenvalues
    eigvals = np.linalg.eigvals(A_global)
    max_real = np.max(eigvals.real)
    min_real = np.min(eigvals.real)
    is_hurwitz = max_real < 0
    
    print(f"   Eigenvalues: max_Re={max_real:.3e}, min_Re={min_real:.3e}")
    print(f"   Hurwitz stable: {is_hurwitz}")

print(f"✅ SINDy setup complete")

#%%
# =============================================================================
# SINDy Simulation
# =============================================================================

print("[SINDy] Simulating dynamics...")

Z_pred_sindy = np.zeros_like(Z_truth_np)

if use_adaptive_sindy:
    # Adaptive: numerical integration is needed because A(U(t)) changes over time
    # Use a simple Euler-style path here; can be improved to RK4 later
    print("  Mode: Adaptive SINDy (Euler integration)")
    
    from scipy.integrate import solve_ivp
    
    for sl in segment_slices:
        L = sl.stop - sl.start
        z0 = Z_truth_np[sl.start]
        U_seg = U_all[sl.start:sl.stop]
        
        # Time grid
        t_grid = np.linspace(0.0, (L-1)*dt_eff, L)
        
        # ODE function for numerical integration
        U_seg_t = torch.tensor(U_seg, dtype=dtype, device=device)
        
        def adaptive_ode(t, z):
            # Find the U index corresponding to time
            t_idx = int(min(t / dt_eff, L-1))
            U_t = U_seg_t[t_idx].unsqueeze(0)
            
            with torch.no_grad():
                a_t, A_t = sindy_model.get_coefficients_batch(U_t)
                a_t = a_t.squeeze().cpu().numpy()
                A_t = A_t.squeeze().cpu().numpy()
            
            dz = a_t + z @ A_t.T
            return dz
        
        # Use solve_ivp
        sol = solve_ivp(adaptive_ode, [0, (L-1)*dt_eff], z0, t_eval=t_grid, method='RK45')
        Z_pred_sindy[sl.start:sl.stop] = sol.y.T
else:
    # lstsq mode: use existing ld.simulate
    for sl in segment_slices:
        L = sl.stop - sl.start
        z0 = Z_truth_np[sl.start]
        U_seg = U_all[sl.start:sl.stop]
        t_grid = np.linspace(0.0, (L-1)*dt_eff, L)
        
        Z_seg = ld.simulate(coef_vec_np, z0, t_grid, U=U_seg)
        Z_pred_sindy[sl.start:sl.stop] = Z_seg

# Decode
with torch.no_grad():
    Z_t = torch.tensor(Z_pred_sindy, dtype=dtype, device=device)
    W_pred_t = ae.decoder(Z_t)
    if W_pred_t.dim() == 4:
        W_pred_t = W_pred_t[:, 0, 0, :]
    elif W_pred_t.dim() == 3:
        W_pred_t = W_pred_t[:, 0, :]
    
    pred_W_sindy = W_pred_t.cpu().numpy()
    pred_frac_sindy = scale_helper.W_to_fraction(
        W_pred_t.unsqueeze(1).unsqueeze(1)
    ).cpu().numpy().reshape(-1, cfg.data.nx)

pred_pop_sindy = pred_frac_sindy * nA_all

print("✅ SINDy simulation complete")

#%%
# =============================================================================
# CSD & Zbar
# =============================================================================

if ap.ion_available:
    print("[Physics] Computing CSD & Zbar...")
    
    truth_csd = ap.compute_csd_numpy(truth_frac)
    pred_csd_ae = ap.compute_csd_numpy(pred_frac_ae)
    pred_csd_sindy = ap.compute_csd_numpy(pred_frac_sindy)
    
    truth_zbar = ap.compute_zbar_numpy(truth_frac)
    pred_zbar_ae = ap.compute_zbar_numpy(pred_frac_ae)
    pred_zbar_sindy = ap.compute_zbar_numpy(pred_frac_sindy)
    
    print("✅ CSD & Zbar computed")
else:
    truth_csd = pred_csd_ae = pred_csd_sindy = None
    truth_zbar = pred_zbar_ae = pred_zbar_sindy = None
    print("⚠️  CSD & Zbar not available (ion index missing)")

#%%
# =============================================================================
# Activated Metrics (MRE, MSE) - Fraction, CSD, Zbar
# =============================================================================

print("\n" + "="*60)
print("[Activated Metrics] Computing MRE/MSE for activated values...")
print("="*60)

# Threshold settings
frac_threshold = FRAC_THRESHOLD if FRAC_THRESHOLD is not None else (getattr(cfg.eval, 'frac_threshold', 1e-5) if hasattr(cfg, 'eval') else 1e-5)
csd_threshold = CSD_THRESHOLD if CSD_THRESHOLD is not None else (getattr(cfg.eval, 'csd_threshold', 1e-5) if hasattr(cfg, 'eval') else 1e-5)

def compute_activated_metrics(truth, pred, threshold=None, name=""):
    """
    Compute activated MRE/MSE.
    
    Args:
        truth: ground truth array
        pred: prediction array
        threshold: None means all values (for Zbar); otherwise only truth >= threshold.
        name: Display name.
    
    Returns:
        dict with mre, mse, n_activated, n_total
    """
    if truth is None or pred is None:
        return None
    
    if threshold is not None:
        mask = np.abs(truth) >= threshold
        t_act = truth[mask]
        p_act = pred[mask]
        n_activated = int(mask.sum())
        n_total = int(mask.size)
    else:
        t_act = truth.flatten()
        p_act = pred.flatten()
        n_activated = t_act.size
        n_total = t_act.size
    
    if n_activated == 0:
        return {'mre': float('nan'), 'mse': float('nan'), 
                'n_activated': 0, 'n_total': n_total, 'name': name}
    
    # MSE
    mse = float(np.mean((t_act - p_act) ** 2))
    
    # MRE (mean relative error)
    denom = np.abs(t_act)
    denom = np.where(denom < 1e-30, 1e-30, denom)  # Avoid zero
    mre = float(np.mean(np.abs(t_act - p_act) / denom))
    
    return {'mre': mre, 'mse': mse, 'n_activated': n_activated, 'n_total': n_total, 'name': name}

# Compute
metrics_activated = {}

# --- Fraction ---
for label, pred_frac in [('AE', pred_frac_ae), ('SINDy', pred_frac_sindy)]:
    m = compute_activated_metrics(truth_frac, pred_frac, threshold=frac_threshold, 
                                   name=f"Fraction_{label}")
    if m:
        metrics_activated[f"frac_{label}"] = m

# --- CSD ---
if truth_csd is not None:
    for label, pred_csd in [('AE', pred_csd_ae), ('SINDy', pred_csd_sindy)]:
        m = compute_activated_metrics(truth_csd, pred_csd, threshold=csd_threshold,
                                       name=f"CSD_{label}")
        if m:
            metrics_activated[f"csd_{label}"] = m

# --- Zbar (no threshold) ---
if truth_zbar is not None:
    for label, pred_zbar in [('AE', pred_zbar_ae), ('SINDy', pred_zbar_sindy)]:
        m = compute_activated_metrics(truth_zbar, pred_zbar, threshold=None,
                                       name=f"Zbar_{label}")
        if m:
            metrics_activated[f"zbar_{label}"] = m

# Print
print(f"\n  Thresholds: fraction={frac_threshold:.1e}, CSD={csd_threshold:.1e}, Zbar=all")
print(f"  {'Metric':<20s} {'MRE (%)':>12s} {'MSE':>12s} {'Activated':>12s} {'Total':>12s}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for key, m in metrics_activated.items():
    print(f"  {m['name']:<20s} {m['mre']*100:>11.2f}% {m['mse']:>12.4e} {m['n_activated']:>12d} {m['n_total']:>12d}")

# Save CSV
activated_csv_path = Path(CONFIG_DIR) / "activated_metrics.csv"
with open(activated_csv_path, "w", encoding="utf-8") as f:
    f.write("metric,mre,mse,n_activated,n_total,threshold\n")
    for key, m in metrics_activated.items():
        th = frac_threshold if 'frac' in key else (csd_threshold if 'csd' in key else 'all')
        f.write(f"{m['name']},{m['mre']:.12e},{m['mse']:.12e},{m['n_activated']},{m['n_total']},{th}\n")
print(f"\n✅ Activated metrics saved to {activated_csv_path}")

#%%
# =============================================================================
# Steady-State Equilibrium Analysis
# =============================================================================

if steady_data and steady_data.enabled:
    print("[Steady] Computing equilibrium...")
    
    if use_adaptive_sindy:
        # Adaptive: Z* = -A(U)^{-1} * a(U)
        try:
            steady_W_t, steady_U_t = steady_data.to_torch(device, dtype)
            
            with torch.no_grad():
                # Z* from adaptive model
                Z_star_t = sindy_model.get_equilibrium_batch(steady_U_t)
                Z_star = Z_star_t.cpu().numpy()
                
                # Decode Equilibrium
                W_pred = ae.decoder(Z_star_t)
                if W_pred.dim() == 4:
                    W_pred = W_pred[:, 0, 0, :]
                elif W_pred.dim() == 3:
                    W_pred = W_pred[:, 0, :]
                
                pred_W_steady_eq = W_pred.cpu().numpy()
                pred_F_steady_eq = scale_helper.W_to_fraction(
                    W_pred.unsqueeze(1).unsqueeze(1)
                ).cpu().numpy().reshape(-1, cfg.data.nx)
                
                # AE reconstruction
                W_4d = steady_W_t.unsqueeze(1).unsqueeze(1)
                Z_steady_ae = ae.encoder(W_4d)
                if Z_steady_ae.dim() == 4:
                    Z_steady_ae = Z_steady_ae[:, 0, 0, :]
                elif Z_steady_ae.dim() == 3:
                    Z_steady_ae = Z_steady_ae[:, 0, :]
                
                W_steady_recon = ae.decoder(Z_steady_ae)
                if W_steady_recon.dim() == 4:
                    W_steady_recon = W_steady_recon[:, 0, 0, :]
                elif W_steady_recon.dim() == 3:
                    W_steady_recon = W_steady_recon[:, 0, :]
                
                pred_W_steady_ae = W_steady_recon.cpu().numpy()
                pred_F_steady_ae = scale_helper.W_to_fraction(
                    W_steady_recon.unsqueeze(1).unsqueeze(1)
                ).cpu().numpy().reshape(-1, cfg.data.nx)
                
                # dZ/dt = a + A*Z should be ~0 at equilibrium
                a_ss, A_ss = sindy_model.get_coefficients_batch(steady_U_t)
                dz_dt = a_ss + torch.bmm(Z_star_t.unsqueeze(1), A_ss.transpose(-1, -2)).squeeze(1)
                norm_dz = torch.norm(dz_dt, dim=1).cpu().numpy()
            
            truth_W_steady = steady_data.W_all
            truth_F_steady = steady_data.P_all / steady_data.P_all.sum(axis=1, keepdims=True)
            truth_P_steady = steady_data.P_all
            
            # CSD & Zbar for steady
            if ap.ion_available:
                truth_csd_steady = ap.compute_csd_numpy(truth_F_steady)
                pred_csd_steady_ae = ap.compute_csd_numpy(pred_F_steady_ae)
                pred_csd_steady_eq = ap.compute_csd_numpy(pred_F_steady_eq)
                
                truth_zbar_steady = ap.compute_zbar_numpy(truth_F_steady)
                pred_zbar_steady_ae = ap.compute_zbar_numpy(pred_F_steady_ae)
                pred_zbar_steady_eq = ap.compute_zbar_numpy(pred_F_steady_eq)
            else:
                truth_csd_steady = pred_csd_steady_ae = pred_csd_steady_eq = None
                truth_zbar_steady = pred_zbar_steady_ae = pred_zbar_steady_eq = None
            
            print("✅ Steady-state equilibrium analysis complete (Adaptive)")
        except Exception as e:
            print(f"⚠️  Steady-state equilibrium failed: {e}")
            Z_star = pred_F_steady_eq = norm_dz = None
            truth_P_steady = None
    else:
        # lstsq: Z* = -A^(-1) * (a + B*U)
        U_steady = steady_data.U_all
        try:
            Z_star = -np.linalg.solve(A_global, (a_global + U_steady @ B_global.T).T).T
            
            # Decode Equilibrium
            with torch.no_grad():
                Z_t = torch.tensor(Z_star, dtype=dtype, device=device)
                W_pred = ae.decoder(Z_t)
                if W_pred.dim() == 4:
                    W_pred = W_pred[:, 0, 0, :]
                elif W_pred.dim() == 3:
                    W_pred = W_pred[:, 0, :]
                
                pred_W_steady_eq = W_pred.cpu().numpy()
                pred_F_steady_eq = scale_helper.W_to_fraction(
                    W_pred.unsqueeze(1).unsqueeze(1)
                ).cpu().numpy().reshape(-1, cfg.data.nx)
            
            # AE reconstruction
            with torch.no_grad():
                W_steady_t = torch.tensor(steady_data.W_all, dtype=dtype, device=device)
                W_4d = W_steady_t.unsqueeze(1).unsqueeze(1)
                Z_steady_ae = ae.encoder(W_4d)
                if Z_steady_ae.dim() == 4:
                    Z_steady_ae = Z_steady_ae[:, 0, 0, :]
                elif Z_steady_ae.dim() == 3:
                    Z_steady_ae = Z_steady_ae[:, 0, :]
                
                W_steady_recon = ae.decoder(Z_steady_ae)
                if W_steady_recon.dim() == 4:
                    W_steady_recon = W_steady_recon[:, 0, 0, :]
                elif W_steady_recon.dim() == 3:
                    W_steady_recon = W_steady_recon[:, 0, :]
                
                pred_W_steady_ae = W_steady_recon.cpu().numpy()
                pred_F_steady_ae = scale_helper.W_to_fraction(
                    W_steady_recon.unsqueeze(1).unsqueeze(1)
                ).cpu().numpy().reshape(-1, cfg.data.nx)
            
            truth_W_steady = steady_data.W_all
            truth_F_steady = steady_data.P_all / steady_data.P_all.sum(axis=1, keepdims=True)
            truth_P_steady = steady_data.P_all  # Original population
            
            # dZ/dt should be ~0
            dz_dt = a_global + Z_star @ A_global.T + U_steady @ B_global.T
            norm_dz = np.linalg.norm(dz_dt, axis=1)
            
            # CSD & Zbar for steady
            if ap.ion_available:
                truth_csd_steady = ap.compute_csd_numpy(truth_F_steady)
                pred_csd_steady_ae = ap.compute_csd_numpy(pred_F_steady_ae)
                pred_csd_steady_eq = ap.compute_csd_numpy(pred_F_steady_eq)
                
                truth_zbar_steady = ap.compute_zbar_numpy(truth_F_steady)
                pred_zbar_steady_ae = ap.compute_zbar_numpy(pred_F_steady_ae)
                pred_zbar_steady_eq = ap.compute_zbar_numpy(pred_F_steady_eq)
            else:
                truth_csd_steady = pred_csd_steady_ae = pred_csd_steady_eq = None
                truth_zbar_steady = pred_zbar_steady_ae = pred_zbar_steady_eq = None
            
            print("✅ Steady-state equilibrium analysis complete (lstsq)")
        except Exception as e:
            print(f"⚠️  Steady-state equilibrium failed: {e}")
            Z_star = pred_F_steady_eq = norm_dz = None
            truth_P_steady = None
else:
    Z_star = pred_F_steady_eq = norm_dz = None
    truth_P_steady = None
    print("⚠️  Steady-state disabled")

#%%
# =============================================================================
# Helper Functions
# =============================================================================

def case_to_seg_idx(case_num):
    """Convert case number to segment index."""
    try:
        return case_numbers.index(case_num)
    except ValueError:
        print(f"⚠️  Case {case_num} not found in available cases")
        return None

def get_seg_info(seg_idx):
    """Return segment information."""
    sl = segment_slices[seg_idx]
    case_num = case_numbers[seg_idx]
    is_train = seg_idx in train_seg_ids
    return sl, case_num, is_train

def save_fig(name):
    """Save a plot."""
    if SAVE_PLOTS and plot_dir:
        path = plot_dir / f"{name}.png"
        plt.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f"  💾 Saved: {path}")

def compute_metrics(pred, truth, name=""):
    """Compute MSE, RMSE, MAE, and MRE(%)."""
    diff = pred - truth
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    mre_pct = float(np.mean(np.abs(diff) / (np.abs(truth) + 1e-10)) * 100)
    
    print(f"[{name}]")
    print(f"  MSE:  {mse:.6e}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  MAE:  {mae:.6e}")
    print(f"  MRE:  {mre_pct:.4f}%")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MRE_pct': mre_pct}

#%%
# =============================================================================
# [1] Training Curves
# =============================================================================

if VIS_SECTIONS['training_curves']:
    print("\n" + "="*60)
    print("[1] Training Curves")
    print("="*60)
    
    csv_to_npy(cfg.losscsv_path, cfg.losslog_path)
    csv_to_npy(cfg.vallosscsv_path, cfg.vallosslog_path)
    metrics_csv_to_npz(cfg.metrics_csv_path, cfg.metrics_path)
    
    # Load metrics
    if os.path.exists(cfg.metrics_path):
        H = np.load(cfg.metrics_path, allow_pickle=True)
        ep = H["epoch"] if "epoch" in H.files else None
        
        if ep is not None:
            n_ep = len(ep)
            
            # ==================== Helper: Compute weighted loss ====================
            def get_weighted(raw_name, weight_name):
                """Compute raw loss * weight."""
                if raw_name in H.files and weight_name in H.files:
                    raw = H[raw_name]
                    w = H[weight_name]
                    n = min(len(raw), len(w), n_ep)
                    return raw[:n] * w[:n]
                return None
            
            # ==================== 1-1. Total Loss Curve ====================
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI, constrained_layout=True)
            
            # Train & Val (already weighted)
            if "total" in H.files:
                axes[0].plot(ep, H["total"], 'b-', lw=1.5, label="Train Total (weighted)")
            if "val_total" in H.files:
                val = H["val_total"]
                n = min(len(ep), len(val))
                axes[0].plot(ep[:n], val[:n], 'r--', lw=1.5, label="Val Total")
            
            axes[0].set_yscale("log")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss (log)")
            axes[0].set_title("Total Loss Curve (Weighted)")
            axes[0].grid(True, which="both", alpha=0.3)
            axes[0].legend()
            
            # Individual RAW losses
            raw_names = ["recW", "FSE", "frac", "ion", "zbar", "sindy_norm", "coef_norm", 
                         "hurwitz", "steady_raw", "rate_W", "rate_N", "rate_CSD", "rate_Zbar"]
            colors = plt.cm.tab20(np.linspace(0, 1, len(raw_names)))
            
            for i, name in enumerate(raw_names):
                if name in H.files:
                    data = H[name]
                    n = min(len(ep), len(data))
                    if np.any(np.isfinite(data[:n]) & (data[:n] > 0)):
                        axes[1].plot(ep[:n], data[:n], color=colors[i], lw=1.2, label=name)
            
            axes[1].set_yscale("log")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss (log)")
            axes[1].set_title("Individual Loss Components (RAW, unweighted)")
            axes[1].grid(True, which="both", alpha=0.3)
            axes[1].legend(ncol=3, fontsize=8, loc='upper right')
            
            save_fig("training_curves_total")
            show_plot()
# ==================== 1-1b. Train/Val Best Epoch Overlay ====================
            # Plot train/val total loss on one graph and mark each best epoch with vertical lines
            fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI, constrained_layout=True)
            
            has_train = "total" in H.files
            has_val = "val_total" in H.files
            
            if has_train:
                train_total = H["total"]
                n_t = min(len(ep), len(train_total))
                ax.plot(ep[:n_t], train_total[:n_t], 'b-', lw=1.5, alpha=0.8, label="Train Total")
                
                # Find train best epoch
                valid_mask = np.isfinite(train_total[:n_t])
                if np.any(valid_mask):
                    train_best_idx = np.nanargmin(train_total[:n_t])
                    train_best_ep = ep[train_best_idx]
                    train_best_val = train_total[train_best_idx]
                    ax.axvline(x=train_best_ep, color='blue', linestyle='--', lw=2, alpha=0.7,
                               label=f"Train-best @ ep {int(train_best_ep)} ({train_best_val:.3e})")
            
            if has_val:
                val_total = H["val_total"]
                n_v = min(len(ep), len(val_total))
                ax.plot(ep[:n_v], val_total[:n_v], 'r-', lw=1.5, alpha=0.8, label="Val Total")
                
                # Find val best epoch
                valid_mask = np.isfinite(val_total[:n_v])
                if np.any(valid_mask):
                    val_best_idx = np.nanargmin(val_total[:n_v])
                    val_best_ep = ep[val_best_idx]
                    val_best_val = val_total[val_best_idx]
                    ax.axvline(x=val_best_ep, color='red', linestyle='--', lw=2, alpha=0.7,
                               label=f"Val-best @ ep {int(val_best_ep)} ({val_best_val:.3e})")
            
            ax.set_yscale("log")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss (log)", fontsize=12)
            ax.set_title("Train / Val Total Loss with Best Epochs", fontsize=14, fontweight='bold')
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=10, loc='upper right')
            
            save_fig("training_curves_best_epochs")
            show_plot()
# ==================== 1-2. RAW Loss Curves (Grouped) ====================
            fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=DPI, constrained_layout=True)
            fig.suptitle("RAW Loss Curves (Unweighted)", fontsize=14, fontweight='bold')
            
            # AE Losses (RAW)
            ax = axes[0, 0]
            ae_losses = ["recW", "FSE", "frac", "ion", "zbar"]
            for name in ae_losses:
                if name in H.files:
                    data = H[name]
                    n = min(len(ep), len(data))
                    ax.plot(ep[:n], data[:n], lw=1.2, label=name)
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log)")
            ax.set_title("AE Losses (RAW)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # SINDy Losses (RAW)
            ax = axes[0, 1]
            sindy_losses = ["sindy_norm", "coef_norm"]
            for name in sindy_losses:
                if name in H.files:
                    data = H[name]
                    n = min(len(ep), len(data))
                    ax.plot(ep[:n], data[:n], lw=1.5, label=name)
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log)")
            ax.set_title("SINDy Losses (RAW)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Hurwitz Loss (RAW)
            ax = axes[0, 2]
            if "hurwitz" in H.files:
                data = H["hurwitz"]
                n = min(len(ep), len(data))
                ax.plot(ep[:n], data[:n], 'r-', lw=1.5, label="hurwitz")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log)")
            ax.set_title("Hurwitz Loss (RAW)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Steady-state Loss (RAW)
            ax = axes[1, 0]
            if "steady_raw" in H.files:
                data = H["steady_raw"]
                n = min(len(ep), len(data))
                ax.plot(ep[:n], data[:n], 'g-', lw=1.5, label="steady_raw")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log)")
            ax.set_title("Steady-State Loss (RAW)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Rate Equation Losses (RAW)
            ax = axes[1, 1]
            rate_losses = ["rate_W", "rate_N", "rate_CSD", "rate_Zbar"]
            for name in rate_losses:
                if name in H.files:
                    data = H[name]
                    n = min(len(ep), len(data))
                    if np.any(np.isfinite(data[:n]) & (data[:n] > 0)):
                        ax.plot(ep[:n], data[:n], lw=1.2, label=name)
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log)")
            ax.set_title("Rate Equation Losses (RAW)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Train vs Val
            ax = axes[1, 2]
            if "total" in H.files:
                ax.plot(ep, H["total"], 'b-', lw=1.5, label="Train")
            if "val_total" in H.files:
                val = H["val_total"]
                n = min(len(ep), len(val))
                ax.plot(ep[:n], val[:n], 'r--', lw=1.5, label="Val")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log)")
            ax.set_title("Train vs Validation (Weighted)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            save_fig("training_curves_raw_grouped")
            show_plot()
# ==================== 1-3. WEIGHTED Loss Curves (Grouped) ====================
            fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=DPI, constrained_layout=True)
            fig.suptitle("WEIGHTED Loss Curves (loss × weight)", fontsize=14, fontweight='bold')
            
            # AE Losses (WEIGHTED)
            ax = axes[0, 0]
            ae_pairs = [("recW", None), ("FSE", "w_fse"), ("frac", "w_frac"), 
                        ("ion", "w_ion"), ("zbar", "w_zbar")]
            for raw_name, w_name in ae_pairs:
                if raw_name in H.files:
                    if w_name and w_name in H.files:
                        data = H[raw_name][:n_ep] * H[w_name][:n_ep]
                    else:
                        # recW weight is fixed in cfg, so display it directly
                        data = H[raw_name][:n_ep]
                    if np.any(np.isfinite(data) & (data > 0)):
                        ax.plot(ep[:len(data)], data, lw=1.2, label=f"{raw_name}×w")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted Loss (log)")
            ax.set_title("AE Losses (WEIGHTED)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # SINDy Losses (WEIGHTED)
            ax = axes[0, 1]
            sindy_pairs = [("sindy_norm", "w_sindy"), ("coef_norm", "w_coef")]
            for raw_name, w_name in sindy_pairs:
                if raw_name in H.files and w_name in H.files:
                    data = H[raw_name][:n_ep] * H[w_name][:n_ep]
                    if np.any(np.isfinite(data) & (data > 0)):
                        ax.plot(ep[:len(data)], data, lw=1.5, label=f"{raw_name}×w")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted Loss (log)")
            ax.set_title("SINDy Losses (WEIGHTED)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Hurwitz Loss (WEIGHTED)
            ax = axes[0, 2]
            if "hurwitz" in H.files and "w_hurwitz" in H.files:
                data = H["hurwitz"][:n_ep] * H["w_hurwitz"][:n_ep]
                if np.any(np.isfinite(data) & (data > 0)):
                    ax.plot(ep[:len(data)], data, 'r-', lw=1.5, label="hurwitz×w")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted Loss (log)")
            ax.set_title("Hurwitz Loss (WEIGHTED)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Steady-state Loss (WEIGHTED)
            ax = axes[1, 0]
            if "steady_raw" in H.files and "w_steady" in H.files:
                data = H["steady_raw"][:n_ep] * H["w_steady"][:n_ep]
                if np.any(np.isfinite(data) & (data > 0)):
                    ax.plot(ep[:len(data)], data, 'g-', lw=1.5, label="steady×w")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted Loss (log)")
            ax.set_title("Steady-State Loss (WEIGHTED)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Rate Equation Losses (WEIGHTED)
            ax = axes[1, 1]
            rate_pairs = [("rate_W", "w_rate_W"), ("rate_N", "w_rate_N"), 
                          ("rate_CSD", "w_rate_CSD"), ("rate_Zbar", "w_rate_Zbar")]
            for raw_name, w_name in rate_pairs:
                if raw_name in H.files and w_name in H.files:
                    data = H[raw_name][:n_ep] * H[w_name][:n_ep]
                    if np.any(np.isfinite(data) & (data > 0)):
                        ax.plot(ep[:len(data)], data, lw=1.2, label=f"{raw_name}×w")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted Loss (log)")
            ax.set_title("Rate Equation Losses (WEIGHTED)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Category-wise Total (WEIGHTED)
            ax = axes[1, 2]
            
            # Compute category totals
            # AE total
            ae_total = np.zeros(n_ep)
            if "recW" in H.files:
                ae_total += H["recW"][:n_ep]  # recW weight is fixed
            for raw_name, w_name in [("FSE", "w_fse"), ("frac", "w_frac"), ("ion", "w_ion"), ("zbar", "w_zbar")]:
                if raw_name in H.files and w_name in H.files:
                    ae_total += H[raw_name][:n_ep] * H[w_name][:n_ep]
            
            # SINDy total
            sindy_total = np.zeros(n_ep)
            for raw_name, w_name in [("sindy_norm", "w_sindy"), ("coef_norm", "w_coef")]:
                if raw_name in H.files and w_name in H.files:
                    sindy_total += H[raw_name][:n_ep] * H[w_name][:n_ep]
            
            # Hurwitz total
            hurwitz_total = np.zeros(n_ep)
            if "hurwitz" in H.files and "w_hurwitz" in H.files:
                hurwitz_total = H["hurwitz"][:n_ep] * H["w_hurwitz"][:n_ep]
            
            # Steady total
            steady_total = np.zeros(n_ep)
            if "steady_raw" in H.files and "w_steady" in H.files:
                steady_total = H["steady_raw"][:n_ep] * H["w_steady"][:n_ep]
            
            # Rate total
            rate_total = np.zeros(n_ep)
            for raw_name, w_name in rate_pairs:
                if raw_name in H.files and w_name in H.files:
                    rate_total += H[raw_name][:n_ep] * H[w_name][:n_ep]
            
            if np.any(ae_total > 0):
                ax.plot(ep, ae_total, lw=1.5, label="AE Total")
            if np.any(sindy_total > 0):
                ax.plot(ep, sindy_total, lw=1.5, label="SINDy Total")
            if np.any(hurwitz_total > 0):
                ax.plot(ep, hurwitz_total, lw=1.5, label="Hurwitz Total")
            if np.any(steady_total > 0):
                ax.plot(ep, steady_total, lw=1.5, label="Steady Total")
            if np.any(rate_total > 0):
                ax.plot(ep, rate_total, lw=1.5, label="Rate Total")
            if "total" in H.files:
                ax.plot(ep, H["total"], 'k--', lw=2, label="TOTAL")
            
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted Loss (log)")
            ax.set_title("Category-wise Total (WEIGHTED)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            save_fig("training_curves_weighted_grouped")
            show_plot()
            print("✅ Training curves plotted")

#%%
# =============================================================================
# [1.5] Weight Curves (NEW)
# =============================================================================

if VIS_SECTIONS['weight_curves']:
    print("\n" + "="*60)
    print("[1.5] Weight Ramp-up Curves")
    print("="*60)
    
    if os.path.exists(cfg.metrics_path):
        H = np.load(cfg.metrics_path, allow_pickle=True)
        ep = H["epoch"] if "epoch" in H.files else None
        
        if ep is not None:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=DPI, constrained_layout=True)
            
            # AE Weights
            ax = axes[0, 0]
            ae_weights = ["w_fse", "w_frac", "w_ion", "w_zbar"]
            for name in ae_weights:
                if name in H.files:
                    data = H[name]
                    n = min(len(ep), len(data))
                    ax.plot(ep[:n], data[:n], lw=1.5, label=name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.set_title("AE Loss Weights")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # SINDy Weights
            ax = axes[0, 1]
            sindy_weights = ["w_sindy", "w_coef"]
            for name in sindy_weights:
                if name in H.files:
                    data = H[name]
                    n = min(len(ep), len(data))
                    ax.plot(ep[:n], data[:n], lw=1.5, label=name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.set_title("SINDy Weights")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Hurwitz Weight
            ax = axes[0, 2]
            if "w_hurwitz" in H.files:
                data = H["w_hurwitz"]
                n = min(len(ep), len(data))
                ax.plot(ep[:n], data[:n], 'r-', lw=1.5, label="w_hurwitz")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.set_title("Hurwitz Weight")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Steady-state Weight
            ax = axes[1, 0]
            if "w_steady" in H.files:
                data = H["w_steady"]
                n = min(len(ep), len(data))
                ax.plot(ep[:n], data[:n], 'g-', lw=1.5, label="w_steady")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.set_title("Steady-State Weight")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Rate Equation Weights
            ax = axes[1, 1]
            rate_weights = ["w_rate_W", "w_rate_N", "w_rate_CSD", "w_rate_Zbar"]
            for name in rate_weights:
                if name in H.files:
                    data = H[name]
                    n = min(len(ep), len(data))
                    ax.plot(ep[:n], data[:n], lw=1.2, label=name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.set_title("Rate Equation Weights")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # All Weights (combined)
            ax = axes[1, 2]
            all_weights = ["w_fse", "w_frac", "w_ion", "w_zbar", "w_sindy", "w_coef", 
                           "w_hurwitz", "w_steady", "w_rate_W", "w_rate_N", "w_rate_CSD", "w_rate_Zbar"]
            colors = plt.cm.tab20(np.linspace(0, 1, len(all_weights)))
            for i, name in enumerate(all_weights):
                if name in H.files:
                    data = H[name]
                    n = min(len(ep), len(data))
                    if np.any(np.isfinite(data[:n]) & (data[:n] > 0)):
                        ax.plot(ep[:n], data[:n], color=colors[i], lw=1.0, label=name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.set_title("All Weights (Ramp-up)")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=7, loc='center right')
            
            save_fig("weight_curves")
            show_plot()
            print("✅ Weight curves plotted")

#%%
# =============================================================================
# [1.6] Learning Rate Curve (NEW)
# =============================================================================

if VIS_SECTIONS['lr_curve']:
    print("\n" + "="*60)
    print("[1.6] Learning Rate Curve")
    print("="*60)
    
    if os.path.exists(cfg.metrics_path):
        H = np.load(cfg.metrics_path, allow_pickle=True)
        ep = H["epoch"] if "epoch" in H.files else None
        
        if ep is not None and "lr" in H.files:
            lr_data = H["lr"]
            n = min(len(ep), len(lr_data))
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI, constrained_layout=True)
            
            # Linear scale
            axes[0].plot(ep[:n], lr_data[:n], 'b-', lw=1.5)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Learning Rate")
            axes[0].set_title("Learning Rate (Linear Scale)")
            axes[0].grid(True, alpha=0.3)
            axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # Log scale
            axes[1].plot(ep[:n], lr_data[:n], 'b-', lw=1.5)
            axes[1].set_yscale("log")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Learning Rate (log)")
            axes[1].set_title("Learning Rate (Log Scale)")
            axes[1].grid(True, which="both", alpha=0.3)
            
            save_fig("lr_curve")
            show_plot()
# Print summary
            print(f"  Initial LR: {lr_data[0]:.2e}")
            print(f"  Final LR:   {lr_data[n-1]:.2e}")
            print(f"  Min LR:     {np.min(lr_data[:n]):.2e}")
            print(f"  Max LR:     {np.max(lr_data[:n]):.2e}")
    
    print("✅ Learning rate curve plotted")

#%%
# =============================================================================
# [2] Comprehensive Metrics (Train/Val Separated)
# =============================================================================

if VIS_SECTIONS['comprehensive_metrics']:
    print("\n" + "="*60)
    print("[2] Comprehensive Metrics")
    print("="*60)
    
    # ==================== TRAIN METRICS ====================
    print("\n" + "="*60)
    print(" TRAIN SET METRICS")
    print("="*60)
    
    print("\n--- AE Reconstruction ---")
    compute_metrics(pred_W_ae[train_idx], truth_W[train_idx], "W-space (Train-AE)")
    compute_metrics(pred_frac_ae[train_idx], truth_frac[train_idx], "Fraction (Train-AE)")
    compute_metrics(pred_pop_ae[train_idx], truth_pop[train_idx], "Population (Train-AE)")
    
    if ap.ion_available:
        compute_metrics(pred_csd_ae[train_idx], truth_csd[train_idx], "CSD (Train-AE)")
        compute_metrics(pred_zbar_ae[train_idx], truth_zbar[train_idx], "Zbar (Train-AE)")
    
    print("\n--- SINDy Prediction ---")
    compute_metrics(pred_W_sindy[train_idx], truth_W[train_idx], "W-space (Train-SINDy)")
    compute_metrics(pred_frac_sindy[train_idx], truth_frac[train_idx], "Fraction (Train-SINDy)")
    compute_metrics(pred_pop_sindy[train_idx], truth_pop[train_idx], "Population (Train-SINDy)")
    
    if ap.ion_available:
        compute_metrics(pred_csd_sindy[train_idx], truth_csd[train_idx], "CSD (Train-SINDy)")
        compute_metrics(pred_zbar_sindy[train_idx], truth_zbar[train_idx], "Zbar (Train-SINDy)")
    
    # ==================== VALIDATION METRICS ====================
    print("\n" + "="*60)
    print(" VALIDATION SET METRICS")
    print("="*60)
    
    print("\n--- AE Reconstruction ---")
    compute_metrics(pred_W_ae[val_idx], truth_W[val_idx], "W-space (Val-AE)")
    compute_metrics(pred_frac_ae[val_idx], truth_frac[val_idx], "Fraction (Val-AE)")
    compute_metrics(pred_pop_ae[val_idx], truth_pop[val_idx], "Population (Val-AE)")
    
    if ap.ion_available:
        compute_metrics(pred_csd_ae[val_idx], truth_csd[val_idx], "CSD (Val-AE)")
        compute_metrics(pred_zbar_ae[val_idx], truth_zbar[val_idx], "Zbar (Val-AE)")
    
    print("\n--- SINDy Prediction ---")
    compute_metrics(pred_W_sindy[val_idx], truth_W[val_idx], "W-space (Val-SINDy)")
    compute_metrics(pred_frac_sindy[val_idx], truth_frac[val_idx], "Fraction (Val-SINDy)")
    compute_metrics(pred_pop_sindy[val_idx], truth_pop[val_idx], "Population (Val-SINDy)")
    
    if ap.ion_available:
        compute_metrics(pred_csd_sindy[val_idx], truth_csd[val_idx], "CSD (Val-SINDy)")
        compute_metrics(pred_zbar_sindy[val_idx], truth_zbar[val_idx], "Zbar (Val-SINDy)")
    
    # ==================== ACTIVATED METRICS (within comprehensive) ====================
    print("\n" + "="*60)
    print(" ACTIVATED METRICS (threshold-filtered)")
    print("="*60)
    
    for split_name, split_idx in [("Train", train_idx), ("Val", val_idx)]:
        print(f"\n--- {split_name} ---")
        for label, pf, pc, pz in [
            ("AE", pred_frac_ae, pred_csd_ae, pred_zbar_ae),
            ("SINDy", pred_frac_sindy, pred_csd_sindy, pred_zbar_sindy)
        ]:
            fm = compute_activated_metrics(truth_frac[split_idx], pf[split_idx],
                                            threshold=frac_threshold, name=f"Frac({split_name}-{label})")
            if fm:
                print(f"  {fm['name']:<25s} MRE={fm['mre']*100:>8.2f}%  MSE={fm['mse']:.4e}  ({fm['n_activated']}/{fm['n_total']} activated)")
            
            if ap.ion_available and pc is not None:
                cm = compute_activated_metrics(truth_csd[split_idx], pc[split_idx],
                                                threshold=csd_threshold, name=f"CSD({split_name}-{label})")
                if cm:
                    print(f"  {cm['name']:<25s} MRE={cm['mre']*100:>8.2f}%  MSE={cm['mse']:.4e}  ({cm['n_activated']}/{cm['n_total']} activated)")
                
                zm = compute_activated_metrics(truth_zbar[split_idx], pz[split_idx],
                                                threshold=None, name=f"Zbar({split_name}-{label})")
                if zm:
                    print(f"  {zm['name']:<25s} MRE={zm['mre']*100:>8.2f}%  MSE={zm['mse']:.4e}")
    
    # ==================== STEADY-STATE METRICS ====================
    if steady_data and steady_data.enabled and Z_star is not None:
        print("\n" + "="*60)
        print(" STEADY-STATE METRICS")
        print("="*60)
        
        print("\n--- AE Reconstruction ---")
        compute_metrics(pred_W_steady_ae, truth_W_steady, "W-space (Steady-AE)")
        compute_metrics(pred_F_steady_ae, truth_F_steady, "Fraction (Steady-AE)")
        
        print("\n--- Equilibrium (Z*) ---")
        compute_metrics(pred_W_steady_eq, truth_W_steady, "W-space (Steady-Equilibrium)")
        compute_metrics(pred_F_steady_eq, truth_F_steady, "Fraction (Steady-Equilibrium)")
        
        if ap.ion_available:
            compute_metrics(pred_csd_steady_ae, truth_csd_steady, "CSD (Steady-AE)")
            compute_metrics(pred_csd_steady_eq, truth_csd_steady, "CSD (Steady-Equilibrium)")
            compute_metrics(pred_zbar_steady_ae, truth_zbar_steady, "Zbar (Steady-AE)")
            compute_metrics(pred_zbar_steady_eq, truth_zbar_steady, "Zbar (Steady-Equilibrium)")
        
        print(f"\ndZ/dt norm statistics:")
        print(f"  Mean: {np.mean(norm_dz):.6e}")
        print(f"  Max:  {np.max(norm_dz):.6e}")
        print(f"  Min:  {np.min(norm_dz):.6e}")
    
    print("\n✅ Comprehensive metrics computed")

#%%
# =============================================================================
# [3] State-by-State RMSE/MRE (Population)
# =============================================================================

if VIS_SECTIONS['state_by_state']:
    print("\n" + "="*60)
    print("[3] State-by-State RMSE/MRE (Population)")
    print("="*60)
    
    nx = cfg.data.nx
    
    # AE
    rmse_per_state_ae = np.zeros(nx)
    mre_per_state_ae = np.zeros(nx)
    
    for s in range(nx):
        rmse_per_state_ae[s] = np.sqrt(np.mean((pred_pop_ae[:, s] - truth_pop[:, s])**2))
        mre_per_state_ae[s] = np.mean(np.abs(pred_pop_ae[:, s] - truth_pop[:, s]) / 
                                       (np.abs(truth_pop[:, s]) + cfg.data.pop_lim)) * 100
    
    # SINDy
    rmse_per_state_sindy = np.zeros(nx)
    mre_per_state_sindy = np.zeros(nx)
    
    for s in range(nx):
        rmse_per_state_sindy[s] = np.sqrt(np.mean((pred_pop_sindy[:, s] - truth_pop[:, s])**2))
        mre_per_state_sindy[s] = np.mean(np.abs(pred_pop_sindy[:, s] - truth_pop[:, s]) / 
                                          (np.abs(truth_pop[:, s]) + cfg.data.pop_lim)) * 100
    
    # Plot RMSE
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=DPI, constrained_layout=True)
    
    x = np.arange(nx)
    width = 0.35
    
    ax.bar(x - width/2, rmse_per_state_ae, width, label='AE', alpha=0.8)
    ax.bar(x + width/2, rmse_per_state_sindy, width, label='SINDy', alpha=0.8)
    
    ax.set_xlabel("State Index")
    ax.set_ylabel("RMSE")
    ax.set_title("State-by-State RMSE (Population)")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    save_fig("state_rmse_population")
    show_plot()
# Plot MRE (%)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=DPI, constrained_layout=True)
    
    ax.bar(x - width/2, mre_per_state_ae, width, label='AE', alpha=0.8)
    ax.bar(x + width/2, mre_per_state_sindy, width, label='SINDy', alpha=0.8)
    
    ax.set_xlabel("State Index")
    ax.set_ylabel("MRE (%)")
    ax.set_title("State-by-State MRE (%) (Population)")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    save_fig("state_mre_population")
    show_plot()
    print("✅ State-by-State metrics plotted")

#%%
# =============================================================================
# [4] Determine segments to visualize
# =============================================================================

if SELECTED_CASE_NUMBERS is None:
    # Default: first 3 cases
    seg_to_plot = list(range(min(3, len(segment_slices))))
else:
    # Convert case numbers to segment indices
    seg_to_plot = []
    for case_num in SELECTED_CASE_NUMBERS:
        seg_idx = case_to_seg_idx(case_num)
        if seg_idx is not None:
            seg_to_plot.append(seg_idx)
    
    if not seg_to_plot:
        print("⚠️  No valid cases found, using first 3 segments")
        seg_to_plot = list(range(min(3, len(segment_slices))))

print(f"\n[Visualization] Selected segments: {seg_to_plot}")
print(f"  Corresponding cases: {[case_numbers[i] for i in seg_to_plot]}")

#%%
# =============================================================================
# [5] Heatmaps - W-space
# =============================================================================

if VIS_SECTIONS['heatmaps']:
    print("\n" + "="*60)
    print("[5] Heatmaps - W-space")
    print("="*60)
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        split_str = "TRAIN" if is_train else "VAL"
        
        print(f"\n  Segment {seg_idx} (case={case_num}, {split_str})")
        
        truth_W_seg = truth_W[sl]
        pred_W_ae_seg = pred_W_ae[sl]
        pred_W_sindy_seg = pred_W_sindy[sl]
        
        vmin_w = 0.0
        vmax_w = 1.0
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=DPI, constrained_layout=True)
        
        im0 = axes[0].imshow(truth_W_seg.T, aspect='auto', origin='lower',
                              cmap='magma', vmin=vmin_w, vmax=vmax_w)
        axes[0].set_title(f"Truth W | seg{case_num} ({split_str})")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("State")
        
        im1 = axes[1].imshow(pred_W_ae_seg.T, aspect='auto', origin='lower',
                              cmap='magma', vmin=vmin_w, vmax=vmax_w)
        axes[1].set_title(f"AE Recon W | seg{case_num}")
        axes[1].set_xlabel("Time step")
        
        im2 = axes[2].imshow(pred_W_sindy_seg.T, aspect='auto', origin='lower',
                              cmap='magma', vmin=vmin_w, vmax=vmax_w)
        axes[2].set_title(f"SINDy W | seg{case_num}")
        axes[2].set_xlabel("Time step")
        
        error_w = np.abs(pred_W_ae_seg - truth_W_seg)
        im3 = axes[3].imshow(error_w.T, aspect='auto', origin='lower',
                              cmap='viridis', vmin=0, vmax=np.percentile(error_w, 99))
        axes[3].set_title(f"AE Error | seg{case_num}")
        axes[3].set_xlabel("Time step")
        
        for ax in axes:
            ax.grid(False)
        
        fig.colorbar(im0, ax=axes[:3].ravel().tolist(), label="W (scaled)", 
                     shrink=0.8, pad=0.02)
        fig.colorbar(im3, ax=axes[3], label="Absolute Error", 
                     shrink=0.8, pad=0.02)
        
        save_fig(f"heatmap_W_seg{case_num}")
        show_plot()
        print("✅ W-space heatmaps plotted")

#%%
# =============================================================================
# [6] Heatmaps - Fraction
# =============================================================================

if VIS_SECTIONS['heatmaps']:
    print("\n" + "="*60)
    print("[6] Heatmaps - Fraction")
    print("="*60)
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        
        truth_frac_seg = truth_frac[sl]
        pred_frac_ae_seg = pred_frac_ae[sl]
        pred_frac_sindy_seg = pred_frac_sindy[sl]
        
        if FRACTION_SCALE == 'log':
            vmin_f = 1e-10
            vmax_f = 1.0
            norm_f = LogNorm(vmin=vmin_f, vmax=vmax_f)
            label_f = "Fraction (log scale)"
        else:
            vmin_f = 0.0
            vmax_f = 1.0
            norm_f = None
            label_f = "Fraction (linear scale)"
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=DPI, constrained_layout=True)
        
        im0 = axes[0].imshow(truth_frac_seg.T, aspect='auto', origin='lower',
                              cmap='magma', norm=norm_f)
        axes[0].set_title(f"Truth Fraction | seg{case_num}")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("State")
        
        im1 = axes[1].imshow(pred_frac_ae_seg.T, aspect='auto', origin='lower',
                              cmap='magma', norm=norm_f)
        axes[1].set_title(f"AE Fraction | seg{case_num}")
        axes[1].set_xlabel("Time step")
        
        im2 = axes[2].imshow(pred_frac_sindy_seg.T, aspect='auto', origin='lower',
                              cmap='magma', norm=norm_f)
        axes[2].set_title(f"SINDy Fraction | seg{case_num}")
        axes[2].set_xlabel("Time step")
        
        rel_err = np.abs(pred_frac_ae_seg - truth_frac_seg) / (truth_frac_seg + 1e-10) * 100
        im3 = axes[3].imshow(rel_err.T, aspect='auto', origin='lower',
                              cmap='viridis', vmin=0, vmax=100)
        axes[3].set_title(f"AE Rel Error (%) | seg{case_num}")
        axes[3].set_xlabel("Time step")
        
        fig.colorbar(im0, ax=axes[:3].ravel().tolist(), label=label_f, 
                     shrink=0.8, pad=0.02)
        fig.colorbar(im3, ax=axes[3], label="Rel Error (%)", 
                     shrink=0.8, pad=0.02)
        
        save_fig(f"heatmap_fraction_{FRACTION_SCALE}_seg{case_num}")
        show_plot()
        print(f"✅ Fraction heatmaps plotted ({FRACTION_SCALE} scale)")

#%%
# =============================================================================
# [7] Heatmaps - Population
# =============================================================================

if VIS_SECTIONS['heatmaps']:
    print("\n" + "="*60)
    print("[7] Heatmaps - Population (Absolute)")
    print("="*60)
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        
        truth_pop_seg = truth_pop[sl]
        pred_pop_ae_seg = pred_pop_ae[sl]
        pred_pop_sindy_seg = pred_pop_sindy[sl]
        
        if POPULATION_SCALE == 'log':
            vmin_p = max(1e-10, np.percentile(truth_pop_seg[truth_pop_seg > 0], 1))
            vmax_p = np.percentile(truth_pop_seg, 99)
            norm_p = LogNorm(vmin=vmin_p, vmax=vmax_p)
            label_p = "Population (log scale)"
        else:
            vmin_p = 0
            vmax_p = np.percentile(truth_pop_seg, 99)
            norm_p = None
            label_p = "Population (linear scale)"
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=DPI, constrained_layout=True)
        
        im0 = axes[0].imshow(truth_pop_seg.T, aspect='auto', origin='lower',
                              cmap='magma', norm=norm_p)
        axes[0].set_title(f"Truth Population | seg{case_num}")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("State")
        
        im1 = axes[1].imshow(pred_pop_ae_seg.T, aspect='auto', origin='lower',
                              cmap='magma', norm=norm_p)
        axes[1].set_title(f"AE Population | seg{case_num}")
        axes[1].set_xlabel("Time step")
        
        im2 = axes[2].imshow(pred_pop_sindy_seg.T, aspect='auto', origin='lower',
                              cmap='magma', norm=norm_p)
        axes[2].set_title(f"SINDy Population | seg{case_num}")
        axes[2].set_xlabel("Time step")
        
        rel_err_pop = np.abs(pred_pop_ae_seg - truth_pop_seg) / (truth_pop_seg + cfg.data.pop_lim) * 100
        im3 = axes[3].imshow(rel_err_pop.T, aspect='auto', origin='lower',
                              cmap='viridis', vmin=0, vmax=100)
        axes[3].set_title(f"AE Rel Error (%) | seg{case_num}")
        axes[3].set_xlabel("Time step")
        
        fig.colorbar(im0, ax=axes[:3].ravel().tolist(), label=label_p, 
                     shrink=0.8, pad=0.02)
        fig.colorbar(im3, ax=axes[3], label="Rel Error (%)", 
                     shrink=0.8, pad=0.02)
        
        save_fig(f"heatmap_population_{POPULATION_SCALE}_seg{case_num}")
        show_plot()
        print(f"✅ Population heatmaps plotted ({POPULATION_SCALE} scale)")

#%%
# =============================================================================
# [8] Line Plots - Snapshots
# =============================================================================

if VIS_SECTIONS['line_plots']:
    print("\n" + "="*60)
    print("[8] Line Plots - Snapshots")
    print("="*60)
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        L = sl.stop - sl.start
        
        n_snapshots = 6
        time_indices = np.linspace(0, L-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=DPI, constrained_layout=True)
        axes = axes.ravel()
        
        for i, t_idx in enumerate(time_indices):
            ax = axes[i]
            global_idx = sl.start + t_idx
            
            x = np.arange(cfg.data.nx)
            
            ax.plot(x, truth_frac[global_idx], 'k-', lw=1.5, alpha=0.8, label='Truth')
            ax.plot(x, pred_frac_ae[global_idx], 'C1-', lw=1.0, alpha=0.8, label='AE')
            ax.plot(x, pred_frac_sindy[global_idx], 'C2--', lw=1.0, alpha=0.8, label='SINDy')
            
            rmse_ae = np.sqrt(np.mean((pred_frac_ae[global_idx] - truth_frac[global_idx])**2))
            rmse_sindy = np.sqrt(np.mean((pred_frac_sindy[global_idx] - truth_frac[global_idx])**2))
            
            ax.set_title(f"t={t_idx} | RMSE: AE={rmse_ae:.2e}, SINDy={rmse_sindy:.2e}", fontsize=9)
            ax.set_yscale('log')
            ax.set_ylim(1e-10, 1.5)
            ax.grid(True, alpha=0.3)
            
            if i % 3 == 0:
                ax.set_ylabel("Fraction")
            if i >= 3:
                ax.set_xlabel("State Index")
            if i == 0:
                ax.legend(fontsize=8)
        
        plt.suptitle(f"Fraction Snapshots | seg{case_num}", fontsize=12)
        save_fig(f"line_snapshots_seg{case_num}")
        show_plot()
        print("✅ Line snapshots plotted")

#%%
# =============================================================================
# [8.5] Fraction Time-Series Lines (each state vs time)
# =============================================================================

if VIS_SECTIONS['line_plots']:
    print("\n" + "="*60)
    print("[8.5] Fraction Time-Series Lines")
    print("="*60)
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        split_str = "TRAIN" if is_train else "VAL"
        
        t_seg = time_axis[sl]
        truth_frac_seg = truth_frac[sl]
        pred_frac_ae_seg = pred_frac_ae[sl]
        pred_frac_sindy_seg = pred_frac_sindy[sl]
        
        nx = cfg.data.nx
        cmap_jet = plt.get_cmap("jet")
        state_colors = [cmap_jet(i / max(1, nx - 1)) for i in range(nx)]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, dpi=DPI, 
                                  constrained_layout=True)
        
        for i in range(nx):
            c = state_colors[i]
            axes[0].plot(t_seg, truth_frac_seg[:, i], color=c, lw=0.8, alpha=0.8)
            axes[1].plot(t_seg, pred_frac_ae_seg[:, i], color=c, lw=0.8, alpha=0.8)
            axes[2].plot(t_seg, pred_frac_sindy_seg[:, i], color=c, lw=0.8, alpha=0.8)
        
        axes[0].set_title(f"Truth Fraction | seg{case_num} ({split_str})")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Fraction")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title(f"AE Fraction | seg{case_num}")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title(f"SINDy Fraction | seg{case_num}")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)
        
        save_fig(f"fraction_timeseries_seg{case_num}")
        show_plot()
        print("✅ Fraction time-series lines plotted")

#%%
# =============================================================================
# [9] CSD Heatmaps
# =============================================================================

if VIS_SECTIONS['physics'] and truth_csd is not None:
    print("\n" + "="*60)
    print("[9] CSD Heatmaps")
    print("="*60)
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        
        truth_csd_seg = truth_csd[sl]
        pred_csd_ae_seg = pred_csd_ae[sl]
        pred_csd_sindy_seg = pred_csd_sindy[sl]
        
        if CSD_SCALE == 'log':
            vmin_csd = 1e-10
            vmax_csd = 1.0
            norm_csd = LogNorm(vmin=vmin_csd, vmax=vmax_csd)
            label_csd = "CSD (log scale)"
        else:
            vmin_csd = 0.0
            vmax_csd = 1.0
            norm_csd = None
            label_csd = "CSD (linear scale)"
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=DPI, constrained_layout=True)
        
        im0 = axes[0].imshow(truth_csd_seg.T, aspect='auto', origin='lower',
                              cmap='viridis', norm=norm_csd)
        axes[0].set_title(f"Truth CSD | seg{case_num}")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Charge State (q)")
        
        im1 = axes[1].imshow(pred_csd_ae_seg.T, aspect='auto', origin='lower',
                              cmap='viridis', norm=norm_csd)
        axes[1].set_title(f"AE CSD | seg{case_num}")
        axes[1].set_xlabel("Time step")
        
        im2 = axes[2].imshow(pred_csd_sindy_seg.T, aspect='auto', origin='lower',
                              cmap='viridis', norm=norm_csd)
        axes[2].set_title(f"SINDy CSD | seg{case_num}")
        axes[2].set_xlabel("Time step")
        
        rel_err_csd = np.abs(pred_csd_ae_seg - truth_csd_seg) / (truth_csd_seg + 1e-10) * 100
        im3 = axes[3].imshow(rel_err_csd.T, aspect='auto', origin='lower',
                              cmap='viridis', vmin=0, vmax=100)
        axes[3].set_title(f"AE Rel Error (%) | seg{case_num}")
        axes[3].set_xlabel("Time step")
        
        fig.colorbar(im0, ax=axes[:3].ravel().tolist(), label=label_csd, 
                     shrink=0.8, pad=0.02)
        fig.colorbar(im3, ax=axes[3], label="Rel Error (%)", 
                     shrink=0.8, pad=0.02)
        
        save_fig(f"csd_{CSD_SCALE}_seg{case_num}")
        show_plot()
        print(f"✅ CSD heatmaps plotted ({CSD_SCALE} scale)")

#%%
# =============================================================================
# [9.5] CSD Time-Series Lines (each charge state vs time)
# =============================================================================

if VIS_SECTIONS['physics'] and truth_csd is not None:
    print("\n" + "="*60)
    print("[9.5] CSD Time-Series Lines")
    print("="*60)
    
    nq = ap.nq
    cmap_jet = plt.get_cmap("jet")
    ion_colors = [cmap_jet(q / max(1, nq - 1)) for q in range(nq)]
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        split_str = "TRAIN" if is_train else "VAL"
        
        t_seg = time_axis[sl]
        truth_csd_seg = truth_csd[sl]
        pred_csd_ae_seg = pred_csd_ae[sl]
        pred_csd_sindy_seg = pred_csd_sindy[sl]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, dpi=DPI, 
                                  constrained_layout=True)
        
        for q in range(nq):
            c = ion_colors[q]
            lbl = f"q={q}" if q < 10 else None
            axes[0].plot(t_seg, truth_csd_seg[:, q], color=c, lw=0.8, alpha=0.8, label=lbl)
            axes[1].plot(t_seg, pred_csd_ae_seg[:, q], color=c, lw=0.8, alpha=0.8, label=lbl)
            axes[2].plot(t_seg, pred_csd_sindy_seg[:, q], color=c, lw=0.8, alpha=0.8, label=lbl)
        
        axes[0].set_title(f"Truth CSD | seg{case_num} ({split_str})")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("CSD (fraction)")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=6, ncol=2, loc='upper right')
        
        axes[1].set_title(f"AE CSD | seg{case_num}")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title(f"SINDy CSD | seg{case_num}")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)
        
        save_fig(f"csd_timeseries_seg{case_num}")
        show_plot()
        print("✅ CSD time-series lines plotted")

#%%
# =============================================================================
# [10] Zbar Curves
# =============================================================================

if VIS_SECTIONS['physics'] and truth_zbar is not None:
    print("\n" + "="*60)
    print("[10] Zbar Curves")
    print("="*60)
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        
        truth_zbar_seg = truth_zbar[sl]
        pred_zbar_ae_seg = pred_zbar_ae[sl]
        pred_zbar_sindy_seg = pred_zbar_sindy[sl]
        
        t_seg = time_axis[sl]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=DPI, constrained_layout=True)
        
        ax1.plot(t_seg, truth_zbar_seg, 'k-', lw=1.5, label='Truth')
        ax1.plot(t_seg, pred_zbar_ae_seg, 'C1-', lw=1.0, alpha=0.8, label='AE')
        ax1.plot(t_seg, pred_zbar_sindy_seg, 'C2--', lw=1.0, alpha=0.8, label='SINDy')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mean Charge (Zbar)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Zbar Comparison | seg{case_num}")
        
        error_ae = pred_zbar_ae_seg - truth_zbar_seg
        error_sindy = pred_zbar_sindy_seg - truth_zbar_seg
        
        ax2.plot(t_seg, error_ae, 'C1-', lw=1.0, label='AE Error')
        ax2.plot(t_seg, error_sindy, 'C2--', lw=1.0, label='SINDy Error')
        ax2.axhline(0, color='k', ls='--', lw=0.5)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Error (Pred - Truth)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Zbar Error")
        
        save_fig(f"zbar_seg{case_num}")
        show_plot()
        print("✅ Zbar curves plotted")

#%%
# =============================================================================
# [11] Ion-by-Ion RMSE
# =============================================================================

if VIS_SECTIONS['physics'] and truth_csd is not None:
    print("\n" + "="*60)
    print("[11] Ion-by-Ion RMSE")
    print("="*60)
    
    nq = ap.nq
    rmse_per_ion_ae = np.zeros(nq)
    rmse_per_ion_sindy = np.zeros(nq)
    
    for q in range(nq):
        rmse_per_ion_ae[q] = np.sqrt(np.mean((pred_csd_ae[:, q] - truth_csd[:, q])**2))
        rmse_per_ion_sindy[q] = np.sqrt(np.mean((pred_csd_sindy[:, q] - truth_csd[:, q])**2))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=DPI, constrained_layout=True)
    
    x = np.arange(nq)
    width = 0.35
    
    ax.bar(x - width/2, rmse_per_ion_ae, width, label='AE', alpha=0.8)
    ax.bar(x + width/2, rmse_per_ion_sindy, width, label='SINDy', alpha=0.8)
    
    ax.set_xlabel("Charge State (q)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Ion-by-Ion RMSE (Z={ap.Z0})")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    save_fig("ion_rmse")
    show_plot()
    print("✅ Ion-by-Ion RMSE plotted")

#%%
# =============================================================================
# [12] Latent Space Z Trajectory
# =============================================================================

if VIS_SECTIONS['latent']:
    print("\n" + "="*60)
    print("[12] Latent Space")
    print("="*60)
    
    nz = cfg.model.latent_dim
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        
        Z_truth_seg = Z_truth_np[sl]
        Z_sindy_seg = Z_pred_sindy[sl]
        t_seg = time_axis[sl]
        
        cmap = plt.cm.viridis
        cZ = [cmap(i / max(1, nz-1)) for i in range(nz)]
        
        # (1) Individual
        fig, axes = plt.subplots(nz, 1, figsize=(10, 3*nz), dpi=DPI, 
                                  sharex=True, constrained_layout=True)
        if nz == 1:
            axes = [axes]
        
        for i in range(nz):
            axes[i].plot(t_seg, Z_truth_seg[:, i], '--', color=cZ[i], lw=2,
                         label=f'Truth Z[{i}]')
            axes[i].plot(t_seg, Z_sindy_seg[:, i], '-', color=cZ[i], lw=1.5,
                         alpha=0.8, label=f'SINDy Z[{i}]')
            axes[i].set_ylabel(f"Z[{i}]")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=9)
        
        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f"Z Trajectory (Individual) | seg{case_num}", fontsize=12)
        save_fig(f"z_trajectory_individual_seg{case_num}")
        show_plot()
# (2) Combined
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=DPI, constrained_layout=True)
        
        for i in range(nz):
            ax.plot(t_seg, Z_truth_seg[:, i], '--', color=cZ[i], lw=2,
                    label=f'Truth Z[{i}]')
            ax.plot(t_seg, Z_sindy_seg[:, i], '-', color=cZ[i], lw=1.5,
                    alpha=0.8, label=f'SINDy Z[{i}]')
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Latent value")
        ax.set_title(f"Z Trajectory (Combined) | seg{case_num}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        
        save_fig(f"z_trajectory_combined_seg{case_num}")
        show_plot()
        print("✅ Latent space plotted")

#%%
# =============================================================================
# [13] SINDy Analysis (lstsq mode only)
# =============================================================================

if VIS_SECTIONS['sindy'] and not use_adaptive_sindy:
    print("\n" + "="*60)
    print("[13] SINDy Analysis")
    print("="*60)
    
    # Coefficient bar plot
    print("\n  SINDy Coefficients:")
    
    feat_names = ["const"]
    for i in range(cfg.model.latent_dim):
        feat_names.append(f"Z{i}")
    for k in range(mu):
        feat_names.append(f"U{k}")
    
    n_feat = len(feat_names)
    
    for eq_idx in range(cfg.model.latent_dim):
        fig, ax = plt.subplots(1, 1, figsize=(max(10, n_feat*0.6), 5), 
                                dpi=DPI, constrained_layout=True)
        
        coef_for_eq = []
        coef_for_eq.append(a_global[eq_idx])
        for j in range(cfg.model.latent_dim):
            coef_for_eq.append(A_global[eq_idx, j])
        for k in range(mu):
            coef_for_eq.append(B_global[eq_idx, k])
        
        coef_for_eq = np.array(coef_for_eq)
        
        x = np.arange(n_feat)
        ax.bar(x, coef_for_eq, alpha=0.8, edgecolor='k', linewidth=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(feat_names, rotation=0)
        ax.set_xlabel("Features")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"SINDy Coefficients — dZ{eq_idx}/dt")
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='k', linewidth=0.5)
        
        save_fig(f"sindy_coef_eq{eq_idx}")
        show_plot()
# Eigenvalue complex plane
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=DPI, constrained_layout=True)
    
    ax.scatter(eigvals.real, eigvals.imag, s=100, c='C0',
               edgecolors='k', linewidths=1.5, zorder=5, label='Eigenvalues')
    
    ax.axvline(0, color='red', ls='--', lw=2, label='Re(λ)=0 (Hurwitz boundary)')
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    
    for i, (re, im) in enumerate(zip(eigvals.real, eigvals.imag)):
        ax.text(re, im, f'  λ{i+1}', fontsize=9, va='center')
    
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(f"Eigenvalue Complex Plane | Hurwitz: {is_hurwitz}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    save_fig("eigenvalue_complex")
    show_plot()
# A & B matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI, constrained_layout=True)
    
    im0 = axes[0].imshow(A_global, cmap='RdBu_r', aspect='auto', origin='lower')
    axes[0].set_title("A Matrix (State Dynamics)")
    axes[0].set_xlabel("From Z_j")
    axes[0].set_ylabel("To Z_i")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(B_global, cmap='RdBu_r', aspect='auto', origin='lower')
    axes[1].set_title("B Matrix (Control Input)")
    axes[1].set_xlabel("Control (T, n)")
    axes[1].set_ylabel("To Z_i")
    plt.colorbar(im1, ax=axes[1])
    
    save_fig("AB_matrices")
    show_plot()
# SINDy equations
    print("\nSINDy Equations: dZ/dt = a + A·Z + B·U")
    print("="*60)
    for i in range(cfg.model.latent_dim):
        terms = [f"{a_global[i]:+.4e}"]
        for j in range(cfg.model.latent_dim):
            if abs(A_global[i, j]) > 1e-8:
                terms.append(f"{A_global[i, j]:+.4e}·Z{j}")
        for k in range(mu):
            if abs(B_global[i, k]) > 1e-8:
                terms.append(f"{B_global[i, k]:+.4e}·U{k}")
        eq = " ".join(terms)
        print(f"dZ{i}/dt = {eq}")
    print("="*60)
    
    print("✅ SINDy analysis plotted")

#%%
# =============================================================================
# [14] Steady-State Analysis (Truth | AE | Equilibrium)
# =============================================================================

if VIS_SECTIONS['steady'] and steady_data and steady_data.enabled and Z_star is not None:
    print("\n" + "="*60)
    print("[14] Steady-State Analysis (3-Panel)")
    print("="*60)
    
    # Random samples
    n_show = 6
    indices = np.random.choice(len(truth_F_steady), min(n_show, len(truth_F_steady)), replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=DPI, constrained_layout=True)
    axes = axes.ravel()
    
    x = np.arange(cfg.data.nx)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        ax.plot(x, truth_F_steady[idx], 'k-', lw=1.5, label='Truth')
        ax.plot(x, pred_F_steady_ae[idx], 'C1-.', lw=1.0, alpha=0.9, label='AE')
        ax.plot(x, pred_F_steady_eq[idx], 'C2--', lw=1.0, alpha=0.9, label='Equilibrium (Z*)')
        
        rmse_ae = np.sqrt(np.mean((pred_F_steady_ae[idx] - truth_F_steady[idx])**2))
        rmse_eq = np.sqrt(np.mean((pred_F_steady_eq[idx] - truth_F_steady[idx])**2))
        
        T_val = steady_data.Uraw_all[idx, 0]
        n_val = steady_data.Uraw_all[idx, 1]
        
        ax.set_title(f"T={T_val:.2f} eV, n={n_val:.2e}\nRMSE: AE={rmse_ae:.2e}, Eq={rmse_eq:.2e}", fontsize=8)
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1.5)
        ax.grid(True, alpha=0.3)
        
        if i % 3 == 0:
            ax.set_ylabel("Fraction")
        if i >= 3:
            ax.set_xlabel("State Index")
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle("Steady-State Comparison (Truth | AE | Equilibrium)", fontsize=12)
    save_fig("steady_comparison")
    show_plot()
# dZ/dt histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=DPI, constrained_layout=True)
    ax.hist(norm_dz, bins=50, edgecolor='k', alpha=0.7)
    ax.set_xlabel("||dZ/dt||")
    ax.set_ylabel("Count")
    ax.set_title("Steady-State dZ/dt (should be ~0)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    save_fig("steady_dz_dt")
    show_plot()
# CSD & Zbar comparison (if available)
    if ap.ion_available:
        # CSD 3-panel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI, constrained_layout=True)
        
        case_idx = np.arange(len(truth_F_steady))
        
        im0 = axes[0].imshow(truth_csd_steady.T, aspect='auto', origin='lower',
                              cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title("Truth CSD (Steady)")
        axes[0].set_xlabel("Case Index")
        axes[0].set_ylabel("Charge State")
        
        im1 = axes[1].imshow(pred_csd_steady_ae.T, aspect='auto', origin='lower',
                              cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title("AE CSD (Steady)")
        axes[1].set_xlabel("Case Index")
        
        im2 = axes[2].imshow(pred_csd_steady_eq.T, aspect='auto', origin='lower',
                              cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title("Equilibrium CSD (Steady)")
        axes[2].set_xlabel("Case Index")
        
        fig.colorbar(im2, ax=axes.ravel().tolist(), label="CSD", 
                     shrink=0.8, pad=0.02)
        save_fig("steady_csd")
        show_plot()
# Zbar comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=DPI, constrained_layout=True)
        ax.plot(case_idx, truth_zbar_steady, 'k-', lw=2, label='Truth')
        ax.plot(case_idx, pred_zbar_steady_ae, 'C1-.', lw=1.5, alpha=0.9, label='AE')
        ax.plot(case_idx, pred_zbar_steady_eq, 'C2--', lw=1.5, alpha=0.9, label='Equilibrium')
        ax.set_xlabel("Case Index")
        ax.set_ylabel("Mean Charge (Zbar)")
        ax.set_title("Steady-State Zbar Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_fig("steady_zbar")
        show_plot()
        print("✅ Steady-state analysis plotted")

#%%
# =============================================================================
# [15] Steady-State Rollout (Control Hold) - lstsq mode only
# =============================================================================

if VIS_SECTIONS['steady_rollout'] and steady_data and steady_data.enabled and not use_adaptive_sindy:
    print("\n" + "="*60)
    print("[15] Steady-State Rollout (Control Hold)")
    print("="*60)
    
    # Select segment
    if ROLLOUT_CASE_NUMBER is None:
        seg_idx = 0
    else:
        seg_idx = case_to_seg_idx(ROLLOUT_CASE_NUMBER)
        if seg_idx is None:
            seg_idx = 0
    
    sl = segment_slices[seg_idx]
    case_num = case_numbers[seg_idx]
    
    # Original data
    orig_len = sl.stop - sl.start
    U_orig = U_all[sl]
    
    # Initial condition
    with torch.no_grad():
        x0_tensor = X_frames[sl.start:sl.start+1].to(device=device, dtype=dtype)
        z0 = ae.encoder(x0_tensor).squeeze().cpu().numpy()
    
    # Time extension
    # Compute extension steps based on the physical dt (e.g., 0.01 ns)
    dt_real = float(cfg.data.dt)       # Physical dt [s] (e.g., 0.01e-9)
    dt_real_ns = dt_real * 1e9          # Convert physical dt to ns (e.g., 0.01)
    dt_sindy = float(dt_eff)            # dt_eff for SINDy integration
    
    n_extend = int(ROLLOUT_EXTEND_NS / dt_real_ns)  # Number of steps based on physical dt
    
    U_last = U_orig[-1]
    U_extend = np.tile(U_last, (n_extend, 1))
    U_total = np.vstack([U_orig, U_extend])
    total_len = U_total.shape[0]
    t_total_ns = np.arange(total_len) * dt_real_ns   # Physical time (ns)
    t_total_sindy = np.arange(total_len) * dt_sindy    # Time for SINDy integration
    
    print(f"[seg{case_num}] Simulating...")
    print(f"  - dt_real: {dt_real_ns:.4g} ns, dt_eff (SINDy): {dt_sindy:.4g}")
    print(f"  - Original: {orig_len} steps ({orig_len*dt_real_ns:.2f} ns)")
    print(f"  - Extended: {n_extend} steps (+{ROLLOUT_EXTEND_NS:.2f} ns)")
    print(f"  - Total: {total_len} steps ({t_total_ns[-1]:.2f} ns)")
    
    # SINDy simulation (using dt_eff time grid for integration)
    Z_sim = ld.simulate(coef_vec_np, z0, t_total_sindy, U=U_total)
    
    # Decode
    Z_sim_t = torch.tensor(Z_sim, dtype=dtype, device=device)
    with torch.no_grad():
        W_sim_t = ae.decoder(Z_sim_t)
        if W_sim_t.dim() == 4:
            W_sim_t = W_sim_t[:, 0, 0, :]
        elif W_sim_t.dim() == 3:
            W_sim_t = W_sim_t[:, 0, :]
        
        X_frac_sim = scale_helper.W_to_fraction(
            W_sim_t.unsqueeze(1).unsqueeze(1)
        ).cpu().numpy().reshape(-1, cfg.data.nx)
    
    # Truth data (original length only)
    truth_frac_orig = truth_frac[sl]
    
    # CSD & Zbar
    if ap.ion_available:
        CSD_sim = ap.compute_csd_numpy(X_frac_sim)
        Zbar_sim = ap.compute_zbar_numpy(X_frac_sim)
        CSD_truth = ap.compute_csd_numpy(truth_frac_orig)
        Zbar_truth = ap.compute_zbar_numpy(truth_frac_orig)
    else:
        CSD_sim = Zbar_sim = CSD_truth = Zbar_truth = None
    
    # ==================== Visualization ====================
    
    # (1) Control Input
    fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
    ax.plot(t_total_ns, U_total[:, 0], 'r-', label='Temp (scaled)')
    ax.plot(t_total_ns, U_total[:, 1], 'b-', label='Density (scaled)')
    ax.axvline(t_total_ns[orig_len-1], color='k', ls='--', label='Hold Start')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Control Input (Scaled)")
    ax.set_title(f"Control Input Trajectory | seg{case_num}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(f"rollout_control_seg{case_num}")
    show_plot()
# (2) Latent Space
    nz = z0.shape[0]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
    cmap = plt.cm.get_cmap('tab10')
    ax.axvline(t_total_ns[orig_len-1], color='k', ls='--', lw=1.5, alpha=0.5, label='Ext. Start')
    for i in range(nz):
        ax.plot(t_total_ns, Z_sim[:, i], color=cmap(i), label=f'z{i+1}')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Latent State")
    ax.set_title(f"Latent Dynamics (Rollout) | seg{case_num}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize='small')
    save_fig(f"rollout_latent_seg{case_num}")
    show_plot()
# (3) Fraction Heatmap
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=DPI, constrained_layout=True)
    vmax = np.percentile(truth_frac_orig, 99)
    vmin = np.percentile(truth_frac_orig, 1)
    extent = [0, t_total_ns[-1], 0, X_frac_sim.shape[1]]
    
    # Pred
    im0 = axes[0].imshow(X_frac_sim.T, aspect='auto', origin='lower', extent=extent, 
                          cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].axvline(t_total_ns[orig_len-1], color='w', ls='--')
    axes[0].set_title("Predicted Fraction (Extended)")
    axes[0].set_ylabel("State Index")
    plt.colorbar(im0, ax=axes[0], label='Frac')
    
    # Truth
    truth_pad = np.full_like(X_frac_sim, np.nan)
    truth_pad[:orig_len, :] = truth_frac_orig
    im1 = axes[1].imshow(truth_pad.T, aspect='auto', origin='lower', extent=extent, 
                          cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].axvline(t_total_ns[orig_len-1], color='w', ls='--')
    axes[1].set_title("Truth Fraction (Original)")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("State Index")
    axes[1].set_facecolor('0.9')
    plt.colorbar(im1, ax=axes[1], label='Frac')
    
    save_fig(f"rollout_fraction_seg{case_num}")
    show_plot()
# (4) CSD & Zbar
    if CSD_sim is not None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=DPI, constrained_layout=True)
        
        vmax_c = np.percentile(CSD_truth, 99)
        extent_c = [0, t_total_ns[-1], 0, CSD_sim.shape[1]]
        
        # CSD Pred
        im_c0 = axes[0].imshow(CSD_sim.T, aspect='auto', origin='lower', extent=extent_c, 
                                cmap='inferno', vmax=vmax_c)
        axes[0].axvline(t_total_ns[orig_len-1], color='w', ls='--')
        axes[0].set_title("Predicted CSD (Extended)")
        axes[0].set_ylabel("Charge State")
        plt.colorbar(im_c0, ax=axes[0], label='Pop')
        
        # CSD Truth
        csd_truth_pad = np.full_like(CSD_sim, np.nan)
        csd_truth_pad[:orig_len, :] = CSD_truth
        im_c1 = axes[1].imshow(csd_truth_pad.T, aspect='auto', origin='lower', extent=extent_c, 
                                cmap='inferno', vmax=vmax_c)
        axes[1].axvline(t_total_ns[orig_len-1], color='w', ls='--')
        axes[1].set_title("Truth CSD (Original)")
        axes[1].set_ylabel("Charge State")
        axes[1].set_facecolor('0.9')
        plt.colorbar(im_c1, ax=axes[1], label='Pop')
        
        # Zbar
        axes[2].plot(t_total_ns[:orig_len], Zbar_truth, 'k--', lw=2.0, label='Truth Zbar')
        axes[2].plot(t_total_ns, Zbar_sim, 'r-', lw=1.5, label='Pred Zbar')
        axes[2].axvline(t_total_ns[orig_len-1], color='k', ls='--', alpha=0.5)
        axes[2].set_xlabel("Time (ns)")
        axes[2].set_ylabel("Zbar")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        save_fig(f"rollout_csd_zbar_seg{case_num}")
        show_plot()
# (5) Convergence
    Z_dot = a_global + Z_sim @ A_global.T + U_total @ B_global.T
    Z_dot_norm = np.linalg.norm(Z_dot, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
    ax.plot(t_total_ns, Z_dot_norm, 'b-', lw=1.5)
    ax.axvline(t_total_ns[orig_len-1], color='k', ls='--', alpha=0.5, label='Hold Start')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("||dZ/dt||")
    ax.set_title(f"Convergence to Equilibrium | seg{case_num}")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(f"rollout_convergence_seg{case_num}")
    show_plot()
    print("✅ Steady-state rollout complete")

#%%
# =============================================================================
# [16] Inference Benchmark - lstsq mode only
# =============================================================================

if VIS_SECTIONS['benchmark'] and not use_adaptive_sindy:
    print("\n" + "="*60)
    print("[16] Inference Benchmark")
    print("="*60)
    
    seg_idx = 0
    sl, case_num, _ = get_seg_info(seg_idx)
    
    U_sample = U_all[sl]
    L = U_sample.shape[0]
    t_sample = np.linspace(0.0, (L-1)*dt_eff, L)
    
    # z0
    X0 = X_frames[sl.start:sl.start+1].to(device=device, dtype=dtype)
    with torch.no_grad():
        z0 = ae.encoder(X0).squeeze().cpu().numpy()
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()
    
    Z_pred = ld.simulate(coef_vec_np, z0, t_sample, U=U_sample)
    
    Z_t = torch.tensor(Z_pred, dtype=dtype, device=device)
    with torch.no_grad():
        W_pred = ae.decoder(Z_t)
        if W_pred.dim() == 4:
            W_pred = W_pred[:, 0, 0, :]
        elif W_pred.dim() == 3:
            W_pred = W_pred[:, 0, :]
        
        _ = scale_helper.W_to_fraction(
            W_pred.unsqueeze(1).unsqueeze(1)
        ).cpu().numpy()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.time()
    
    elapsed = t_end - t_start
    
    print(f"\nInference Benchmark | seg{case_num}")
    print(f"{'='*60}")
    print(f"Segment Length     : {L} steps")
    print(f"Total Time         : {elapsed:.6f} sec")
    print(f"Time per Step      : {elapsed / L * 1000:.4f} ms")
    print(f"{'='*60}")
    
    print("✅ Benchmark complete")

#%%
# =============================================================================
# [17] Rate Equation Analysis
# =============================================================================

if VIS_SECTIONS['rate_equation']:
    print("\n" + "="*60)
    print("[17] Rate Equation Analysis")
    print("="*60)
    
    def central_diff_np(X, dt):
        """Compute time derivatives with central differences (NumPy)."""
        return (X[2:] - X[:-2]) / (2.0 * dt)
    
    def compute_rate_rmse(pred, truth):
        """Compute rate-equation RMSE."""
        diff = pred - truth
        return float(np.sqrt(np.mean(diff**2)))
    
    def compute_rate_mre(pred, truth, eps=1e-10):
        """Compute rate-equation MRE (%)."""
        diff = np.abs(pred - truth)
        denom = np.abs(truth) + eps
        return float(np.mean(diff / denom) * 100)
    
    dt = cfg.data.dt
    
    # ==================== Global Rate Equation Metrics ====================
    print("\n--- Global Rate Equation Metrics ---")
    
    # dW/dt
    dW_truth = central_diff_np(truth_W, dt)
    dW_pred_ae = central_diff_np(pred_W_ae, dt)
    dW_pred_sindy = central_diff_np(pred_W_sindy, dt)
    
    print(f"\n[dW/dt]")
    print(f"  AE    RMSE: {compute_rate_rmse(dW_pred_ae, dW_truth):.6e}")
    print(f"  SINDy RMSE: {compute_rate_rmse(dW_pred_sindy, dW_truth):.6e}")
    
    # dFrac/dt
    dFrac_truth = central_diff_np(truth_frac, dt)
    dFrac_pred_ae = central_diff_np(pred_frac_ae, dt)
    dFrac_pred_sindy = central_diff_np(pred_frac_sindy, dt)
    
    print(f"\n[dFrac/dt]")
    print(f"  AE    RMSE: {compute_rate_rmse(dFrac_pred_ae, dFrac_truth):.6e}")
    print(f"  SINDy RMSE: {compute_rate_rmse(dFrac_pred_sindy, dFrac_truth):.6e}")
    
    # dN/dt (Population)
    dN_truth = central_diff_np(truth_pop, dt)
    dN_pred_ae = central_diff_np(pred_pop_ae, dt)
    dN_pred_sindy = central_diff_np(pred_pop_sindy, dt)
    
    print(f"\n[dN/dt (Population)]")
    print(f"  AE    RMSE: {compute_rate_rmse(dN_pred_ae, dN_truth):.6e}")
    print(f"  SINDy RMSE: {compute_rate_rmse(dN_pred_sindy, dN_truth):.6e}")
    
    # dCSD/dt & dZbar/dt (if available)
    if ap.ion_available and truth_csd is not None:
        dCSD_truth = central_diff_np(truth_csd, dt)
        dCSD_pred_ae = central_diff_np(pred_csd_ae, dt)
        dCSD_pred_sindy = central_diff_np(pred_csd_sindy, dt)
        
        print(f"\n[dCSD/dt]")
        print(f"  AE    RMSE: {compute_rate_rmse(dCSD_pred_ae, dCSD_truth):.6e}")
        print(f"  SINDy RMSE: {compute_rate_rmse(dCSD_pred_sindy, dCSD_truth):.6e}")
        
        dZbar_truth = central_diff_np(truth_zbar.reshape(-1, 1), dt)
        dZbar_pred_ae = central_diff_np(pred_zbar_ae.reshape(-1, 1), dt)
        dZbar_pred_sindy = central_diff_np(pred_zbar_sindy.reshape(-1, 1), dt)
        
        print(f"\n[dZbar/dt]")
        print(f"  AE    RMSE: {compute_rate_rmse(dZbar_pred_ae, dZbar_truth):.6e}")
        print(f"  SINDy RMSE: {compute_rate_rmse(dZbar_pred_sindy, dZbar_truth):.6e}")
    
    # ==================== Per-Segment Rate Equation Plots ====================
    print("\n--- Per-Segment Rate Equation Plots ---")
    
    for seg_idx in seg_to_plot:
        sl, case_num, is_train = get_seg_info(seg_idx)
        split_str = "TRAIN" if is_train else "VAL"
        
        print(f"\n  Segment {seg_idx} (case={case_num}, {split_str})")
        
        # Extract segment data
        t_seg = time_axis[sl]
        
        # dZbar/dt comparison plot
        if ap.ion_available and truth_zbar is not None:
            zbar_seg_truth = truth_zbar[sl]
            zbar_seg_ae = pred_zbar_ae[sl]
            zbar_seg_sindy = pred_zbar_sindy[sl]
            
            dZbar_seg_truth = central_diff_np(zbar_seg_truth.reshape(-1, 1), dt).flatten()
            dZbar_seg_ae = central_diff_np(zbar_seg_ae.reshape(-1, 1), dt).flatten()
            dZbar_seg_sindy = central_diff_np(zbar_seg_sindy.reshape(-1, 1), dt).flatten()
            t_seg_diff = t_seg[1:-1]  # Exclude both ends for central differences
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI, constrained_layout=True)
            
            # dZbar/dt comparison
            axes[0].plot(t_seg_diff, dZbar_seg_truth, 'k-', lw=1.5, label='Truth')
            axes[0].plot(t_seg_diff, dZbar_seg_ae, 'C1-', lw=1.0, alpha=0.8, label='AE')
            axes[0].plot(t_seg_diff, dZbar_seg_sindy, 'C2--', lw=1.0, alpha=0.8, label='SINDy')
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("dZbar/dt")
            axes[0].set_title(f"dZbar/dt Comparison | seg{case_num} ({split_str})")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # dZbar/dt Error
            err_ae = dZbar_seg_ae - dZbar_seg_truth
            err_sindy = dZbar_seg_sindy - dZbar_seg_truth
            axes[1].plot(t_seg_diff, err_ae, 'C1-', lw=1.0, label='AE Error')
            axes[1].plot(t_seg_diff, err_sindy, 'C2--', lw=1.0, label='SINDy Error')
            axes[1].axhline(0, color='k', ls='--', lw=0.5)
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Error")
            axes[1].set_title("dZbar/dt Error")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            save_fig(f"rate_dZbar_seg{case_num}")
            show_plot()
# dCSD/dt heatmap
        if ap.ion_available and truth_csd is not None:
            csd_seg_truth = truth_csd[sl]
            csd_seg_ae = pred_csd_ae[sl]
            csd_seg_sindy = pred_csd_sindy[sl]
            
            dCSD_seg_truth = central_diff_np(csd_seg_truth, dt)
            dCSD_seg_ae = central_diff_np(csd_seg_ae, dt)
            dCSD_seg_sindy = central_diff_np(csd_seg_sindy, dt)
            
            vmax = np.percentile(np.abs(dCSD_seg_truth), 99)
            vmin = -vmax
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=DPI, constrained_layout=True)
            
            im0 = axes[0].imshow(dCSD_seg_truth.T, aspect='auto', origin='lower',
                                  cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[0].set_title(f"Truth dCSD/dt | seg{case_num}")
            axes[0].set_xlabel("Time step")
            axes[0].set_ylabel("Charge State")
            
            im1 = axes[1].imshow(dCSD_seg_ae.T, aspect='auto', origin='lower',
                                  cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[1].set_title(f"AE dCSD/dt | seg{case_num}")
            axes[1].set_xlabel("Time step")
            
            im2 = axes[2].imshow(dCSD_seg_sindy.T, aspect='auto', origin='lower',
                                  cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[2].set_title(f"SINDy dCSD/dt | seg{case_num}")
            axes[2].set_xlabel("Time step")
            
            fig.colorbar(im0, ax=axes.ravel().tolist(), label="dCSD/dt", 
                         shrink=0.8, pad=0.02)
            
            save_fig(f"rate_dCSD_seg{case_num}")
            show_plot()
            print("\n✅ Rate equation analysis complete")

#%%
# =============================================================================
# Adaptive SINDy Analysis (A(U), a(U) visualization)
# =============================================================================

if VIS_SECTIONS.get('adaptive_sindy', False) and use_adaptive_sindy and sindy_model is not None:
    print("\n" + "="*60)
    print(" Adaptive SINDy Analysis")
    print("="*60)
    
    # Extract U range (W-scaled)
    U_all_scaled = torch.tensor(U_all, dtype=dtype, device=device)
    
    T_scaled_min, T_scaled_max = U_all_scaled[:, 0].min().item(), U_all_scaled[:, 0].max().item()
    n_scaled_min, n_scaled_max = U_all_scaled[:, 1].min().item(), U_all_scaled[:, 1].max().item()
    
    print(f"Training U range (W-scaled):")
    print(f"  T: [{T_scaled_min:.4f}, {T_scaled_max:.4f}]")
    print(f"  n: [{n_scaled_min:.4f}, {n_scaled_max:.4f}]")
    
    # Helper function: W-scaled -> original scale
    def ctrl_inverse_transform(U_scaled):
        """Control scaler inverse transform (array-supported)."""
        U_scaled = np.atleast_2d(U_scaled)
        result = np.zeros_like(U_scaled)
        for j in range(U_scaled.shape[1]):
            for i in range(U_scaled.shape[0]):
                result[i, j] = ctrl_scaler.inverse_transform_single(U_scaled[i, j], j)
        return result
    
    # Convert to original scale
    U_original = ctrl_inverse_transform(U_all)
    T_orig_min, T_orig_max = U_original[:, 0].min(), U_original[:, 0].max()
    n_orig_min, n_orig_max = U_original[:, 1].min(), U_original[:, 1].max()
    
    print(f"Training U range (original scale):")
    print(f"  T: [{T_orig_min:.4e}, {T_orig_max:.4e}]")
    print(f"  n: [{n_orig_min:.4e}, {n_orig_max:.4e}]")
    
    nz = cfg.model.latent_dim
    grid_res = ADAPTIVE_GRID_RESOLUTION
    extrap_factor = ADAPTIVE_EXTRAP_FACTOR
    
    # ==========================================================
    # Boxes for each (T, n) point used in training (train/val separated)
    # ==========================================================
    # Convert all U points to the original scale
    U_all_orig = ctrl_inverse_transform(U_all)
    
    # Determine box size based on grid spacing
    T_all_orig = U_all_orig[:, 0]
    n_all_orig = U_all_orig[:, 1]
    
    # Extract unique (T, n) points and separate train/val
    train_points_orig = []  # [(T, n), ...]
    val_points_orig = []
    
    for seg_idx, sl in enumerate(segment_slices):
        U_seg_orig = U_all_orig[sl.start:sl.stop]
        is_train = seg_idx in train_seg_ids
        
        for t_val, n_val in U_seg_orig:
            if is_train:
                train_points_orig.append((t_val, n_val))
            else:
                val_points_orig.append((t_val, n_val))
    
    train_points_orig = np.array(train_points_orig) if train_points_orig else np.empty((0, 2))
    val_points_orig = np.array(val_points_orig) if val_points_orig else np.empty((0, 2))
    
    # Box size as a fraction of data range
    T_range_data = T_all_orig.max() - T_all_orig.min()
    n_range_data = n_all_orig.max() - n_all_orig.min()
    box_T_size = T_range_data * 0.02  # 2% of range
    box_n_size = n_range_data * 0.02
    
    def draw_point_boxes(ax, points, color, alpha=0.5, label=None):
        """Draw each (T, n) point as a small box."""
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        
        if len(points) == 0:
            return
        
        patches = []
        for T_val, n_val in points:
            rect = Rectangle(
                (n_val - box_n_size/2, T_val - box_T_size/2),
                box_n_size,
                box_T_size
            )
            patches.append(rect)
        
        pc = PatchCollection(patches, facecolor=color, edgecolor='none', alpha=alpha, label=label)
        ax.add_collection(pc)
        
        # Dummy handles for legend
        if label:
            ax.plot([], [], 's', color=color, alpha=alpha, label=label, markersize=8)
    
    # ==========================================================
    # 1. Training Range (Interpolation)
    # ==========================================================
    print("\n[1/2] Training range visualization...")
    
    # Create grid (W-scaled)
    T_grid_train = torch.linspace(T_scaled_min, T_scaled_max, grid_res, device=device, dtype=dtype)
    n_grid_train = torch.linspace(n_scaled_min, n_scaled_max, grid_res, device=device, dtype=dtype)
    
    # Original-scale grid for visualization
    T_grid_orig_train = np.linspace(T_orig_min, T_orig_max, grid_res)
    n_grid_orig_train = np.linspace(n_orig_min, n_orig_max, grid_res)
    
    # Meshgrid
    TT_train, NN_train = torch.meshgrid(T_grid_train, n_grid_train, indexing='ij')
    U_grid_train = torch.stack([TT_train.flatten(), NN_train.flatten()], dim=1)  # (grid_res^2, 2)
    
    # Compute coefficients
    with torch.no_grad():
        a_grid_train, A_grid_train = sindy_model.get_coefficients_batch(U_grid_train)
        # a_grid_train: (grid_res^2, nz)
        # A_grid_train: (grid_res^2, nz, nz)
        
        # Eigenvalues
        eigvals_train = sindy_model.get_eigenvalues_batch(U_grid_train)  # (grid_res^2, nz)
        max_real_train = eigvals_train.real.max(dim=1).values  # (grid_res^2,)
    
    a_grid_train_np = a_grid_train.cpu().numpy().reshape(grid_res, grid_res, nz)
    A_grid_train_np = A_grid_train.cpu().numpy().reshape(grid_res, grid_res, nz, nz)
    max_real_train_np = max_real_train.cpu().numpy().reshape(grid_res, grid_res)
    
    # A matrix components
    for i in range(nz):
        for j in range(nz):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI, constrained_layout=True)
            
            A_ij = A_grid_train_np[:, :, i, j]
            
            im = ax.pcolormesh(n_grid_orig_train, T_grid_orig_train, A_ij, 
                               shading='auto', cmap='RdBu_r')
            draw_point_boxes(ax, train_points_orig, color='blue', alpha=0.3, label='Train')
            draw_point_boxes(ax, val_points_orig, color='red', alpha=0.3, label='Val')
            ax.set_xlabel("Density (original scale)")
            ax.set_ylabel("Temperature (original scale)")
            ax.set_title(f"A[{i},{j}](T, n) - Training Range")
            ax.legend(loc='upper right')
            fig.colorbar(im, ax=ax, label=f"A[{i},{j}]")
            
            save_fig(f"adaptive_A_{i}{j}_train")
            show_plot()
# a vector components
    for i in range(nz):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI, constrained_layout=True)
        
        a_i = a_grid_train_np[:, :, i]
        
        im = ax.pcolormesh(n_grid_orig_train, T_grid_orig_train, a_i,
                           shading='auto', cmap='viridis')
        draw_point_boxes(ax, train_points_orig, color='blue', alpha=0.3, label='Train')
        draw_point_boxes(ax, val_points_orig, color='red', alpha=0.3, label='Val')
        ax.set_xlabel("Density (original scale)")
        ax.set_ylabel("Temperature (original scale)")
        ax.set_title(f"a[{i}](T, n) - Training Range")
        ax.legend(loc='upper right')
        fig.colorbar(im, ax=ax, label=f"a[{i}]")
        
        save_fig(f"adaptive_a_{i}_train")
        show_plot()
# Max eigenvalue real part
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI, constrained_layout=True)
    
    im = ax.pcolormesh(n_grid_orig_train, T_grid_orig_train, max_real_train_np,
                       shading='auto', cmap='RdBu_r', 
                       vmin=min(max_real_train_np.min(), -0.01),
                       vmax=max(max_real_train_np.max(), 0.01))
    draw_point_boxes(ax, train_points_orig, color='blue', alpha=0.3, label='Train')
    draw_point_boxes(ax, val_points_orig, color='red', alpha=0.3, label='Val')
    ax.set_xlabel("Density (original scale)")
    ax.set_ylabel("Temperature (original scale)")
    ax.set_title(f"max Re(λ) of A(T, n) - Training Range\n(all should be < 0 for stability)")
    ax.legend(loc='upper right')
    fig.colorbar(im, ax=ax, label="max Re(λ)")
    
    # Display zero contour
    ax.contour(n_grid_orig_train, T_grid_orig_train, max_real_train_np, 
               levels=[0], colors='white', linewidths=2, linestyles='--')
    
    save_fig("adaptive_eigenvalue_max_real_train")
    show_plot()
    print(f"  max Re(λ) in training range: [{max_real_train_np.min():.4e}, {max_real_train_np.max():.4e}]")
    print(f"  All stable: {max_real_train_np.max() < 0}")
    
    # ==========================================================
    # 2. Extrapolation Range
    # ==========================================================
    print("\n[2/2] Extrapolation range visualization...")
    
    # Extrapolation range (W-scaled)
    T_range = T_scaled_max - T_scaled_min
    n_range = n_scaled_max - n_scaled_min
    
    T_extrap_min = T_scaled_min - (extrap_factor - 1) * T_range / 2
    T_extrap_max = T_scaled_max + (extrap_factor - 1) * T_range / 2
    n_extrap_min = n_scaled_min - (extrap_factor - 1) * n_range / 2
    n_extrap_max = n_scaled_max + (extrap_factor - 1) * n_range / 2
    
    # Original-scale transform for extrapolation range
    T_extrap_orig_min = ctrl_inverse_transform([[T_extrap_min, n_scaled_min]])[0, 0]
    T_extrap_orig_max = ctrl_inverse_transform([[T_extrap_max, n_scaled_min]])[0, 0]
    n_extrap_orig_min = ctrl_inverse_transform([[T_scaled_min, n_extrap_min]])[0, 1]
    n_extrap_orig_max = ctrl_inverse_transform([[T_scaled_min, n_extrap_max]])[0, 1]
    
    print(f"Extrapolation U range (original scale):")
    print(f"  T: [{T_extrap_orig_min:.4e}, {T_extrap_orig_max:.4e}]")
    print(f"  n: [{n_extrap_orig_min:.4e}, {n_extrap_orig_max:.4e}]")
    
    # Create grid (extrapolation)
    T_grid_extrap = torch.linspace(T_extrap_min, T_extrap_max, grid_res, device=device, dtype=dtype)
    n_grid_extrap = torch.linspace(n_extrap_min, n_extrap_max, grid_res, device=device, dtype=dtype)
    
    # Original-scale grid for visualization
    T_grid_orig_extrap = np.linspace(T_extrap_orig_min, T_extrap_orig_max, grid_res)
    n_grid_orig_extrap = np.linspace(n_extrap_orig_min, n_extrap_orig_max, grid_res)
    
    # Meshgrid
    TT_extrap, NN_extrap = torch.meshgrid(T_grid_extrap, n_grid_extrap, indexing='ij')
    U_grid_extrap = torch.stack([TT_extrap.flatten(), NN_extrap.flatten()], dim=1)
    
    # Compute coefficients
    with torch.no_grad():
        a_grid_extrap, A_grid_extrap = sindy_model.get_coefficients_batch(U_grid_extrap)
        eigvals_extrap = sindy_model.get_eigenvalues_batch(U_grid_extrap)
        max_real_extrap = eigvals_extrap.real.max(dim=1).values
    
    a_grid_extrap_np = a_grid_extrap.cpu().numpy().reshape(grid_res, grid_res, nz)
    A_grid_extrap_np = A_grid_extrap.cpu().numpy().reshape(grid_res, grid_res, nz, nz)
    max_real_extrap_np = max_real_extrap.cpu().numpy().reshape(grid_res, grid_res)
    
    # A matrix components (extrapolation)
    for i in range(nz):
        for j in range(nz):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI, constrained_layout=True)
            
            A_ij = A_grid_extrap_np[:, :, i, j]
            
            im = ax.pcolormesh(n_grid_orig_extrap, T_grid_orig_extrap, A_ij,
                               shading='auto', cmap='RdBu_r')
            
            # Training data overlay (point boxes)
            draw_point_boxes(ax, train_points_orig, color='blue', alpha=0.3, label='Train')
            draw_point_boxes(ax, val_points_orig, color='red', alpha=0.3, label='Val')
            
            ax.set_xlabel("Density (original scale)")
            ax.set_ylabel("Temperature (original scale)")
            ax.set_title(f"A[{i},{j}](T, n) - Extrapolation")
            ax.legend(loc='upper right')
            fig.colorbar(im, ax=ax, label=f"A[{i},{j}]")
            
            save_fig(f"adaptive_A_{i}{j}_extrap")
            show_plot()
# a vector components (extrapolation)
    for i in range(nz):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI, constrained_layout=True)
        
        a_i = a_grid_extrap_np[:, :, i]
        
        im = ax.pcolormesh(n_grid_orig_extrap, T_grid_orig_extrap, a_i,
                           shading='auto', cmap='viridis')
        
        draw_point_boxes(ax, train_points_orig, color='blue', alpha=0.3, label='Train')
        draw_point_boxes(ax, val_points_orig, color='red', alpha=0.3, label='Val')
        
        ax.set_xlabel("Density (original scale)")
        ax.set_ylabel("Temperature (original scale)")
        ax.set_title(f"a[{i}](T, n) - Extrapolation")
        ax.legend(loc='upper right')
        fig.colorbar(im, ax=ax, label=f"a[{i}]")
        
        save_fig(f"adaptive_a_{i}_extrap")
        show_plot()
# Max eigenvalue (extrapolation)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI, constrained_layout=True)
    
    im = ax.pcolormesh(n_grid_orig_extrap, T_grid_orig_extrap, max_real_extrap_np,
                       shading='auto', cmap='RdBu_r',
                       vmin=min(max_real_extrap_np.min(), -0.01),
                       vmax=max(max_real_extrap_np.max(), 0.01))
    
    draw_point_boxes(ax, train_points_orig, color='blue', alpha=0.3, label='Train')
    draw_point_boxes(ax, val_points_orig, color='red', alpha=0.3, label='Val')
    
    ax.set_xlabel("Density (original scale)")
    ax.set_ylabel("Temperature (original scale)")
    ax.set_title(f"max Re(λ) of A(T, n) - Extrapolation\n(all should be < 0 for stability)")
    ax.legend(loc='upper right')
    fig.colorbar(im, ax=ax, label="max Re(λ)")
    
    # 0 contour
    ax.contour(n_grid_orig_extrap, T_grid_orig_extrap, max_real_extrap_np,
               levels=[0], colors='white', linewidths=2, linestyles='-')
    
    save_fig("adaptive_eigenvalue_max_real_extrap")
    show_plot()
    print(f"  max Re(λ) in extrapolation range: [{max_real_extrap_np.min():.4e}, {max_real_extrap_np.max():.4e}]")
    print(f"  All stable in extrapolation: {max_real_extrap_np.max() < 0}")
    
    # ==========================================================
    # 3. Steady-state comparison (if steady data available)
    # ==========================================================
    if steady_data is not None and steady_data.enabled:
        print("\n[Bonus] Steady-state Z*(U) comparison...")
        
        steady_W_t, steady_U_t = steady_data.to_torch(device, dtype)
        
        with torch.no_grad():
            # Encoded Z_ss
            steady_W4 = steady_W_t.unsqueeze(1).unsqueeze(1)
            Z_ss_encoded = ae.encoder(steady_W4)
            if Z_ss_encoded.dim() > 2:
                Z_ss_encoded = Z_ss_encoded.squeeze()
            
            # Predicted Z* = -A^{-1} a
            Z_star_pred = sindy_model.get_equilibrium_batch(steady_U_t)
        
        Z_ss_np = Z_ss_encoded.cpu().numpy()
        Z_star_np = Z_star_pred.cpu().numpy()
        U_ss_orig = ctrl_inverse_transform(steady_U_t.cpu().numpy())
        
        # Compare by Z component
        for i in range(nz):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI, constrained_layout=True)
            
            # Color by T
            sc = ax.scatter(Z_ss_np[:, i], Z_star_np[:, i], 
                           c=U_ss_orig[:, 0], cmap='plasma', alpha=0.7, s=30)
            
            # y=x line
            lim_min = min(Z_ss_np[:, i].min(), Z_star_np[:, i].min())
            lim_max = max(Z_ss_np[:, i].max(), Z_star_np[:, i].max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, label='y=x')
            
            ax.set_xlabel(f"Z_ss[{i}] (encoded)")
            ax.set_ylabel(f"Z*[{i}] (predicted = -A⁻¹a)")
            ax.set_title(f"Steady-state Latent Comparison: Z[{i}]")
            ax.legend()
            fig.colorbar(sc, ax=ax, label="Temperature (original)")
            
            save_fig(f"adaptive_steady_Z{i}_comparison")
            show_plot()
# Compute MSE
        mse_steady = np.mean((Z_ss_np - Z_star_np)**2)
        print(f"  Steady-state MSE (Z_ss vs Z*): {mse_steady:.4e}")
    
    print("\n✅ Adaptive SINDy analysis complete")


#%%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print(" Evaluation Complete!")
print("="*80)
print(f"\nModel: {CONFIG_DIR}")
print(f"Device: {device}")
print(f"Segments: {len(segment_slices)}")
print(f"Train: {len(train_seg_ids)}, Val: {len(val_seg_ids)}")

if use_adaptive_sindy:
    print(f"\nSINDy: Adaptive (CoefNet)")
    print(f"  Structure: dZ/dt = a(U) + A(U)*Z")
    print(f"  A(U) = -P(U)*P(U)^T - eps*I (Hurwitz guaranteed)")
    print(f"  eps: {cfg.sindy.adaptive_eps}")
    print(f"  hidden_dims: {cfg.sindy.adaptive_hidden}")
else:
    print(f"\nSINDy: lstsq")
    print(f"  Hurwitz stable: {is_hurwitz}")
    print(f"  max Re(lambda): {max_real:.3e}")
    print(f"  min Re(lambda): {min_real:.3e}")

if SAVE_PLOTS:
    print(f"\nPlots saved to: {plot_dir}")

print("\n" + "="*80)
print("Done!")
print("="*80)
# === Final wait after all figures are shown (terminal execution) ===
if SHOW_PLOTS:
    print("\n[Done] All plots are displayed. Close any figure window to exit.")
    plt.show(block=True)