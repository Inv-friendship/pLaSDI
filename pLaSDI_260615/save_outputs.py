# -*- coding: utf-8 -*-
"""
LaSDIc Output Saver - VSCode Edition
=====================================
Save outputs from a trained model in multiple formats.

Available data to save:
- Population (Truth/AE/SINDy)
- W-space (scaled)
- Z (latent)
- Fraction (normalized)
- CSD (Charge State Distribution)
- Zbar (mean charge)
- Control variables (T, density)
- Steady-state data

Usage:
    1. Open this file in VSCode.
    2. Edit the CONFIG section below.
    3. Execute cells with Shift+Enter.
"""
#%%
# =============================================================================
# CONFIG - edit only this section
# =============================================================================

# Path settings
CONFIG_DIR = "./runs/case1"          # Model checkpoint directory
OUTPUT_DIR = None                   # If None, use CONFIG_DIR/saved_outputs

# Best model selection: "train" or "val"
BEST_TYPE = "train"

# Select data types to save
SAVE_TYPES = {
    'population': True,    # Population (Truth/AE/SINDy)
    'W': True,             # W-space (scaled)
    'Z': True,             # Z (latent)
    'fraction': True,      # Fraction (normalized)
    'csd': True,           # CSD
    'zbar': True,          # Zbar
    'control': True,       # Control variables
    'steady': True,        # Steady-state
}

# Segment selection by case number
# None means all; a list selects only those cases
SELECTED_CASE_NUMBERS = None        # Example: [4, 10, 25] or None

# Save train/val separately
SPLIT_TRAIN_VAL = True              # True: separate train/val folders

# Whether to save SINDy predictions
SAVE_SINDY_PREDICTIONS = True

#%%
# =============================================================================
# Setup & Imports
# =============================================================================

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import LaSDIcConfig, create_default_config
from src.scaling import PopulationScaler, ControlScaler, TorchScaleHelper
from src.atomic_physics import AtomicPhysics
from src.data_utils import *

print("✅ Imports complete")

#%%
# =============================================================================
# Load Configuration & Setup
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

if OUTPUT_DIR is None:
    output_dir = Path(CONFIG_DIR) / "saved_outputs"
else:
    output_dir = Path(OUTPUT_DIR)

output_dir.mkdir(parents=True, exist_ok=True)

print(f"✅ Config loaded")
print(f"   Input:  {CONFIG_DIR}")
print(f"   Output: {output_dir}")

#%%
# =============================================================================
# Load Data
# =============================================================================

print("\n[Data] Loading...")

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

print(f"✅ Data loaded")
print(f"   Total segments: {len(segment_slices)}")
print(f"   Train: {len(train_seg_ids)}, Val: {len(val_seg_ids)}")

#%%
# =============================================================================
# Load Model
# =============================================================================

print("\n[Model] Loading...")

from src.autoencoder import Autoencoder

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
    print(f"✅ Model loaded from {_ckpt_path}")
    print(f"   {_ckpt_label}: epoch {best_epoch}")
else:
    raise FileNotFoundError(f"Checkpoint not found: {_ckpt_path}")

# SINDy setup
dt_eff = cfg.sindy.dt_eff if cfg.sindy.dt_eff else cfg.data.dt
use_adaptive_sindy = cfg.sindy.use_adaptive
sindy_model = None
ld = None
coef_vec_np = None

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
        print("⚠️ No sindy_model_state in checkpoint")
    sindy_model.eval()
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
print(f"   SINDy mode: {'Adaptive' if use_adaptive_sindy else 'lstsq'}")
print("✅ Model ready")

#%%
# =============================================================================
# Compute Predictions
# =============================================================================

print("\n[Compute] Generating predictions...")

# AE reconstruction
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

print("  ✓ AE predictions computed")

# SINDy predictions
if SAVE_SINDY_PREDICTIONS:
    print("  Computing SINDy predictions...")
    
    nz = cfg.model.latent_dim
    
    if use_adaptive_sindy:
        # Adaptive SINDy: simulate by segment
        Z_pred_sindy = np.zeros_like(Z_truth_np)
        
        for sl in segment_slices:
            L = sl.stop - sl.start
            z0 = Z_truth_np[sl.start]
            U_seg = U_all[sl.start:sl.stop]
            t_grid = np.linspace(0.0, (L - 1) * dt_eff, L)
            
            z0_t = torch.tensor(z0, dtype=dtype, device=device)
            U_seg_t = torch.tensor(U_seg, dtype=dtype, device=device)
            
            with torch.no_grad():
                Z_seg = sindy_model.simulate(z0_t, t_grid, U_seg_t)
            
            if isinstance(Z_seg, torch.Tensor):
                Z_seg = Z_seg.cpu().numpy()
            Z_pred_sindy[sl.start:sl.stop] = Z_seg
    else:
        # lstsq SINDy: compute global coefficients, then simulate
        
        # Check precomputed file
        _precomputed_path = Path(CONFIG_DIR) / "precomputed.npz"
        _has_precomputed = _precomputed_path.exists()
        _coef_key = 'sindy_coef_vec_val' if BEST_TYPE == 'val' else 'sindy_coef_vec_train'
        _coef_loaded = False
        
        if _has_precomputed:
            _pc = np.load(str(_precomputed_path), allow_pickle=True)
            if _coef_key in _pc:
                coef_vec_np = _pc[_coef_key]
                _coef_loaded = True
                print(f"    Coefficients loaded from precomputed.npz ({_coef_key})")
            elif 'sindy_coef_vec_train' in _pc:
                coef_vec_np = _pc['sindy_coef_vec_train']
                _coef_loaded = True
                print(f"    Coefficients loaded from precomputed.npz (sindy_coef_vec_train, fallback)")
        
        if not _coef_loaded:
            # Fallback: recompute lstsq using train data only
            print(f"    Calibrating lstsq from scratch (train segments only)...")
            Z_train_list, U_train_list = [], []
            for i in train_seg_ids:
                sl = segment_slices[i]
                Z_train_list.append(Z_truth_np[sl.start:sl.stop])
                U_train_list.append(U_all[sl.start:sl.stop])
            
            Z_train = np.vstack(Z_train_list)
            U_train = np.vstack(U_train_list)
            
            with torch.no_grad():
                Z_t = torch.tensor(Z_train, dtype=dtype, device=device)
                U_t = torch.tensor(U_train, dtype=dtype, device=device)
                coef_vec = ld.calibrate(Z_t, U_t, float(dt_eff), compute_loss=False, numpy=False)
                coef_vec_np = coef_vec.detach().cpu().numpy().reshape(-1)
        
        # Simulate all segments
        Z_pred_sindy = np.zeros_like(Z_truth_np)
        
        for sl in segment_slices:
            L = sl.stop - sl.start
            z0 = Z_truth_np[sl.start]
            U_seg = U_all[sl.start:sl.stop]
            t_grid = np.linspace(0.0, (L - 1) * dt_eff, L)
            
            Z_seg = ld.simulate(coef_vec_np, z0, t_grid, U=U_seg)
            Z_pred_sindy[sl.start:sl.stop] = Z_seg
    
    # Decode SINDy predictions
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
    
    print("  ✓ SINDy predictions computed")
else:
    pred_W_sindy = None
    pred_frac_sindy = None
    pred_pop_sindy = None
    Z_pred_sindy = None

# CSD & Zbar
if ap.ion_available:
    print("  Computing CSD & Zbar...")
    
    truth_csd = ap.compute_csd_numpy(truth_frac)
    pred_csd_ae = ap.compute_csd_numpy(pred_frac_ae)
    
    truth_zbar = ap.compute_zbar_numpy(truth_frac)
    pred_zbar_ae = ap.compute_zbar_numpy(pred_frac_ae)
    
    if SAVE_SINDY_PREDICTIONS:
        pred_csd_sindy = ap.compute_csd_numpy(pred_frac_sindy)
        pred_zbar_sindy = ap.compute_zbar_numpy(pred_frac_sindy)
    else:
        pred_csd_sindy = None
        pred_zbar_sindy = None
    
    print("  ✓ CSD & Zbar computed")
else:
    truth_csd = pred_csd_ae = pred_csd_sindy = None
    truth_zbar = pred_zbar_ae = pred_zbar_sindy = None

print("✅ All predictions computed")

#%%
# =============================================================================
# Helper Functions
# =============================================================================

def case_to_seg_idx(case_num):
    """Convert case number to segment index."""
    try:
        return case_numbers.index(case_num)
    except ValueError:
        print(f"⚠️  Case {case_num} not found")
        return None

def save_txt(path: Path, data: np.ndarray, header: str = ""):
    """Save as a text file."""
    with open(path, 'w') as f:
        if header:
            f.write(header + "\n")
        
        if data.ndim == 2:
            for row in data:
                line = " ".join(f"{val:.12e}" for val in row)
                f.write(line + "\n")
        elif data.ndim == 1:
            line = " ".join(f"{val:.12e}" for val in data)
            f.write(line + "\n")
        else:
            raise ValueError(f"Unsupported ndim: {data.ndim}")

#%%
# =============================================================================
# Determine segments to save
# =============================================================================

if SELECTED_CASE_NUMBERS is None:
    # Save all segments
    seg_to_save = list(range(len(segment_slices)))
else:
    # Convert case numbers to segment indices
    seg_to_save = []
    for case_num in SELECTED_CASE_NUMBERS:
        seg_idx = case_to_seg_idx(case_num)
        if seg_idx is not None:
            seg_to_save.append(seg_idx)
    
    if not seg_to_save:
        print("⚠️  No valid cases found, saving all segments")
        seg_to_save = list(range(len(segment_slices)))

print(f"\n[Save] Selected segments: {seg_to_save}")
print(f"       Corresponding cases: {[case_numbers[i] for i in seg_to_save]}")

#%%
# =============================================================================
# Save Segment Data
# =============================================================================

print("\n" + "="*60)
print(" Saving Segment Data")
print("="*60)

types_list = [k for k, v in SAVE_TYPES.items() if v and k != 'steady']

for seg_idx in seg_to_save:
    sl = segment_slices[seg_idx]
    case_num = case_numbers[seg_idx]
    
    # Determine subdirectory
    if SPLIT_TRAIN_VAL:
        if seg_idx in train_seg_ids:
            subdir = "train"
            split_str = "TRAIN"
        else:
            subdir = "val"
            split_str = "VAL"
    else:
        subdir = "all"
        split_str = "ALL"
    
    seg_dir = output_dir / subdir
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Case {case_num}] ({split_str}) → {subdir}/")
    
    # Population
    if 'population' in types_list:
        save_txt(
            seg_dir / f"population_seg{case_num}_truth.txt",
            truth_pop[sl],
            header=f"# Truth Population | case={case_num} | shape: {truth_pop[sl].shape}"
        )
        save_txt(
            seg_dir / f"population_seg{case_num}_pred_ae.txt",
            pred_pop_ae[sl],
            header=f"# AE Predicted Population | case={case_num}"
        )
        if SAVE_SINDY_PREDICTIONS:
            save_txt(
                seg_dir / f"population_seg{case_num}_pred_sindy.txt",
                pred_pop_sindy[sl],
                header=f"# SINDy Predicted Population | case={case_num}"
            )
        print(f"  ✓ Population")
    
    # W-space
    if 'W' in types_list:
        save_txt(
            seg_dir / f"W_scaled_seg{case_num}_truth.txt",
            truth_W[sl],
            header=f"# Truth W (scaled) | case={case_num}"
        )
        save_txt(
            seg_dir / f"W_scaled_seg{case_num}_pred_ae.txt",
            pred_W_ae[sl],
            header=f"# AE Predicted W | case={case_num}"
        )
        if SAVE_SINDY_PREDICTIONS:
            save_txt(
                seg_dir / f"W_scaled_seg{case_num}_pred_sindy.txt",
                pred_W_sindy[sl],
                header=f"# SINDy Predicted W | case={case_num}"
            )
        print(f"  ✓ W-space")
    
    # Fraction
    if 'fraction' in types_list:
        save_txt(
            seg_dir / f"fraction_seg{case_num}_truth.txt",
            truth_frac[sl],
            header=f"# Truth Fraction | case={case_num}"
        )
        save_txt(
            seg_dir / f"fraction_seg{case_num}_pred_ae.txt",
            pred_frac_ae[sl],
            header=f"# AE Predicted Fraction | case={case_num}"
        )
        if SAVE_SINDY_PREDICTIONS:
            save_txt(
                seg_dir / f"fraction_seg{case_num}_pred_sindy.txt",
                pred_frac_sindy[sl],
                header=f"# SINDy Predicted Fraction | case={case_num}"
            )
        print(f"  ✓ Fraction")
    
    # Z (latent)
    if 'Z' in types_list:
        save_txt(
            seg_dir / f"Z_latent_seg{case_num}_truth.txt",
            Z_truth_np[sl],
            header=f"# Truth Latent Z | case={case_num}"
        )
        if SAVE_SINDY_PREDICTIONS:
            save_txt(
                seg_dir / f"Z_latent_seg{case_num}_pred_sindy.txt",
                Z_pred_sindy[sl],
                header=f"# SINDy Predicted Z | case={case_num}"
            )
        print(f"  ✓ Z (latent)")
    
    # CSD
    if 'csd' in types_list and truth_csd is not None:
        save_txt(
            seg_dir / f"CSD_seg{case_num}_truth.txt",
            truth_csd[sl],
            header=f"# Truth CSD | case={case_num}"
        )
        save_txt(
            seg_dir / f"CSD_seg{case_num}_pred_ae.txt",
            pred_csd_ae[sl],
            header=f"# AE Predicted CSD | case={case_num}"
        )
        if SAVE_SINDY_PREDICTIONS:
            save_txt(
                seg_dir / f"CSD_seg{case_num}_pred_sindy.txt",
                pred_csd_sindy[sl],
                header=f"# SINDy Predicted CSD | case={case_num}"
            )
        print(f"  ✓ CSD")
    
    # Zbar
    if 'zbar' in types_list and truth_zbar is not None:
        save_txt(
            seg_dir / f"Zbar_seg{case_num}_truth.txt",
            truth_zbar[sl].reshape(-1, 1),
            header=f"# Truth Zbar | case={case_num}"
        )
        save_txt(
            seg_dir / f"Zbar_seg{case_num}_pred_ae.txt",
            pred_zbar_ae[sl].reshape(-1, 1),
            header=f"# AE Predicted Zbar | case={case_num}"
        )
        if SAVE_SINDY_PREDICTIONS:
            save_txt(
                seg_dir / f"Zbar_seg{case_num}_pred_sindy.txt",
                pred_zbar_sindy[sl].reshape(-1, 1),
                header=f"# SINDy Predicted Zbar | case={case_num}"
            )
        print(f"  ✓ Zbar")
    
    # Control
    if 'control' in types_list:
        U_raw = U_all_raw[sl]
        time = time_axis[sl].reshape(-1, 1)
        control_data = np.hstack([time, U_raw])
        
        save_txt(
            seg_dir / f"historyfile_seg{case_num}.txt",
            control_data,
            header="# time(s), Temperature(eV), Density(g/cc)"
        )
        print(f"  ✓ Control")

print("\n✅ Segment data saved")

#%%
# =============================================================================
# Save Steady-State Data
# =============================================================================

if SAVE_TYPES['steady'] and steady_data and steady_data.enabled:
    print("\n" + "="*60)
    print(" Saving Steady-State Data")
    print("="*60)
    
    steady_dir = output_dir / "steady"
    steady_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Truth ----
    truth_F_steady = steady_data.P_all / steady_data.P_all.sum(axis=1, keepdims=True)
    
    save_txt(
        steady_dir / "steady_population_truth.txt",
        steady_data.P_all,
        header=f"# Steady-state Truth Population | shape: {steady_data.P_all.shape}"
    )
    save_txt(
        steady_dir / "steady_W_scaled_truth.txt",
        steady_data.W_all,
        header=f"# Steady-state Truth W (scaled) | shape: {steady_data.W_all.shape}"
    )
    save_txt(
        steady_dir / "steady_fraction_truth.txt",
        truth_F_steady,
        header=f"# Steady-state Truth Fraction | shape: {truth_F_steady.shape}"
    )
    save_txt(
        steady_dir / "steady_history.txt",
        steady_data.Uraw_all,
        header="# Temperature(eV), Density(g/cc)"
    )
    print(f"  ✓ Truth (Population, W, Fraction, Control)")
    
    # ---- AE Reconstruction: truth W → encode → decode ----
    with torch.no_grad():
        steady_W_t = torch.tensor(steady_data.W_all, dtype=dtype, device=device)
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
        Z_steady_ae_np = Z_steady_ae.cpu().numpy()
    
    save_txt(
        steady_dir / "steady_Z_latent_ae.txt",
        Z_steady_ae_np,
        header=f"# Steady-state Z (from AE encode) | shape: {Z_steady_ae_np.shape}"
    )
    save_txt(
        steady_dir / "steady_W_scaled_pred_ae.txt",
        pred_W_steady_ae,
        header=f"# Steady-state AE Reconstructed W | shape: {pred_W_steady_ae.shape}"
    )
    save_txt(
        steady_dir / "steady_fraction_pred_ae.txt",
        pred_F_steady_ae,
        header=f"# Steady-state AE Reconstructed Fraction | shape: {pred_F_steady_ae.shape}"
    )
    print(f"  ✓ AE Reconstruction (Z, W, Fraction)")
    
    # ---- SINDy Equilibrium: Z* = -A^{-1}(a + B*U) ----
    pred_F_steady_eq = None
    pred_W_steady_eq = None
    Z_star_np = None
    
    if SAVE_SINDY_PREDICTIONS:
        try:
            if use_adaptive_sindy:
                # Adaptive: Z* = -A(U)^{-1} * a(U)
                steady_U_t = torch.tensor(steady_data.U_all, dtype=dtype, device=device)
                with torch.no_grad():
                    Z_star_t = sindy_model.get_equilibrium_batch(steady_U_t)
                    Z_star_np = Z_star_t.cpu().numpy()
                    
                    W_pred_eq = ae.decoder(Z_star_t)
                    if W_pred_eq.dim() == 4:
                        W_pred_eq = W_pred_eq[:, 0, 0, :]
                    elif W_pred_eq.dim() == 3:
                        W_pred_eq = W_pred_eq[:, 0, :]
                    
                    pred_W_steady_eq = W_pred_eq.cpu().numpy()
                    pred_F_steady_eq = scale_helper.W_to_fraction(
                        W_pred_eq.unsqueeze(1).unsqueeze(1)
                    ).cpu().numpy().reshape(-1, cfg.data.nx)
            else:
                # lstsq: Z* = -A^{-1} * (a + B*U)
                nz = cfg.model.latent_dim
                nu = mu
                p = 1 + nz + nu
                C_mat = coef_vec_np.reshape(p, nz)
                a_global = C_mat[0, :]
                A_global = C_mat[1:1+nz, :].T
                B_global = C_mat[1+nz:, :].T if nu > 0 else np.zeros((nz, 0))
                
                U_steady = steady_data.U_all
                Z_star_np = -np.linalg.solve(
                    A_global, (a_global + U_steady @ B_global.T).T
                ).T
                
                with torch.no_grad():
                    Z_star_t = torch.tensor(Z_star_np, dtype=dtype, device=device)
                    W_pred_eq = ae.decoder(Z_star_t)
                    if W_pred_eq.dim() == 4:
                        W_pred_eq = W_pred_eq[:, 0, 0, :]
                    elif W_pred_eq.dim() == 3:
                        W_pred_eq = W_pred_eq[:, 0, :]
                    
                    pred_W_steady_eq = W_pred_eq.cpu().numpy()
                    pred_F_steady_eq = scale_helper.W_to_fraction(
                        W_pred_eq.unsqueeze(1).unsqueeze(1)
                    ).cpu().numpy().reshape(-1, cfg.data.nx)
            
            save_txt(
                steady_dir / "steady_Z_latent_eq.txt",
                Z_star_np,
                header=f"# Steady-state Z* (SINDy equilibrium) | shape: {Z_star_np.shape}"
            )
            save_txt(
                steady_dir / "steady_W_scaled_pred_eq.txt",
                pred_W_steady_eq,
                header=f"# Steady-state Equilibrium W | shape: {pred_W_steady_eq.shape}"
            )
            save_txt(
                steady_dir / "steady_fraction_pred_eq.txt",
                pred_F_steady_eq,
                header=f"# Steady-state Equilibrium Fraction | shape: {pred_F_steady_eq.shape}"
            )
            print(f"  ✓ SINDy Equilibrium (Z*, W, Fraction)")
        except Exception as e:
            print(f"  ⚠️  SINDy equilibrium failed: {e}")
    
    # ---- CSD & Zbar (Truth / AE / Equilibrium) ----
    if ap.ion_available:
        truth_csd_steady = ap.compute_csd_numpy(truth_F_steady)
        truth_zbar_steady = ap.compute_zbar_numpy(truth_F_steady)
        pred_csd_steady_ae = ap.compute_csd_numpy(pred_F_steady_ae)
        pred_zbar_steady_ae = ap.compute_zbar_numpy(pred_F_steady_ae)
        
        save_txt(
            steady_dir / "steady_CSD_truth.txt",
            truth_csd_steady,
            header=f"# Steady-state Truth CSD | shape: {truth_csd_steady.shape}"
        )
        save_txt(
            steady_dir / "steady_CSD_pred_ae.txt",
            pred_csd_steady_ae,
            header=f"# Steady-state AE CSD | shape: {pred_csd_steady_ae.shape}"
        )
        save_txt(
            steady_dir / "steady_Zbar_truth.txt",
            truth_zbar_steady.reshape(-1, 1),
            header=f"# Steady-state Truth Zbar"
        )
        save_txt(
            steady_dir / "steady_Zbar_pred_ae.txt",
            pred_zbar_steady_ae.reshape(-1, 1),
            header=f"# Steady-state AE Zbar"
        )
        
        if pred_F_steady_eq is not None:
            pred_csd_steady_eq = ap.compute_csd_numpy(pred_F_steady_eq)
            pred_zbar_steady_eq = ap.compute_zbar_numpy(pred_F_steady_eq)
            
            save_txt(
                steady_dir / "steady_CSD_pred_eq.txt",
                pred_csd_steady_eq,
                header=f"# Steady-state Equilibrium CSD | shape: {pred_csd_steady_eq.shape}"
            )
            save_txt(
                steady_dir / "steady_Zbar_pred_eq.txt",
                pred_zbar_steady_eq.reshape(-1, 1),
                header=f"# Steady-state Equilibrium Zbar"
            )
        
        print(f"  ✓ CSD & Zbar (Truth / AE / Equilibrium)")
    
    print("\n✅ Steady-state data saved")

#%%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print(" Save Complete!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print(f"\nSaved segments: {len(seg_to_save)}")
print(f"  Train: {sum(1 for i in seg_to_save if i in train_seg_ids)}")
print(f"  Val:   {sum(1 for i in seg_to_save if i in val_seg_ids)}")

print(f"\nSaved data types:")
for k, v in SAVE_TYPES.items():
    status = "✓" if v else "✗"
    print(f"  {status} {k}")

if SAVE_SINDY_PREDICTIONS:
    print(f"\n✓ SINDy predictions included ({'Adaptive' if use_adaptive_sindy else 'lstsq'})")

print("\n" + "="*80)
print("✅ Done!")
print("="*80)
