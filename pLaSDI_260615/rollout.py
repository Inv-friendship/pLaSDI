# -*- coding: utf-8 -*-
"""
LaSDIc Rollout & Expand Viewer
===============================
Dedicated script for SINDy rollout and time extension (expand)

Features:
- SINDy rollout for selected cases (Z0 -> Z(t)).
- Extend time beyond the original window with fixed controls.
- Visualize Population, W-space, CSD, and Zbar (Truth vs Rollout).
- Compare CSD/Zbar at the final expanded time against SS reference data.
  - If no exact (T, rho) match exists, use interpolation and report it.
- Save result files.

Usage:
    1. Edit the CONFIG section below.
    2. Execute with Shift+Enter or run the whole file.
"""

#%%
# =============================================================================
# CONFIG - edit only this section
# =============================================================================

# Path settings
CONFIG_DIR = "./runs/case1"

# Best model selection: "train" or "val"
BEST_TYPE = "train"

# Visualization
SHOW_PLOTS = True
SAVE_PLOTS = False
DPI = 150

# ===== Mode 1: select cases from config (cases included in train/val) =====
# List of case numbers. If None, use only the first case.
ROLLOUT_CASE_NUMBERS = None  # Example: [4, 10, 25] or None

# ===== Mode 2: test cases specified by seg number (paths resolved from data_dir) =====
# Arbitrary seg-number list independent of config.
TEST_SEG_NUMBERS = None      # Example: [100, 101, 102] or None
TEST_DATA_DIR = None         # If None, use cfg.data.data_dir
TEST_HISTORY_DIR = None      # If None, use cfg.data.history_dir

# ===== Mode 2b: test cases specified by explicit file paths =====
# List of (population_file, history_file) pairs.
TEST_FILES = None            # Example: [("path/to/pop.txt", "path/to/hist.txt"), ...] or None

# Extension settings
EXTEND_NS = 5.0  # Extension time (ns)

# Saving
SAVE_DATA = False  # Save result txt files

#%%
# =============================================================================
# Setup & Imports
# =============================================================================

import os
import sys
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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import LaSDIcConfig, create_default_config
from src.scaling import PopulationScaler, ControlScaler, TorchScaleHelper
from src.atomic_physics import AtomicPhysics
from src.data_utils import *

print("✅ Imports complete")

#%%
# =============================================================================
# Load Configuration
# =============================================================================

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
    plot_dir = Path(CONFIG_DIR) / "rollout_plots"
    plot_dir.mkdir(exist_ok=True)
else:
    plot_dir = None

if SAVE_DATA:
    data_dir = Path(CONFIG_DIR) / "rollout_data"
    data_dir.mkdir(exist_ok=True)
else:
    data_dir = None

def save_fig(name):
    if plot_dir is not None:
        plt.savefig(plot_dir / f"{name}.png", dpi=DPI, bbox_inches='tight')

def save_txt(path, arr, header=""):
    np.savetxt(str(path), arr, fmt="%.12e", header=header)

print(f"   Device: {device}")

#%%
# =============================================================================
# Load Data (supports precomputed fast-load)
# =============================================================================

# Check precomputed file
_precomputed_path = Path(CONFIG_DIR) / "precomputed.npz"
_has_precomputed = _precomputed_path.exists()

# Decide whether config cases are needed (test-only does not require full data load)
_need_config_cases = (
    TEST_SEG_NUMBERS is None and TEST_FILES is None
) or ROLLOUT_CASE_NUMBERS is not None

_need_full_data = _need_config_cases or not _has_precomputed

if _need_full_data:
    print("\n[Data] Loading full data...")
    
    pops = load_or_build_pops(cfg.data_files, cfg.data.nx, cfg.data.data_dir)
    pops = [np.asarray(p, dtype=np.float64) for p in pops]
    pop = np.concatenate(pops, axis=0)
    
    tmp = np.sum(pop, axis=1, keepdims=True)
    pop = pop / tmp + cfg.data.pop_lim
    pop = pop * tmp
    
    pop_scaler = PopulationScaler(eps=cfg.data.pop_lim, normalize=True)
    W_all = pop_scaler.fit_transform(pop, axis=1)
    W_all_exp = np.expand_dims(W_all, axis=1)
    X_frames = torch.tensor(W_all_exp[:, None, :, :], dtype=dtype)
    
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
    
    segment_slices = build_segment_slices(pops)
    case_numbers = cfg.case_numbers
    
    (train_idx, val_idx, train_slices, val_slices,
     train_seg_ids, val_seg_ids) = split_train_val_random_segments(
        segment_slices, cfg.train.val_ratio, seed=cfg.seed
    )
    
    print(f"✅ Full data loaded: {nt_total} timesteps, {len(segment_slices)} segments")

else:
    # ===== Fast path: restore scalers from precomputed.npz and skip full data load =====
    print(f"\n[Data] Fast-load from precomputed.npz (test cases only)")
    _pc = np.load(str(_precomputed_path), allow_pickle=True)
    
    pop_scaler = PopulationScaler.from_state(_pc['pop_scaler_state'].item())
    ctrl_scaler = ControlScaler.from_state(_pc['ctrl_scaler_state'].item())
    scale_helper = TorchScaleHelper(pop_scaler, dtype)
    
    mu = len(ctrl_scaler.col_params)
    case_numbers = cfg.case_numbers
    
    # Dummy placeholders (config cases are unused)
    pops = None
    pop = None
    W_all = None
    X_frames = None
    nA_all = None
    U_all = None
    U_all_raw = None
    segment_slices = None
    train_seg_ids = []
    val_seg_ids = []
    train_idx = []
    val_idx = []
    nt_total = 0
    
    print(f"✅ Scalers restored from precomputed.npz (full data load skipped)")

# Atomic physics
state_names = load_state_names(cfg.data.names_file, cfg.data.nx)
ap = AtomicPhysics(state_names, cfg.data.nx, dtype)

# Steady-state
steady_data = None
if cfg.data.steady_enable and _need_full_data:
    steady_data = SteadyStateData(
        cfg.steady_pop_hist_pairs, pop_scaler, ctrl_scaler,
        random_pick=cfg.data.steady_random_pick,
        num_samples=cfg.data.steady_num_samples,
        seed=cfg.data.steady_random_seed,
        pop_lim=cfg.data.pop_lim
    )

print(f"✅ Data setup complete")

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

# Select best model path
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

dt_eff = cfg.sindy.dt_eff if cfg.sindy.dt_eff else cfg.data.dt

# Load SINDy model
use_adaptive_sindy = cfg.sindy.use_adaptive
sindy_model = None
ld = None
coef_vec_np = None

if use_adaptive_sindy:
    from src.sindyc_adaptive import AdaptiveSINDyC
    sindy_model = AdaptiveSINDyC(
        nz=cfg.model.latent_dim,
        mu=2,
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
        nt=max(len(train_idx), 10),
        fd_type=cfg.sindy.fd_type,
        use_global_coefs=cfg.sindy.use_global_coefs
    )
    # Set mu (needed by simulate)
    ld._set_mu(mu)

ae.eval()
print(f"   SINDy mode: {'Adaptive' if use_adaptive_sindy else 'lstsq'}")
print("✅ Model ready")

#%%
# =============================================================================
# lstsq: global coefficient calibration (prefer precomputed, fallback to recomputation)
# =============================================================================

if not use_adaptive_sindy:
    nz = cfg.model.latent_dim
    nu = mu
    
    # Try loading from precomputed.npz
    _coef_key = 'sindy_coef_vec_val' if BEST_TYPE == 'val' else 'sindy_coef_vec_train'
    _coef_loaded = False
    
    if _has_precomputed:
        _pc = np.load(str(_precomputed_path), allow_pickle=True)
        if _coef_key in _pc:
            coef_vec_np = _pc[_coef_key]
            _coef_loaded = True
            print(f"\n[SINDy] Coefficients loaded from precomputed.npz ({_coef_key})")
        elif 'sindy_coef_vec_train' in _pc:
            coef_vec_np = _pc['sindy_coef_vec_train']
            _coef_loaded = True
            print(f"\n[SINDy] Coefficients loaded from precomputed.npz (sindy_coef_vec_train, fallback)")
    
    if not _coef_loaded:
        # Fallback: encode full data and recompute lstsq
        print("\n[SINDy] Calibrating global coefficients (train segments only)...")
        
        if X_frames is None:
            raise RuntimeError(
                "precomputed.npz is missing and full data was not loaded. "
                "Use config cases or rerun train.py."
            )
        
        with torch.no_grad():
            X_dev = X_frames.to(device=device, dtype=dtype)
            Z_all_enc = ae.encoder(X_dev)
            if Z_all_enc.dim() == 4:
                Z_all_enc = Z_all_enc[:, 0, 0, :]
            elif Z_all_enc.dim() == 3:
                Z_all_enc = Z_all_enc[:, 0, :]
            Z_all_np = Z_all_enc.cpu().numpy()
        
        Z_train_list, U_train_list = [], []
        for i in train_seg_ids:
            sl = segment_slices[i]
            Z_train_list.append(Z_all_np[sl.start:sl.stop])
            U_train_list.append(U_all[sl.start:sl.stop])
        
        Z_tr = torch.tensor(np.vstack(Z_train_list), dtype=dtype, device=device)
        U_tr = torch.tensor(np.vstack(U_train_list), dtype=dtype, device=device)
        coef_vec_np = ld.calibrate(Z_tr, U_tr, float(dt_eff), compute_loss=False, numpy=True)
        print("✅ Global coefficients calibrated (from scratch)")
    
    # Split coefficients: lstsq output is a row-major flattened (p, nz) matrix.
    # C[0, :] = a, C[1:1+nz, :] = A^T (lstsq convention), C[1+nz:, :] = B^T.
    # Split using the same convention as evaluate.py.
    p = 1 + nz + nu
    C_mat = coef_vec_np.reshape(p, nz)
    a_global = C_mat[0, :]
    A_global = C_mat[1:1+nz, :].T      # (nz, nz) - mathematical A matrix
    B_global = C_mat[1+nz:, :].T if nu > 0 else np.zeros((nz, 0))  # (nz, nu)
    
    print(f"✅ Global coefficients ready (nz={nz}, nu={nu})")

#%%
# =============================================================================
# Prepare Steady-State Reference
# =============================================================================

ss_ref_available = False
ss_Uraw = None  # (n_ss, 2) - original T, density
ss_frac = None   # (n_ss, nx)
ss_csd = None    # (n_ss, n_charges)
ss_zbar = None   # (n_ss,)

if steady_data is not None and steady_data.enabled:
    ss_Uraw = steady_data.Uraw_all  # (n_ss, 2) original scale
    ss_P = steady_data.P_all        # (n_ss, nx)
    ss_frac = ss_P / ss_P.sum(axis=1, keepdims=True)
    
    if ap.ion_available:
        ss_csd = ap.compute_csd_numpy(ss_frac)
        ss_zbar = ap.compute_zbar_numpy(ss_frac)
    
    ss_ref_available = True
    print(f"\n[SS Reference] {ss_Uraw.shape[0]} steady-state samples loaded")
    print(f"   T range:   [{ss_Uraw[:, 0].min():.4e}, {ss_Uraw[:, 0].max():.4e}]")
    print(f"   rho range: [{ss_Uraw[:, 1].min():.4e}, {ss_Uraw[:, 1].max():.4e}]")
else:
    print("\n[SS Reference] No steady-state data available")


def find_ss_reference(T_target, rho_target, tol_T=0.01, tol_rho=0.01):
    """
    Find or interpolate CSD/Zbar for (T, rho) in steady-state reference data.
    
    Returns:
        dict with keys: csd, zbar, frac, method ('exact'|'interpolation'), 
              nearest_T, nearest_rho, distance
    """
    if not ss_ref_available or ss_Uraw is None:
        return None
    
    # Compute relative distance
    T_vals = ss_Uraw[:, 0]
    rho_vals = ss_Uraw[:, 1]
    
    rel_T = np.abs(T_vals - T_target) / (np.abs(T_target) + 1e-30)
    rel_rho = np.abs(rho_vals - rho_target) / (np.abs(rho_target) + 1e-30)
    
    # Nearest point
    dist = np.sqrt(rel_T**2 + rel_rho**2)
    nearest_idx = np.argmin(dist)
    nearest_T = T_vals[nearest_idx]
    nearest_rho = rho_vals[nearest_idx]
    nearest_dist = dist[nearest_idx]
    
    result = {
        'nearest_T': nearest_T,
        'nearest_rho': nearest_rho,
        'distance': nearest_dist,
    }
    
    # Check for an exact matching point
    if rel_T[nearest_idx] < tol_T and rel_rho[nearest_idx] < tol_rho:
        result['method'] = 'exact'
        result['csd'] = ss_csd[nearest_idx] if ss_csd is not None else None
        result['zbar'] = ss_zbar[nearest_idx] if ss_zbar is not None else None
        result['frac'] = ss_frac[nearest_idx] if ss_frac is not None else None
        return result
    
    # Try interpolation
    # Interpolate in log scale because T and rho vary exponentially
    log_T = np.log10(T_vals + 1e-30)
    log_rho = np.log10(rho_vals + 1e-30)
    log_T_target = np.log10(T_target + 1e-30)
    log_rho_target = np.log10(rho_target + 1e-30)
    
    points = np.column_stack([log_T, log_rho])
    target_pt = np.array([[log_T_target, log_rho_target]])
    
    result['method'] = 'interpolation'
    
    try:
        if ss_csd is not None:
            interp_csd = LinearNDInterpolator(points, ss_csd)
            csd_val = interp_csd(target_pt)
            if np.any(np.isnan(csd_val)):
                # Extrapolation: nearest neighbor fallback
                result['method'] = 'extrapolation (nearest)'
                nearest_interp = NearestNDInterpolator(points, ss_csd)
                csd_val = nearest_interp(target_pt)
            result['csd'] = csd_val.flatten()
        
        if ss_zbar is not None:
            interp_zbar = LinearNDInterpolator(points, ss_zbar.reshape(-1, 1))
            zbar_val = interp_zbar(target_pt)
            if np.any(np.isnan(zbar_val)):
                nearest_interp = NearestNDInterpolator(points, ss_zbar.reshape(-1, 1))
                zbar_val = nearest_interp(target_pt)
            result['zbar'] = float(zbar_val.flatten()[0])
        
        if ss_frac is not None:
            interp_frac = LinearNDInterpolator(points, ss_frac)
            frac_val = interp_frac(target_pt)
            if np.any(np.isnan(frac_val)):
                nearest_interp = NearestNDInterpolator(points, ss_frac)
                frac_val = nearest_interp(target_pt)
            result['frac'] = frac_val.flatten()
    except Exception as e:
        print(f"  ⚠️  Interpolation failed: {e}")
        # Fallback to nearest
        result['method'] = 'nearest (fallback)'
        result['csd'] = ss_csd[nearest_idx] if ss_csd is not None else None
        result['zbar'] = ss_zbar[nearest_idx] if ss_zbar is not None else None
        result['frac'] = ss_frac[nearest_idx] if ss_frac is not None else None
    
    return result

#%%
# =============================================================================
# Rollout Helper
# =============================================================================

def rollout_sindy(z0, U_total, total_len):
    """SINDy rollout with automatic adaptive/lstsq dispatch."""
    from scipy.integrate import solve_ivp
    
    t_grid_sindy = np.arange(total_len) * dt_eff
    
    if use_adaptive_sindy:
        U_total_t = torch.tensor(U_total, dtype=dtype, device=device)
        
        def _ode(t, z):
            t_idx = int(min(t / dt_eff, total_len - 1))
            U_t = U_total_t[t_idx].unsqueeze(0)
            with torch.no_grad():
                a_t, A_t = sindy_model.get_coefficients_batch(U_t)
            a_np = a_t.squeeze().cpu().numpy()
            A_np = A_t.squeeze().cpu().numpy()
            return a_np + z @ A_np.T
        
        sol = solve_ivp(_ode, [0, (total_len - 1) * dt_eff], z0, 
                        t_eval=t_grid_sindy, method='RK45')
        return sol.y.T
    else:
        return ld.simulate(coef_vec_np, z0, t_grid_sindy, U=U_total)


def decode_Z(Z_np):
    """Z → W, fraction, population, CSD, Zbar"""
    Z_t = torch.tensor(Z_np, dtype=dtype, device=device)
    with torch.no_grad():
        W_t = ae.decoder(Z_t)
        if W_t.dim() == 4:
            W_t = W_t[:, 0, 0, :]
        elif W_t.dim() == 3:
            W_t = W_t[:, 0, :]
        
        W_np = W_t.cpu().numpy()
        frac_np = scale_helper.W_to_fraction(
            W_t.unsqueeze(1).unsqueeze(1)
        ).cpu().numpy().reshape(-1, cfg.data.nx)
    
    result = {'W': W_np, 'frac': frac_np}
    
    # Population = frac * nA (rollout has no nA in expanded region, so caller uses last value)
    result['pop'] = None  # Caller must multiply by nA
    
    if ap.ion_available:
        result['csd'] = ap.compute_csd_numpy(frac_np)
        result['zbar'] = ap.compute_zbar_numpy(frac_np)
    else:
        result['csd'] = None
        result['zbar'] = None
    
    return result

#%%
# =============================================================================
# Select cases - unify config cases and test cases
# =============================================================================

# Normalize each case to a dict: {label, pop, U_raw, U_scaled, nA, truth_frac, W, X_frame, split}

cases_to_run = []

# --- Mode 1: config cases ---
def case_to_seg_idx(case_num):
    """Convert case number -> segment index."""
    for i, cn in enumerate(case_numbers):
        if cn == case_num:
            return i
    return None

if ROLLOUT_CASE_NUMBERS is not None:
    _config_seg_indices = []
    for cn in ROLLOUT_CASE_NUMBERS:
        idx = case_to_seg_idx(cn)
        if idx is not None:
            _config_seg_indices.append(idx)
        else:
            print(f"⚠️  Case {cn} not found in case_numbers={case_numbers}, skipping")
elif TEST_SEG_NUMBERS is None and TEST_FILES is None:
    # If nothing is specified, use the first config case
    _config_seg_indices = [0]
else:
    _config_seg_indices = []

for seg_idx in _config_seg_indices:
    sl = segment_slices[seg_idx]
    cn = case_numbers[seg_idx]
    is_train = seg_idx in train_seg_ids
    
    cases_to_run.append({
        'label': f"seg{cn}",
        'case_num': cn,
        'split': "TRAIN" if is_train else "VAL",
        'pop': pop[sl],
        'U_raw': U_all_raw[sl],
        'U_scaled': U_all[sl],
        'nA': nA_all[sl],
        'truth_frac': (pop / pop.sum(axis=1, keepdims=True))[sl],
        'W': W_all[sl],
        'X_frame': X_frames[sl.start:sl.start+1],
    })

# --- Mode 2: test cases by seg number ---
def _load_test_case(pop_path, hist_path, label):
    """Load an external file pair and build a case dict using the training-data scaler unchanged."""
    raw_pop = load_pop_matrix_auto(Path(pop_path), nx=cfg.data.nx)
    raw_pop = np.asarray(raw_pop, dtype=np.float64)
    
    # Apply pop_lim
    tmp = np.sum(raw_pop, axis=1, keepdims=True)
    raw_pop = raw_pop / tmp + cfg.data.pop_lim
    raw_pop = raw_pop * tmp
    
    # W transform using the fitted training scaler
    W_test = pop_scaler.transform(raw_pop)
    nA_test = np.sum(raw_pop, axis=1, keepdims=True)
    frac_test = raw_pop / raw_pop.sum(axis=1, keepdims=True)
    
    # X_frame (first time point)
    W_exp = np.expand_dims(W_test, axis=1)
    X_frame_test = torch.tensor(W_exp[:1, None, :, :], dtype=dtype)
    
    # Control
    t_h, U_h = load_history_file(hist_path)
    L = raw_pop.shape[0]
    t_seg = np.arange(L) * cfg.data.dt
    if len(t_h) != L:
        U_h = align_controls(t_h, U_h, t_seg)
    U_raw_test = U_h.astype(np.float64)
    U_scaled_test = ctrl_scaler.transform(U_raw_test)
    
    return {
        'label': label,
        'case_num': label,
        'split': "TEST",
        'pop': raw_pop,
        'U_raw': U_raw_test,
        'U_scaled': U_scaled_test,
        'nA': nA_test,
        'truth_frac': frac_test,
        'W': W_test,
        'X_frame': X_frame_test,
    }

if TEST_SEG_NUMBERS is not None:
    _test_data_dir = TEST_DATA_DIR if TEST_DATA_DIR else cfg.data.data_dir
    _test_hist_dir = TEST_HISTORY_DIR if TEST_HISTORY_DIR else cfg.data.history_dir
    
    for seg_num in TEST_SEG_NUMBERS:
        pop_path = os.path.join(_test_data_dir, f"density_population_seg{seg_num}.txt")
        hist_path = os.path.join(_test_hist_dir, f"historyfile_seg{seg_num}.txt")
        
        if not os.path.exists(pop_path):
            print(f"⚠️  Test seg {seg_num}: population file not found: {pop_path}")
            continue
        if not os.path.exists(hist_path):
            print(f"⚠️  Test seg {seg_num}: history file not found: {hist_path}")
            continue
        
        try:
            case = _load_test_case(pop_path, hist_path, f"test_seg{seg_num}")
            cases_to_run.append(case)
            print(f"  ✅ Test seg {seg_num} loaded ({case['pop'].shape[0]} timesteps)")
        except Exception as e:
            print(f"⚠️  Test seg {seg_num} load failed: {e}")

# --- Mode 2b: test cases by explicit file paths ---
if TEST_FILES is not None:
    for fi, (pop_path, hist_path) in enumerate(TEST_FILES):
        if not os.path.exists(pop_path):
            print(f"⚠️  TEST_FILES[{fi}]: population file not found: {pop_path}")
            continue
        if not os.path.exists(hist_path):
            print(f"⚠️  TEST_FILES[{fi}]: history file not found: {hist_path}")
            continue
        
        try:
            label = f"test_file{fi}"
            case = _load_test_case(pop_path, hist_path, label)
            cases_to_run.append(case)
            print(f"  ✅ {label} loaded ({case['pop'].shape[0]} timesteps)")
        except Exception as e:
            print(f"⚠️  TEST_FILES[{fi}] load failed: {e}")

print(f"\n[Rollout] Total cases: {len(cases_to_run)}")
for c in cases_to_run:
    print(f"  - {c['label']} ({c['split']}, {c['pop'].shape[0]} steps)")

#%%
# =============================================================================
# Rollout + Expand + Visualize + Save
# =============================================================================

for case_info in cases_to_run:
    case_num = case_info['case_num']
    split_label = case_info['split']
    case_pop = case_info['pop']
    U_orig = case_info['U_scaled']
    U_orig_raw = case_info['U_raw']
    nA_seg = case_info['nA']
    truth_frac_seg = case_info['truth_frac']
    truth_W = case_info['W']
    truth_pop = case_pop
    x0_frame = case_info['X_frame']
    tag = str(case_info['label'])
    
    print(f"\n{'='*80}")
    print(f" [{case_num}] ({split_label}) Rollout + Expand")
    print(f"{'='*80}")
    
    # ===== Basic information =====
    orig_len = case_pop.shape[0]
    
    dt_real = float(cfg.data.dt)
    dt_real_ns = dt_real * 1e9
    
    # Initial condition
    with torch.no_grad():
        x0 = x0_frame.to(device=device, dtype=dtype)
        z0 = ae.encoder(x0).squeeze().cpu().numpy()
    
    # Expand: hold the last control fixed
    n_extend = int(EXTEND_NS / dt_real_ns)
    U_last = U_orig[-1]
    U_last_raw = U_orig_raw[-1]
    U_extend = np.tile(U_last, (n_extend, 1))
    U_total = np.vstack([U_orig, U_extend])
    total_len = U_total.shape[0]
    t_total_ns = np.arange(total_len) * dt_real_ns
    
    T_hold = U_last_raw[0]
    rho_hold = U_last_raw[1]
    
    print(f"  Original: {orig_len} steps ({orig_len * dt_real_ns:.2f} ns)")
    print(f"  Extended: {n_extend} steps (+{EXTEND_NS:.2f} ns)")
    print(f"  Total:    {total_len} steps ({t_total_ns[-1]:.2f} ns)")
    print(f"  Hold T={T_hold:.4e}, rho={rho_hold:.4e}")
    
    # ===== SINDy Rollout =====
    Z_sim = rollout_sindy(z0, U_total, total_len)
    decoded = decode_Z(Z_sim)
    
    W_sim = decoded['W']
    frac_sim = decoded['frac']
    csd_sim = decoded['csd']
    zbar_sim = decoded['zbar']
    
    # Truth (already loaded in case_info)
    # Rollout population: frac * nA (use last nA in the expanded region)
    nA_last = nA_seg[-1]
    nA_total = np.zeros((total_len, 1))
    nA_total[:orig_len] = nA_seg
    nA_total[orig_len:] = nA_last
    pop_sim = frac_sim * nA_total
    
    if ap.ion_available:
        csd_truth = ap.compute_csd_numpy(truth_frac_seg)
        zbar_truth = ap.compute_zbar_numpy(truth_frac_seg)
    else:
        csd_truth = zbar_truth = None
    
    # ===== dZ/dt convergence =====
    if use_adaptive_sindy:
        # adaptive: compute a(U), A(U) at each time point
        dZ_norm = np.zeros(total_len)
        U_total_t = torch.tensor(U_total, dtype=dtype, device=device)
        with torch.no_grad():
            for ti in range(total_len):
                U_t = U_total_t[ti].unsqueeze(0)
                a_t, A_t = sindy_model.get_coefficients_batch(U_t)
                a_np = a_t.squeeze().cpu().numpy()
                A_np = A_t.squeeze().cpu().numpy()
                dz = a_np + Z_sim[ti] @ A_np.T
                dZ_norm[ti] = np.linalg.norm(dz)
    else:
        dZ_dt = a_global + Z_sim @ A_global.T + U_total @ B_global.T
        dZ_norm = np.linalg.norm(dZ_dt, axis=1)
    
    # ==================== VISUALIZATION ====================
    tag = f"seg{case_num}"
    
    # --- (1) Control Input ---
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=DPI, constrained_layout=True)
    U_total_raw = np.vstack([U_orig_raw, np.tile(U_last_raw, (n_extend, 1))])
    ax.plot(t_total_ns, U_total_raw[:, 0], 'r-', lw=1.5, label=f'T (eV)')
    ax2 = ax.twinx()
    ax2.plot(t_total_ns, U_total_raw[:, 1], 'b-', lw=1.5, label=f'ρ (g/cc)')
    ax.axvline(t_total_ns[orig_len-1], color='k', ls='--', alpha=0.5, label='Hold start')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Temperature (eV)", color='r')
    ax2.set_ylabel("Density (g/cc)", color='b')
    ax.set_title(f"Control Input | Case {case_num} ({split_label})")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax.grid(True, alpha=0.3)
    save_fig(f"rollout_control_{tag}")
    show_plot()
# --- (2) W-space Heatmap ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=DPI, constrained_layout=True)
    extent = [0, t_total_ns[-1], 0, cfg.data.nx]
    
    vmin_w = min(np.percentile(truth_W, 1), np.percentile(W_sim[:orig_len], 1))
    vmax_w = max(np.percentile(truth_W, 99), np.percentile(W_sim[:orig_len], 99))
    
    im0 = axes[0].imshow(W_sim.T, aspect='auto', origin='lower', extent=extent,
                          cmap='magma', vmin=vmin_w, vmax=vmax_w)
    axes[0].axvline(t_total_ns[orig_len-1], color='w', ls='--')
    axes[0].set_title("SINDy Rollout W-space")
    axes[0].set_ylabel("State Index")
    plt.colorbar(im0, ax=axes[0], label='W')
    
    truth_pad_W = np.full_like(W_sim, np.nan)
    truth_pad_W[:orig_len] = truth_W
    im1 = axes[1].imshow(truth_pad_W.T, aspect='auto', origin='lower', extent=extent,
                          cmap='magma', vmin=vmin_w, vmax=vmax_w)
    axes[1].axvline(t_total_ns[orig_len-1], color='w', ls='--')
    axes[1].set_title("Truth W-space")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("State Index")
    axes[1].set_facecolor('0.9')
    plt.colorbar(im1, ax=axes[1], label='W')
    
    fig.suptitle(f"W-space | Case {case_num} ({split_label})", fontsize=13, fontweight='bold')
    save_fig(f"rollout_W_{tag}")
    show_plot()
# --- (3) Population Heatmap ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=DPI, constrained_layout=True)
    
    vmax_pop = np.percentile(truth_pop, 99)
    im0 = axes[0].imshow(pop_sim.T, aspect='auto', origin='lower', extent=extent,
                          cmap='viridis', vmin=0, vmax=vmax_pop)
    axes[0].axvline(t_total_ns[orig_len-1], color='w', ls='--')
    axes[0].set_title("SINDy Rollout Population")
    axes[0].set_ylabel("State Index")
    plt.colorbar(im0, ax=axes[0], label='N')
    
    pop_pad = np.full_like(pop_sim, np.nan)
    pop_pad[:orig_len] = truth_pop
    im1 = axes[1].imshow(pop_pad.T, aspect='auto', origin='lower', extent=extent,
                          cmap='viridis', vmin=0, vmax=vmax_pop)
    axes[1].axvline(t_total_ns[orig_len-1], color='w', ls='--')
    axes[1].set_title("Truth Population")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("State Index")
    axes[1].set_facecolor('0.9')
    plt.colorbar(im1, ax=axes[1], label='N')
    
    fig.suptitle(f"Population | Case {case_num} ({split_label})", fontsize=13, fontweight='bold')
    save_fig(f"rollout_pop_{tag}")
    show_plot()
# --- (3b) Fraction Heatmap ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=DPI, constrained_layout=True)
    
    vmin_f = min(np.percentile(truth_frac_seg, 1), np.percentile(frac_sim[:orig_len], 1))
    vmax_f = max(np.percentile(truth_frac_seg, 99), np.percentile(frac_sim[:orig_len], 99))
    
    im0 = axes[0].imshow(frac_sim.T, aspect='auto', origin='lower', extent=extent,
                          cmap='magma', vmin=vmin_f, vmax=vmax_f)
    axes[0].axvline(t_total_ns[orig_len-1], color='w', ls='--')
    axes[0].set_title("SINDy Rollout Fraction")
    axes[0].set_ylabel("State Index")
    plt.colorbar(im0, ax=axes[0], label='Frac')
    
    frac_pad = np.full_like(frac_sim, np.nan)
    frac_pad[:orig_len] = truth_frac_seg
    im1 = axes[1].imshow(frac_pad.T, aspect='auto', origin='lower', extent=extent,
                          cmap='magma', vmin=vmin_f, vmax=vmax_f)
    axes[1].axvline(t_total_ns[orig_len-1], color='w', ls='--')
    axes[1].set_title("Truth Fraction")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("State Index")
    axes[1].set_facecolor('0.9')
    plt.colorbar(im1, ax=axes[1], label='Frac')
    
    fig.suptitle(f"Fraction | Case {case_num} ({split_label})", fontsize=13, fontweight='bold')
    save_fig(f"rollout_frac_{tag}")
    show_plot()
# --- (4) CSD & Zbar ---
    if csd_sim is not None:
        n_charges = csd_sim.shape[1]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=DPI, constrained_layout=True)
        extent_c = [0, t_total_ns[-1], 0, n_charges]
        
        vmax_c = np.percentile(csd_truth, 99) if csd_truth is not None else np.percentile(csd_sim[:orig_len], 99)
        
        # CSD Rollout
        im0 = axes[0].imshow(csd_sim.T, aspect='auto', origin='lower', extent=extent_c,
                              cmap='inferno', vmin=0, vmax=vmax_c)
        axes[0].axvline(t_total_ns[orig_len-1], color='w', ls='--')
        axes[0].set_title("SINDy Rollout CSD")
        axes[0].set_ylabel("Charge State")
        plt.colorbar(im0, ax=axes[0])
        
        # CSD Truth
        csd_pad = np.full_like(csd_sim, np.nan)
        if csd_truth is not None:
            csd_pad[:orig_len] = csd_truth
        im1 = axes[1].imshow(csd_pad.T, aspect='auto', origin='lower', extent=extent_c,
                              cmap='inferno', vmin=0, vmax=vmax_c)
        axes[1].axvline(t_total_ns[orig_len-1], color='w', ls='--')
        axes[1].set_title("Truth CSD")
        axes[1].set_ylabel("Charge State")
        axes[1].set_facecolor('0.9')
        plt.colorbar(im1, ax=axes[1])
        
        # Zbar
        if zbar_truth is not None:
            axes[2].plot(t_total_ns[:orig_len], zbar_truth, 'k--', lw=2.0, label='Truth')
        axes[2].plot(t_total_ns, zbar_sim, 'r-', lw=1.5, label='SINDy Rollout')
        axes[2].axvline(t_total_ns[orig_len-1], color='k', ls='--', alpha=0.5, label='Hold start')
        axes[2].set_xlabel("Time (ns)")
        axes[2].set_ylabel("Zbar")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        fig.suptitle(f"CSD & Zbar | Case {case_num} ({split_label})", fontsize=13, fontweight='bold')
        save_fig(f"rollout_csd_zbar_{tag}")
        show_plot()
# --- (5) Convergence (dZ/dt norm) ---
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=DPI, constrained_layout=True)
    ax.plot(t_total_ns, dZ_norm, 'b-', lw=1.5)
    ax.axvline(t_total_ns[orig_len-1], color='k', ls='--', alpha=0.5, label='Hold start')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("||dZ/dt||")
    ax.set_title(f"Convergence to Equilibrium | Case {case_num} ({split_label})")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(f"rollout_convergence_{tag}")
    show_plot()
# --- (6) Latent space ---
    nz = z0.shape[0]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
    cmap_z = plt.cm.get_cmap('tab10')
    for i in range(nz):
        ax.plot(t_total_ns, Z_sim[:, i], color=cmap_z(i), lw=1.2, label=f'z{i+1}')
    ax.axvline(t_total_ns[orig_len-1], color='k', ls='--', alpha=0.5, label='Hold start')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Z")
    ax.set_title(f"Latent Dynamics | Case {case_num} ({split_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize='small')
    save_fig(f"rollout_latent_{tag}")
    show_plot()
# ==================== STEADY-STATE COMPARISON ====================
    if ss_ref_available and ap.ion_available:
        print(f"\n  [SS Comparison] Hold conditions: T={T_hold:.4e} eV, rho={rho_hold:.4e} g/cc")
        
        ss_ref = find_ss_reference(T_hold, rho_hold)
        
        if ss_ref is not None:
            method = ss_ref['method']
            nearest_T = ss_ref['nearest_T']
            nearest_rho = ss_ref['nearest_rho']
            dist = ss_ref['distance']
            
            if method == 'exact':
                print(f"  ✅ Found exact matching SS data")
                print(f"     SS (T={nearest_T:.4e}, rho={nearest_rho:.4e})")
            else:
                print(f"  ⚠️  No exact matching (T, rho) data; used {method}.")
                print(f"     Requested:   T={T_hold:.4e}, rho={rho_hold:.4e}")
                print(f"     Nearest SS: T={nearest_T:.4e}, rho={nearest_rho:.4e} (rel. dist={dist:.4f})")
            
            # Values at the final expanded time point
            rollout_csd_final = csd_sim[-1]
            rollout_zbar_final = zbar_sim[-1]
            
            ss_csd_ref = ss_ref.get('csd', None)
            ss_zbar_ref = ss_ref.get('zbar', None)
            
            # --- CSD bar plot (final expanded value vs SS reference) ---
            if ss_csd_ref is not None:
                fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI, constrained_layout=True)
                
                q = np.arange(len(rollout_csd_final))
                width = 0.35
                
                ax.bar(q - width/2, rollout_csd_final, width, color='tab:red', alpha=0.8,
                       label=f'Rollout final (t={t_total_ns[-1]:.1f} ns)')
                ax.bar(q + width/2, ss_csd_ref, width, color='tab:blue', alpha=0.8,
                       label=f'SS ref ({method})')
                
                ax.set_xlabel("Charge State q")
                ax.set_ylabel("CSD (fraction)")
                
                title_str = f"CSD at Steady State | Case {case_num}\n"
                title_str += f"T={T_hold:.3e} eV, ρ={rho_hold:.3e} g/cc"
                if method != 'exact':
                    title_str += f"\n(SS ref: T={nearest_T:.3e}, ρ={nearest_rho:.3e}, {method})"
                ax.set_title(title_str, fontsize=11)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Display Zbar
                if ss_zbar_ref is not None:
                    ax.axvline(rollout_zbar_final, color='tab:red', ls='--', lw=1.5, alpha=0.7,
                               label=f'Rollout Zbar={rollout_zbar_final:.2f}')
                    ax.axvline(ss_zbar_ref, color='tab:blue', ls='--', lw=1.5, alpha=0.7,
                               label=f'SS ref Zbar={ss_zbar_ref:.2f}')
                    ax.legend()
                
                save_fig(f"rollout_ss_csd_{tag}")
                show_plot()
# MRE calculation
                mask = ss_csd_ref > 1e-6
                if np.any(mask):
                    mre = np.mean(np.abs(rollout_csd_final[mask] - ss_csd_ref[mask]) / ss_csd_ref[mask])
                    print(f"  CSD MRE (activated): {mre*100:.2f}%")
                if ss_zbar_ref is not None:
                    zbar_err = abs(rollout_zbar_final - ss_zbar_ref)
                    print(f"  Zbar: rollout={rollout_zbar_final:.4f}, SS ref={ss_zbar_ref:.4f}, |err|={zbar_err:.4f}")
            
            # --- Fraction bar plot (final expanded value vs SS reference) ---
            ss_frac_ref = ss_ref.get('frac', None)
            rollout_frac_final = frac_sim[-1]
            
            if ss_frac_ref is not None:
                # Show only the top N states because the full nx is too large
                n_top = 30
                combined_max = np.maximum(rollout_frac_final, ss_frac_ref)
                top_idx = np.argsort(combined_max)[::-1][:n_top]
                top_idx_sorted = np.sort(top_idx)
                
                fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI, constrained_layout=True)
                
                x_pos = np.arange(len(top_idx_sorted))
                width = 0.35
                
                ax.bar(x_pos - width/2, rollout_frac_final[top_idx_sorted], width, 
                       color='tab:red', alpha=0.8, label=f'Rollout final')
                ax.bar(x_pos + width/2, ss_frac_ref[top_idx_sorted], width,
                       color='tab:blue', alpha=0.8, label=f'SS ref ({method})')
                
                ax.set_xlabel("State Index")
                ax.set_ylabel("Fraction")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(i) for i in top_idx_sorted], fontsize=7, rotation=45)
                
                title_str = f"Fraction at Steady State (top {n_top}) | Case {case_num}\n"
                title_str += f"T={T_hold:.3e} eV, ρ={rho_hold:.3e} g/cc"
                if method != 'exact':
                    title_str += f"\n(SS ref: T={nearest_T:.3e}, ρ={nearest_rho:.3e}, {method})"
                ax.set_title(title_str, fontsize=11)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                save_fig(f"rollout_ss_frac_{tag}")
                show_plot()
# Fraction MRE
                frac_mask = ss_frac_ref > 1e-6
                if np.any(frac_mask):
                    frac_mre = np.mean(np.abs(rollout_frac_final[frac_mask] - ss_frac_ref[frac_mask]) / ss_frac_ref[frac_mask])
                    print(f"  Fraction MRE (activated): {frac_mre*100:.2f}%")
            
            # --- Population bar plot (final expanded value vs SS reference) ---
            if ss_frac_ref is not None:
                # Population = fraction * nA (use nA from the last time point)
                rollout_pop_final = rollout_frac_final * float(nA_last)
                ss_pop_ref = ss_frac_ref * float(nA_last)  # Compare with the same nA
                
                combined_max_pop = np.maximum(rollout_pop_final, ss_pop_ref)
                top_idx_pop = np.argsort(combined_max_pop)[::-1][:n_top]
                top_idx_pop_sorted = np.sort(top_idx_pop)
                
                fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI, constrained_layout=True)
                
                x_pos = np.arange(len(top_idx_pop_sorted))
                
                ax.bar(x_pos - width/2, rollout_pop_final[top_idx_pop_sorted], width,
                       color='tab:red', alpha=0.8, label=f'Rollout final')
                ax.bar(x_pos + width/2, ss_pop_ref[top_idx_pop_sorted], width,
                       color='tab:blue', alpha=0.8, label=f'SS ref ({method})')
                
                ax.set_xlabel("State Index")
                ax.set_ylabel("Population (N)")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(i) for i in top_idx_pop_sorted], fontsize=7, rotation=45)
                
                title_str = f"Population at Steady State (top {n_top}) | Case {case_num}\n"
                title_str += f"T={T_hold:.3e} eV, ρ={rho_hold:.3e} g/cc, nA={float(nA_last):.3e}"
                if method != 'exact':
                    title_str += f"\n(SS ref: T={nearest_T:.3e}, ρ={nearest_rho:.3e}, {method})"
                ax.set_title(title_str, fontsize=11)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                save_fig(f"rollout_ss_pop_{tag}")
                show_plot()
# ==================== SAVE DATA ====================
    if SAVE_DATA and data_dir is not None:
        case_dir = data_dir / f"case_{case_num}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Common ---
        save_txt(case_dir / "time_ns.txt", t_total_ns.reshape(-1, 1), 
                 header=f"time(ns) | case={case_num}, orig={orig_len}, extend={n_extend}")
        save_txt(case_dir / "control_raw.txt", U_total_raw,
                 header=f"T(eV), density(g/cc) | total_len={total_len}")
        save_txt(case_dir / "control_scaled.txt", U_total,
                 header=f"T(scaled), density(scaled) | total_len={total_len}")
        
        # --- Full rollout (original + expanded) ---
        save_txt(case_dir / "Z_rollout.txt", Z_sim,
                 header=f"Z rollout | shape={Z_sim.shape}")
        save_txt(case_dir / "W_rollout.txt", W_sim,
                 header=f"W rollout | shape={W_sim.shape}")
        save_txt(case_dir / "fraction_rollout.txt", frac_sim,
                 header=f"fraction rollout | shape={frac_sim.shape}")
        save_txt(case_dir / "population_rollout.txt", pop_sim,
                 header=f"population rollout | shape={pop_sim.shape}")
        if csd_sim is not None:
            save_txt(case_dir / "CSD_rollout.txt", csd_sim,
                     header=f"CSD rollout | shape={csd_sim.shape}")
        if zbar_sim is not None:
            save_txt(case_dir / "Zbar_rollout.txt", zbar_sim.reshape(-1, 1),
                     header=f"Zbar rollout | len={len(zbar_sim)}")
        save_txt(case_dir / "dZdt_norm.txt", dZ_norm.reshape(-1, 1),
                 header=f"||dZ/dt|| convergence | len={len(dZ_norm)}")
        
        # --- Truth (original window only) ---
        save_txt(case_dir / "W_truth.txt", truth_W,
                 header=f"W truth | orig_len={orig_len}")
        save_txt(case_dir / "fraction_truth.txt", truth_frac_seg,
                 header=f"fraction truth | orig_len={orig_len}")
        save_txt(case_dir / "population_truth.txt", truth_pop,
                 header=f"population truth | orig_len={orig_len}")
        if csd_truth is not None:
            save_txt(case_dir / "CSD_truth.txt", csd_truth,
                     header=f"CSD truth | orig_len={orig_len}")
        if zbar_truth is not None:
            save_txt(case_dir / "Zbar_truth.txt", zbar_truth.reshape(-1, 1),
                     header=f"Zbar truth | orig_len={orig_len}")
        
        # --- SS comparison data ---
        if ss_ref_available and ap.ion_available:
            ss_ref_save = find_ss_reference(T_hold, rho_hold)
            if ss_ref_save is not None:
                ss_dir = case_dir / "ss_comparison"
                ss_dir.mkdir(parents=True, exist_ok=True)
                
                # Metadata
                with open(ss_dir / "ss_info.txt", "w", encoding="utf-8") as f:
                    f.write(f"# Steady-state comparison info\n")
                    f.write(f"method: {ss_ref_save['method']}\n")
                    f.write(f"target_T: {T_hold:.12e}\n")
                    f.write(f"target_rho: {rho_hold:.12e}\n")
                    f.write(f"nearest_T: {ss_ref_save['nearest_T']:.12e}\n")
                    f.write(f"nearest_rho: {ss_ref_save['nearest_rho']:.12e}\n")
                    f.write(f"rel_distance: {ss_ref_save['distance']:.12e}\n")
                    f.write(f"nA_last: {float(nA_last):.12e}\n")
                
                # Final rollout time point
                save_txt(ss_dir / "rollout_final_fraction.txt", frac_sim[-1:],
                         header=f"rollout final fraction | t={t_total_ns[-1]:.2f} ns")
                save_txt(ss_dir / "rollout_final_population.txt", (frac_sim[-1:] * float(nA_last)),
                         header=f"rollout final population (frac*nA) | nA={float(nA_last):.4e}")
                if csd_sim is not None:
                    save_txt(ss_dir / "rollout_final_CSD.txt", csd_sim[-1:],
                             header=f"rollout final CSD")
                if zbar_sim is not None:
                    save_txt(ss_dir / "rollout_final_Zbar.txt", np.array([[zbar_sim[-1]]]),
                             header=f"rollout final Zbar")
                
                # SS reference
                ss_frac_r = ss_ref_save.get('frac', None)
                ss_csd_r = ss_ref_save.get('csd', None)
                ss_zbar_r = ss_ref_save.get('zbar', None)
                
                if ss_frac_r is not None:
                    save_txt(ss_dir / "ss_ref_fraction.txt", ss_frac_r.reshape(1, -1),
                             header=f"SS ref fraction | method={ss_ref_save['method']}")
                    save_txt(ss_dir / "ss_ref_population.txt", (ss_frac_r * float(nA_last)).reshape(1, -1),
                             header=f"SS ref population (frac*nA) | nA={float(nA_last):.4e}")
                if ss_csd_r is not None:
                    save_txt(ss_dir / "ss_ref_CSD.txt", ss_csd_r.reshape(1, -1),
                             header=f"SS ref CSD | method={ss_ref_save['method']}")
                if ss_zbar_r is not None:
                    save_txt(ss_dir / "ss_ref_Zbar.txt", np.array([[ss_zbar_r]]),
                             header=f"SS ref Zbar | method={ss_ref_save['method']}")
                
                print(f"  ✅ SS comparison data saved to {ss_dir}")
        
        print(f"  ✅ Data saved to {case_dir}")
    
    print(f"\n  ✅ Case {case_num} complete")

#%%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print(" Rollout & Expand Complete!")
print("="*80)
print(f"\nModel: {_ckpt_path} ({_ckpt_label}, epoch {best_epoch})")
print(f"SINDy: {'Adaptive' if use_adaptive_sindy else 'lstsq'}")
print(f"\nCases: {[c['label'] for c in cases_to_run]}")
print(f"Extend: +{EXTEND_NS:.1f} ns")
if SAVE_PLOTS:
    print(f"Plots: {plot_dir}")
if SAVE_DATA:
    print(f"Data:  {data_dir}")
print("="*80)
# === Final wait after all figures are shown (terminal execution) ===
if SHOW_PLOTS:
    print("\n[Done] All plots are displayed. Close any figure window to exit.")
    plt.show(block=True)

