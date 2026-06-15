# -*- coding: utf-8 -*-
"""
Truth Data Viewer
=================
Standalone script for visualizing raw data without a model.

Inspect Population, Fraction, CSD, and Zbar with heatmaps and line plots.

Usage:
    1. Edit only the CONFIG section below.
    2. Run python view_truth.py or execute cells in VSCode.
"""

#%%
# =============================================================================
# CONFIG - edit only this section
# =============================================================================

# Data paths
DATA_DIR    = "../../data_storage/lithography_4ns"
NAMES_FILE  = "./name_total.txt"      # State-name file (nx lines)
NX          = 1583                     # Number of states
DT          = 1e-13                    # Time interval (s)
POP_LIM     = 1e-30                    # Lower population bound

# File numbers to view (segment number list)
CASE_NUMBERS = [42, 43, 44, 63,59,60]

# Plot options
DPI = 150
SHOW_PLOTS = True
SAVE_PLOTS = False
PLOT_DIR   = "./truth_plots"

# Scale options
FRACTION_SCALE = 'log'    # 'linear' or 'log'
CSD_SCALE      = 'linear' # 'linear' or 'log'
POP_SCALE      = 'linear' # 'linear' or 'log'

#%%
# =============================================================================
# Imports & Setup
# =============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def show_plot():
    """Show non-blocking plots when SHOW_PLOTS is True; otherwise close them."""
    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.05)
    else:
        plt.close()
from matplotlib.colors import LogNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.atomic_physics import AtomicPhysics
from src.data_utils import load_pop_matrix_auto, load_history_file, load_state_names
import torch

if SAVE_PLOTS:
    Path(PLOT_DIR).mkdir(exist_ok=True)

def save_fig(name):
    if SAVE_PLOTS:
        path = Path(PLOT_DIR) / f"{name}.png"
        plt.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f"  💾 Saved: {path}")

print("✅ Imports complete")

#%%
# =============================================================================
# Load Data
# =============================================================================

print("[Data] Loading...")

# Load population files
pops = []
for case_num in CASE_NUMBERS:
    fp = Path(DATA_DIR) / f"density_population_seg{case_num}.txt"
    if not fp.exists():
        raise FileNotFoundError(f"Not found: {fp}")
    arr = load_pop_matrix_auto(fp, NX)
    arr = np.asarray(arr, dtype=np.float64)
    pops.append(arr)
    print(f"  seg{case_num}: {arr.shape}")

# Load history files
histories = []
for case_num in CASE_NUMBERS:
    hp = Path(DATA_DIR) / f"historyfile_seg{case_num}.txt"
    if hp.exists():
        t_h, U_h = load_history_file(str(hp))
        histories.append((t_h, U_h))
    else:
        histories.append(None)
        print(f"  ⚠️ historyfile_seg{case_num}.txt not found")

# State names & AtomicPhysics
state_names = load_state_names(NAMES_FILE, NX)
ap = AtomicPhysics(state_names, NX, torch.float64)

if ap.ion_available:
    print(f"  Z0={ap.Z0}, nq={ap.nq}")
else:
    print("  ⚠️ CSD/Zbar not available (failed to build ion index)")

print(f"✅ {len(CASE_NUMBERS)} segments loaded")

#%%
# =============================================================================
# Compute Derived Quantities
# =============================================================================

print("[Compute] Fraction, CSD, Zbar...")

all_data = []
for i, (case_num, pop_raw) in enumerate(zip(CASE_NUMBERS, pops)):
    # Apply pop_lim
    nA = np.sum(pop_raw, axis=1, keepdims=True)
    pop = pop_raw / nA + POP_LIM
    pop = pop * nA
    nA = np.sum(pop, axis=1, keepdims=True)  # Recompute

    # Fraction
    frac = pop / nA

    # Time axis
    nt = pop.shape[0]
    t = np.arange(nt) * DT

    # CSD & Zbar
    csd = ap.compute_csd_numpy(frac) if ap.ion_available else None
    zbar = ap.compute_zbar_numpy(frac) if ap.ion_available else None

    # History
    T_hist, n_hist = None, None
    if histories[i] is not None:
        _, U_h = histories[i]
        L = min(nt, len(U_h))
        T_hist = U_h[:L, 0]
        n_hist = U_h[:L, 1]

    all_data.append({
        'case': case_num,
        'pop': pop, 'frac': frac, 'nA': nA,
        'csd': csd, 'zbar': zbar,
        't': t, 'nt': nt,
        'T_hist': T_hist, 'n_hist': n_hist,
    })

print("✅ Derived quantities computed")

#%%
# =============================================================================
# Color setup
# =============================================================================

cmap_jet = plt.get_cmap("jet")
state_colors = [cmap_jet(i / max(1, NX - 1)) for i in range(NX)]

if ap.ion_available:
    nq = ap.nq
    ion_colors = [cmap_jet(q / max(1, nq - 1)) for q in range(nq)]

#%%
# =============================================================================
# [1] Population Heatmap
# =============================================================================

print("\n[1] Population Heatmaps")

for d in all_data:
    case_num, pop, t = d['case'], d['pop'], d['t']

    if POP_SCALE == 'log':
        vmin_p = max(1e-10, np.percentile(pop[pop > 0], 1))
        vmax_p = np.percentile(pop, 99)
        norm_p = LogNorm(vmin=vmin_p, vmax=vmax_p)
        label_p = "Population (log)"
    else:
        vmin_p, vmax_p = 0, np.percentile(pop, 99)
        norm_p = None
        label_p = "Population"

    fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
    im = ax.imshow(pop.T, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], 0, NX-1],
                   cmap='magma', norm=norm_p)
    ax.set_title(f"Population | seg{case_num}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    fig.colorbar(im, ax=ax, label=label_p, shrink=0.8)
    save_fig(f"pop_heatmap_seg{case_num}")
    show_plot()
print("✅ Population heatmaps done")

#%%
# =============================================================================
# [2] Population Line Plot
# =============================================================================

print("\n[2] Population Lines")

for d in all_data:
    case_num, pop, t = d['case'], d['pop'], d['t']

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI, constrained_layout=True)
    for i in range(NX):
        ax.plot(t, pop[:, i], color=state_colors[i], lw=0.5, alpha=0.8)
    ax.set_title(f"Population | seg{case_num}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population")
    ax.grid(True, alpha=0.3)
    save_fig(f"pop_line_seg{case_num}")
    show_plot()
print("✅ Population lines done")

#%%
# =============================================================================
# [3] Fraction Heatmap
# =============================================================================

print("\n[3] Fraction Heatmaps")

for d in all_data:
    case_num, frac, t = d['case'], d['frac'], d['t']

    if FRACTION_SCALE == 'log':
        vmin_f, vmax_f = 1e-10, 1.0
        norm_f = LogNorm(vmin=vmin_f, vmax=vmax_f)
        label_f = "Fraction (log)"
    else:
        vmin_f, vmax_f = 0.0, 1.0
        norm_f = None
        label_f = "Fraction"

    fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
    im = ax.imshow(frac.T, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], 0, NX-1],
                   cmap='magma', norm=norm_f)
    ax.set_title(f"Fraction | seg{case_num}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    fig.colorbar(im, ax=ax, label=label_f, shrink=0.8)
    save_fig(f"frac_heatmap_seg{case_num}")
    show_plot()
print("✅ Fraction heatmaps done")

#%%
# =============================================================================
# [4] Fraction Line Plot
# =============================================================================

print("\n[4] Fraction Lines")

for d in all_data:
    case_num, frac, t = d['case'], d['frac'], d['t']

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI, constrained_layout=True)
    for i in range(NX):
        ax.plot(t, frac[:, i], color=state_colors[i], lw=0.5, alpha=0.8)
    ax.set_title(f"Fraction | seg{case_num}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    save_fig(f"frac_line_seg{case_num}")
    show_plot()
print("✅ Fraction lines done")

#%%
# =============================================================================
# [5] CSD Heatmap
# =============================================================================

if ap.ion_available:
    print("\n[5] CSD Heatmaps")

    for d in all_data:
        case_num, csd, t = d['case'], d['csd'], d['t']
        if csd is None:
            continue

        if CSD_SCALE == 'log':
            vmin_c, vmax_c = 1e-10, 1.0
            norm_c = LogNorm(vmin=vmin_c, vmax=vmax_c)
            label_c = "CSD (log)"
        else:
            vmin_c, vmax_c = 0.0, 1.0
            norm_c = None
            label_c = "CSD"

        fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
        im = ax.imshow(csd.T, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], 0, nq-1],
                       cmap='viridis', norm=norm_c)
        ax.set_title(f"CSD | seg{case_num} (Z0={ap.Z0})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Charge State (q)")
        fig.colorbar(im, ax=ax, label=label_c, shrink=0.8)
        save_fig(f"csd_heatmap_seg{case_num}")
        show_plot()
print("✅ CSD heatmaps done")

#%%
# =============================================================================
# [6] CSD Line Plot
# =============================================================================

if ap.ion_available:
    print("\n[6] CSD Lines")

    for d in all_data:
        case_num, csd, t = d['case'], d['csd'], d['t']
        if csd is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI, constrained_layout=True)
        for q in range(nq):
            lbl = f"q={q}" if q <= 10 else None
            ax.plot(t, csd[:, q], color=ion_colors[q], lw=0.8, alpha=0.8, label=lbl)
        ax.set_title(f"CSD | seg{case_num} (Z0={ap.Z0})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CSD (fraction)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=3, loc='upper right')
        save_fig(f"csd_line_seg{case_num}")
        show_plot()
print("✅ CSD lines done")

#%%
# =============================================================================
# [7] Zbar
# =============================================================================

if ap.ion_available:
    print("\n[7] Zbar Curves")

    for d in all_data:
        case_num, zbar, t = d['case'], d['zbar'], d['t']
        if zbar is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
        ax.plot(t, zbar, 'k-', lw=1.5)
        ax.set_title(f"Mean Charge (Zbar) | seg{case_num}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\bar{Z}$")
        ax.grid(True, alpha=0.3)
        save_fig(f"zbar_seg{case_num}")
        show_plot()
print("✅ Zbar curves done")

#%%
# =============================================================================
# [8] Control Variables (T, n)
# =============================================================================

print("\n[8] Control Variables")

for d in all_data:
    case_num, t = d['case'], d['t']
    T_hist, n_hist = d['T_hist'], d['n_hist']

    if T_hist is None:
        print(f"  seg{case_num}: no history file, skip")
        continue

    L = min(len(t), len(T_hist))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    dpi=DPI, constrained_layout=True)
    ax1.plot(t[:L], T_hist[:L], 'r-', lw=1.2)
    ax1.set_ylabel("Temperature")
    ax1.set_title(f"Control Variables | seg{case_num}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t[:L], n_hist[:L], 'b-', lw=1.2)
    ax2.set_ylabel("Density")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    save_fig(f"control_seg{case_num}")
    show_plot()
print("✅ Control variables done")

#%%
# =============================================================================
# [9] dZbar/dt
# =============================================================================

if ap.ion_available:
    print("\n[9] dZbar/dt")

    for d in all_data:
        case_num, zbar, t = d['case'], d['zbar'], d['t']
        if zbar is None:
            continue

        dzbar_dt = np.gradient(zbar, DT)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                        dpi=DPI, constrained_layout=True)
        ax1.plot(t, zbar, 'k-', lw=1.2)
        ax1.set_ylabel(r"$\bar{Z}$")
        ax1.set_title(f"Zbar & dZbar/dt | seg{case_num}")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, dzbar_dt, 'C0-', lw=1.0)
        ax2.axhline(0, color='k', ls='--', lw=0.5)
        ax2.set_ylabel(r"$d\bar{Z}/dt$")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, alpha=0.3)

        save_fig(f"dzbar_dt_seg{case_num}")
        show_plot()
print("✅ dZbar/dt done")

#%%
# =============================================================================
# [10] dCSD/dt Heatmap & Line
# =============================================================================

if ap.ion_available:
    print("\n[10] dCSD/dt")

    for d in all_data:
        case_num, csd, t = d['case'], d['csd'], d['t']
        if csd is None:
            continue

        dcsd_dt = np.gradient(csd, DT, axis=0)

        # Heatmap
        vmax_dc = np.percentile(np.abs(dcsd_dt), 99)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, constrained_layout=True)
        im = ax.imshow(dcsd_dt.T, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], 0, nq-1],
                       cmap='RdBu_r', vmin=-vmax_dc, vmax=vmax_dc)
        ax.set_title(f"dCSD/dt | seg{case_num}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Charge State (q)")
        fig.colorbar(im, ax=ax, label="dCSD/dt", shrink=0.8)
        save_fig(f"dcsd_dt_heatmap_seg{case_num}")
        show_plot()
# Line
        fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI, constrained_layout=True)
        for q in range(nq):
            lbl = f"q={q}" if q <= 10 else None
            ax.plot(t, dcsd_dt[:, q], color=ion_colors[q], lw=0.8, alpha=0.8, label=lbl)
        ax.axhline(0, color='k', ls='--', lw=0.5)
        ax.set_title(f"dCSD/dt | seg{case_num}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dCSD/dt")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=3, loc='upper right')
        save_fig(f"dcsd_dt_line_seg{case_num}")
        show_plot()
print("✅ dCSD/dt done")

#%%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print(" Truth Data Viewer Complete!")
print("="*60)
for d in all_data:
    print(f"  seg{d['case']}: nt={d['nt']}, nx={NX}, "
          f"pop=[{d['pop'].min():.2e}, {d['pop'].max():.2e}], "
          f"zbar=[{d['zbar'].min():.2f}, {d['zbar'].max():.2f}]" if d['zbar'] is not None else "")
if SAVE_PLOTS:
    print(f"\nPlots saved to: {PLOT_DIR}")
print("="*60)
if SHOW_PLOTS:
    print("\n[Done] All plots are displayed. Close any figure window to exit.")
    plt.show(block=True)
