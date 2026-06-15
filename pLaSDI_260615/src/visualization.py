# -*- coding: utf-8 -*-
"""
Visualization Module
====================
Visualize training results (learning curves, heatmaps, error analysis, etc.).

Load and visualize trained models.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Utilities
# =============================================================================

def safe_percentile(arr_list: List[np.ndarray], p_low: int = 1, p_high: int = 99,
                    fallback: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
    """Compute percentiles safely."""
    try:
        flat = np.concatenate([np.ravel(a) for a in arr_list if a is not None])
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return fallback
        lo, hi = np.nanpercentile(flat, [p_low, p_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return fallback
        return float(lo), float(hi)
    except Exception:
        return fallback


def safe_imshow(ax, A, extent=None, cmap="magma", vmin=None, vmax=None,
                title=None, xlabel=None, ylabel=None, interpolation="nearest"):
    """Safe imshow wrapper."""
    if A is None or A.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return None
    A = np.asarray(A)
    if not np.isfinite(A).any():
        ax.text(0.5, 0.5, "All values non-finite", ha="center", va="center", transform=ax.transAxes)
        return None
    im = ax.imshow(A, aspect="auto", origin="lower", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return im


def add_segment_guides(ax, edges, mids, labels, line_color="red", 
                       line_ls="--", line_lw=1.5, label_fs=10):
    """Add segment guide lines."""
    if edges is None or len(edges) < 2:
        return
    nseg = len(edges) - 1
    for t in edges[1:-1]:
        ax.axvline(t, color=line_color, ls=line_ls, lw=line_lw, zorder=5)
    ylim = ax.get_ylim()
    y_top = ylim[0] + 0.98 * (ylim[1] - ylim[0])
    for m, lb in zip(mids, labels):
        ax.text(m, y_top, lb, ha="center", va="top", fontsize=label_fs, color="black",
                bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"), zorder=6)


# =============================================================================
# Learning curves
# =============================================================================

def plot_training_curve(loss_path: str, val_loss_path: str = None,
                        title: str = "Training Curve", save_path: str = None):
    """Plot learning curves."""
    if not os.path.exists(loss_path):
        print(f"[plot] Loss file not found: {loss_path}")
        return
    
    curve = np.load(loss_path)
    
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(np.arange(1, len(curve)+1), curve, label="Train")
    
    if val_loss_path and os.path.exists(val_loss_path):
        vcurve = np.load(val_loss_path)
        if len(vcurve) > 0:
            plt.plot(np.arange(1, len(vcurve)+1), vcurve, label="Validation")
    
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Saved: {save_path}")
    plt.show(block=False)
    plt.pause(0.05)


def plot_metrics_breakdown(metrics_path: str, save_path: str = None):
    """Plot metric breakdown."""
    if not os.path.exists(metrics_path):
        print(f"[plot] Metrics file not found: {metrics_path}")
        return
    
    H = np.load(metrics_path, allow_pickle=True)
    
    ep = H["epoch"] if "epoch" in H.files else None
    if ep is None:
        print("[plot] No epoch data")
        return
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    def _plot(arr, name, style="-"):
        if arr is not None and name in H.files:
            data = H[name]
            n = min(len(ep), len(data))
            plt.plot(ep[:n], data[:n], style, label=name, lw=1.5)
    
    _plot(ep, "hurwitz")
    _plot(ep, "recW")
    _plot(ep, "frac")
    _plot(ep, "ion")
    _plot(ep, "zbar")
    _plot(ep, "FSE")
    _plot(ep, "steady_raw")
    
    if "total" in H.files:
        plt.plot(ep, H["total"], lw=2, label="total")
    if "val_total" in H.files:
        val = H["val_total"]
        n = min(len(ep), len(val))
        plt.plot(ep[:n], val[:n], "--", lw=2, label="val_total")
    
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("Training Metrics Breakdown")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(0.05)


# =============================================================================
# Prediction comparison
# =============================================================================

def plot_fraction_comparison(truth: np.ndarray, pred_ae: np.ndarray, 
                              pred_sindy: np.ndarray = None,
                              time_axis: np.ndarray = None,
                              nx: int = None, title: str = "",
                              save_path: str = None):
    """Plot fraction comparison as lines."""
    if nx is None:
        nx = truth.shape[1]
    
    if time_axis is None:
        time_axis = np.arange(truth.shape[0])
    
    cmap = plt.get_cmap("jet")
    colors = [cmap(i / max(1, nx-1)) for i in range(nx)]
    
    n_plots = 2 if pred_sindy is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5), sharey=True, dpi=150)
    
    # Truth
    for i in range(nx):
        axes[0].plot(time_axis, truth[:, i], color=colors[i], lw=0.8)
    axes[0].set_title(f"Truth {title}")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(0, 1)
    
    # AE
    for i in range(nx):
        axes[1].plot(time_axis, pred_ae[:, i], color=colors[i], lw=0.8)
    axes[1].set_title(f"AE Reconstruction {title}")
    axes[1].set_xlabel("Time")
    
    # SINDy
    if pred_sindy is not None:
        for i in range(nx):
            axes[2].plot(time_axis, pred_sindy[:, i], color=colors[i], lw=0.8)
        axes[2].set_title(f"SINDy Prediction {title}")
        axes[2].set_xlabel("Time")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(0.05)


def plot_heatmap_comparison(truth: np.ndarray, pred_ae: np.ndarray,
                            pred_sindy: np.ndarray = None,
                            time_axis: np.ndarray = None,
                            title: str = "", save_path: str = None):
    """Plot heatmap comparison."""
    if time_axis is None:
        time_axis = np.arange(truth.shape[0])
    
    n_plots = 2 if pred_sindy is None else 3
    
    vmin, vmax = safe_percentile([truth, pred_ae, pred_sindy], 1, 99)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4), 
                              sharex=True, sharey=True, dpi=150)
    
    extent = [time_axis[0], time_axis[-1], 0, truth.shape[1]-1]
    
    im0 = safe_imshow(axes[0], truth.T, extent=extent, cmap="magma",
                      vmin=vmin, vmax=vmax, title=f"Truth {title}",
                      xlabel="Time", ylabel="State")
    
    im1 = safe_imshow(axes[1], pred_ae.T, extent=extent, cmap="magma",
                      vmin=vmin, vmax=vmax, title=f"AE {title}",
                      xlabel="Time")
    
    if pred_sindy is not None:
        im2 = safe_imshow(axes[2], pred_sindy.T, extent=extent, cmap="magma",
                          vmin=vmin, vmax=vmax, title=f"SINDy {title}",
                          xlabel="Time")
    
    fig.colorbar(im0 if im0 else im1, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(0.05)


# =============================================================================
# Error analysis
# =============================================================================

def compute_error_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Compute error metrics."""
    diff = pred - truth
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    
    # Relative error (avoid division by zero)
    denom = np.abs(truth) + 1e-10
    mre = float(np.mean(np.abs(diff) / denom))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MRE': mre
    }


def plot_per_time_error(truth: np.ndarray, pred: np.ndarray,
                        time_axis: np.ndarray = None,
                        title: str = "Per-time Error",
                        save_path: str = None):
    """Plot error over time."""
    if time_axis is None:
        time_axis = np.arange(truth.shape[0])
    
    diff = pred - truth
    mae_t = np.mean(np.abs(diff), axis=1)
    rmse_t = np.sqrt(np.mean(diff**2, axis=1))
    
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(time_axis, mae_t, marker='.', lw=1.0, label='MAE')
    plt.plot(time_axis, rmse_t, marker='.', lw=1.0, label='RMSE')
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(0.05)


def print_error_summary(name: str, pred: np.ndarray, truth: np.ndarray):
    """Print an error summary."""
    metrics = compute_error_metrics(pred, truth)
    print(f"[{name}] MSE={metrics['MSE']:.4e}, RMSE={metrics['RMSE']:.4e}, "
          f"MAE={metrics['MAE']:.4e}, MRE={metrics['MRE']:.4e}")


# =============================================================================
# CSD / Zbar visualization
# =============================================================================

def plot_csd_comparison(truth_csd: np.ndarray, pred_csd: np.ndarray,
                        time_axis: np.ndarray = None,
                        title: str = "CSD Comparison",
                        save_path: str = None):
    """Plot Charge State Distribution comparison."""
    if time_axis is None:
        time_axis = np.arange(truth_csd.shape[0])
    
    nq = truth_csd.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, dpi=150)
    
    vmin, vmax = safe_percentile([truth_csd, pred_csd], 1, 99)
    extent = [time_axis[0], time_axis[-1], 0, nq-1]
    
    im0 = safe_imshow(axes[0], truth_csd.T, extent=extent, cmap="viridis",
                      vmin=vmin, vmax=vmax, title="Truth CSD",
                      xlabel="Time", ylabel="Charge State")
    
    im1 = safe_imshow(axes[1], pred_csd.T, extent=extent, cmap="viridis",
                      vmin=vmin, vmax=vmax, title="Pred CSD",
                      xlabel="Time")
    
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8, label="Population")
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(0.05)


def plot_zbar_comparison(truth_zbar: np.ndarray, pred_zbar: np.ndarray,
                         time_axis: np.ndarray = None,
                         title: str = "Mean Charge Comparison",
                         save_path: str = None):
    """Plot mean-charge comparison."""
    if time_axis is None:
        time_axis = np.arange(len(truth_zbar))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    
    # Comparison
    axes[0].plot(time_axis, truth_zbar, 'k-', lw=1.5, label='Truth')
    axes[0].plot(time_axis, pred_zbar, 'r--', lw=1.5, label='Pred')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mean Charge (Zbar)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)
    
    # Error
    error = pred_zbar - truth_zbar
    axes[1].plot(time_axis, error, 'b-', lw=1.0)
    axes[1].axhline(0, color='k', ls='--', lw=0.5)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error (Pred - Truth)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Zbar Error")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(0.05)


# =============================================================================
# Steady-state visualization
# =============================================================================

def plot_steady_state_comparison(truth_F: np.ndarray, pred_F: np.ndarray,
                                  U_raw: np.ndarray = None,
                                  indices: List[int] = None,
                                  title: str = "Steady-State Comparison",
                                  save_path: str = None):
    """Plot steady-state comparison."""
    n_samples = truth_F.shape[0]
    
    if indices is None:
        indices = np.random.choice(n_samples, min(6, n_samples), replace=False)
    
    n_show = len(indices)
    cols = min(3, n_show)
    rows = int(np.ceil(n_show / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    
    nx = truth_F.shape[1]
    x = np.arange(nx)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(x, truth_F[idx], 'k-', lw=1.5, label='Truth')
        ax.plot(x, pred_F[idx], 'r--', lw=1.0, label='Pred')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1.5)
        ax.grid(True, alpha=0.3)
        
        if U_raw is not None:
            T, n = U_raw[idx, 0], U_raw[idx, 1]
            ax.set_title(f"T={T:.1f}eV, n={n:.2e}", fontsize=9)
        else:
            ax.set_title(f"Sample {idx}")
        
        if i == 0:
            ax.legend(fontsize=8)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.pause(0.05)
