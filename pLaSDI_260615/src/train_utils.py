# -*- coding: utf-8 -*-
"""
Training Utilities Module
=========================
Training utilities for timers, logging, learning-rate management, checkpoints, and related helpers.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, List, Dict, Any

import numpy as np
import torch


# =============================================================================
# Timers
# =============================================================================

class CudaTimer:
    """Timer that accounts for CUDA synchronization."""
    
    def __init__(self, sync: bool = True):
        self.t = 0.0
        self._t0 = None
        self.sync = sync
    
    def reset(self):
        self.t, self._t0 = 0.0, None
    
    def start(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.time()
    
    def stop(self):
        if self._t0 is not None:
            if self.sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t += time.time() - self._t0
            self._t0 = None
    
    def value(self) -> float:
        return float(self.t)


class EpochTimer:
    """Epoch ETA timer."""
    
    def __init__(self, smooth: int = 100):
        self.smooth = max(1, smooth)
        self.buf = deque(maxlen=self.smooth)
        self.t0 = None
    
    def start(self):
        self.t0 = time.time()
    
    def stop_and_update(self) -> Optional[float]:
        if self.t0 is None:
            return None
        dtv = time.time() - self.t0
        self.buf.append(dtv)
        self.t0 = None
        return dtv
    
    def eta_str(self, current_ep: int, total_ep: int) -> str:
        if not self.buf:
            return "ETA: --"
        
        avg = sum(self.buf) / len(self.buf)
        left = max(0, total_ep - current_ep)
        sec_left = avg * left
        finish = datetime.now() + timedelta(seconds=sec_left)
        
        def _fmt(sec):
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)
            parts = []
            if d: parts.append(f"{d}d")
            if h: parts.append(f"{h}h")
            if m: parts.append(f"{m}m")
            parts.append(f"{s}s")
            return " ".join(parts)
        
        return f"ETA: ~{_fmt(sec_left)} (finish ≈ {finish.strftime('%Y-%m-%d %H:%M:%S')})"


# =============================================================================
# Logging
# =============================================================================

class TeeLogger:
    """Log to both console and file."""
    
    def __init__(self, filename: str, mode: str = "a"):
        self.out_terminal = sys.stdout
        self.err_terminal = sys.stderr
        self.log = open(filename, mode, encoding="utf-8")
    
    def write(self, message):
        if sys.stdout is self:
            self.out_terminal.write(message)
            self.log.write(message)
        elif sys.stderr is self:
            self.err_terminal.write(message)
            self.log.write(message)
        else:
            self.out_terminal.write(message)
            self.log.write(message)
    
    def flush(self):
        try: self.out_terminal.flush()
        except (OSError, ValueError): pass
        try: self.err_terminal.flush()
        except (OSError, ValueError): pass
        try: self.log.flush()
        except (OSError, ValueError): pass
    
    def close(self):
        try: self.log.close()
        except (OSError, ValueError): pass


# =============================================================================
# Learning-rate management
# =============================================================================

def get_lr(optimizer) -> float:
    """Return the current learning rate."""
    return optimizer.param_groups[0]["lr"]


def set_lr(optimizer, new_lr: float):
    """Set the learning rate."""
    for g in optimizer.param_groups:
        g["lr"] = new_lr


def in_frozen_span(epoch: int, spans: List[tuple]) -> bool:
    """Check whether the epoch is inside a frozen interval."""
    return any(s <= epoch <= e for s, e in spans)


def ramp_weight(epoch: int, T: int, target: float, mode: str = "linear") -> float:
    """
    Compute the weight ramp-up.
    
    Args:
        epoch: Current epoch.
        T: Ramp-up duration.
        target: Target weight.
        mode: "linear", "exp_late", "exp_early", "cos"
              - exp_late: slow early and steep late (stabilizes AE first)
              - exp_early: steep early and saturated late (introduces auxiliary losses early)
    
    Returns:
        Current weight.
    """
    if T <= 0:
        return target
    
    t = min(epoch / T, 1.0)
    
    if mode == "linear":
        return target * t
    elif mode == "exp_late":
        # Slow early and steep late: (exp(3t) - 1) / (exp(3) - 1)
        return target * (np.exp(3 * t) - 1) / (np.exp(3) - 1)
    elif mode == "exp_early":
        # Steep early and saturated late: 1 - exp(-3t)
        return target * (1 - np.exp(-3 * t))
    elif mode == "exp_slow":
        # Backward compatibility: same as exp_late
        return target * (np.exp(3 * t) - 1) / (np.exp(3) - 1)
    elif mode == "cos":
        return target * 0.5 * (1 - np.cos(np.pi * t))
    else:
        return target * t


# =============================================================================
# CSV logging
# =============================================================================

def append_csv_header_if_needed(path: str, header: str):
    """Add a CSV header if it does not exist."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        with open(p, "w", encoding="utf-8") as f:
            f.write(header + "\n")


def append_loss_csv(path: str, epoch: int, loss: float):
    """Append a loss value to CSV."""
    append_csv_header_if_needed(path, "epoch,loss")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{loss:.12e}\n")


def append_metrics_csv(path: str, row: dict, header_order: List[str]):
    """Append one metrics row to CSV."""
    append_csv_header_if_needed(path, ",".join(header_order))
    
    out = []
    for k in header_order:
        v = row.get(k, None)
        if v is None:
            out.append("")
        else:
            try:
                if isinstance(v, (int, np.integer)):
                    out.append(f"{int(v)}")
                elif isinstance(v, (float, np.floating)):
                    if np.isfinite(float(v)):
                        out.append(f"{float(v):.12e}")
                    else:
                        out.append("")
                else:
                    out.append(str(v))
            except Exception:
                out.append("")
    
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(out) + "\n")


def csv_to_npy(csv_path: str, npy_path: str):
    """Convert CSV to NPY."""
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return
    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=1, dtype=float)
        if data.ndim == 0:
            data = np.array([float(data)])
        np.save(npy_path, data)
    except Exception:
        pass


def metrics_csv_to_npz(csv_path: str, npz_path: str):
    """Convert metrics CSV to NPZ."""
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return
    
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    
    if not lines:
        return
    
    header = lines[0].split(",")
    cols = {h: [] for h in header}
    
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        for h, cell in zip(header, parts):
            if cell == "":
                cols[h].append(np.nan)
            else:
                try:
                    cols[h].append(float(cell))
                except Exception:
                    cols[h].append(np.nan)
    
    data = {k: np.array(v, dtype=float) for k, v in cols.items()}
    np.savez(npz_path, **data)


# =============================================================================
# GPU-CPU Transfer Profiler
# =============================================================================

class TransferProfiler:
    """
    Tool for measuring GPU-to-CPU transfer overhead.
    
    Accumulates GPU synchronization and transfer time caused by .item() calls.
    
    Usage:
        profiler = TransferProfiler()
        
        # Section to measure
        with profiler.measure():
            val = tensor.item()
        
        # Print at the end of the epoch
        profiler.report()
        profiler.reset()
    """
    
    def __init__(self):
        self.total_time = 0.0
        self.call_count = 0
        self._t0 = None
    
    def reset(self):
        self.total_time = 0.0
        self.call_count = 0
        self._t0 = None
    
    class _MeasureCtx:
        def __init__(self, profiler):
            self._p = profiler
        
        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._p._t0 = time.time()
            return self
        
        def __exit__(self, *args):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self._p._t0 is not None:
                self._p.total_time += time.time() - self._p._t0
                self._p.call_count += 1
                self._p._t0 = None
    
    def measure(self):
        """Measure a section with a context manager."""
        return self._MeasureCtx(self)
    
    def record(self, n: int = 1):
        """
        Record only the call count after sync is already complete, without measuring time.
        """
        self.call_count += n
    
    def report(self, epoch: int = None, total_epoch_time: float = None) -> str:
        """
        Return a measurement report string.
        
        Args:
            epoch: Current epoch for display.
            total_epoch_time: Total epoch time for ratio calculation.
        
        Returns:
            report string
        """
        ep_str = f"[Epoch {epoch}] " if epoch is not None else ""
        ratio_str = ""
        if total_epoch_time and total_epoch_time > 0:
            ratio = 100 * self.total_time / total_epoch_time
            ratio_str = f" ({ratio:.1f}% of epoch)"
        
        return (
            f"{ep_str}[Transfer] GPU→CPU overhead: "
            f"{self.total_time*1000:.2f}ms | "
            f"calls={self.call_count}{ratio_str}"
        )


# =============================================================================
# Checkpoints
# =============================================================================

def save_checkpoint(path: str, model, optimizer=None, scheduler=None, 
                    epoch: int = 0, extra: dict = None):
    """Save a checkpoint."""
    ckpt = {
        'model_state': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        ckpt['opt_state'] = optimizer.state_dict()
        ckpt['current_lr'] = get_lr(optimizer)
        ckpt['lrs'] = [g["lr"] for g in optimizer.param_groups]
    
    if scheduler is not None:
        ckpt['sched_state'] = scheduler.state_dict()
    
    if extra is not None:
        ckpt.update(extra)
    
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, 
                    device=None) -> dict:
    """Load a checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(ckpt['model_state'])
    
    if optimizer is not None and 'opt_state' in ckpt:
        optimizer.load_state_dict(ckpt['opt_state'])
        if 'lrs' in ckpt:
            for g, lr_g in zip(optimizer.param_groups, ckpt['lrs']):
                g["lr"] = lr_g
    
    if scheduler is not None and 'sched_state' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['sched_state'])
        except Exception as e:
            print(f"[warn] Failed to restore scheduler: {e}")
    
    return ckpt


# =============================================================================
# Metrics header
# =============================================================================

METRICS_HEADER = [
    "epoch", "total", "recW", "FSE", "frac", "ion", "zbar",
    "sindy", "coef", "sindy_norm", "coef_norm", "hurwitz", "steady_raw",
    "lr", "w_fse", "w_frac", "w_ion", "w_zbar", "val_total", "steady",
    "recW_s", "fse_s", "frac_s", "ion_s", "zbar_s",
    "w_sindy", "w_coef", "w_hurwitz", "w_steady",
    "rate_W", "rate_N", "rate_CSD", "rate_Zbar",
    "w_rate_W", "w_rate_N", "w_rate_CSD", "w_rate_Zbar",
    "condA_max", "steady_skipped",
]
