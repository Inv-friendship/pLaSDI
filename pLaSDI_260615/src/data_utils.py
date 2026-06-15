# -*- coding: utf-8 -*-
"""
Data Module
===========
Data loading, preprocessing, and segment management.

Population/history data loading and caching.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch


# =============================================================================
# File I/O
# =============================================================================

def load_pop_matrix_auto(fp: Path, nx: int) -> np.ndarray:
    """
    Automatically load a population matrix and adjust its shape.
    
    Args:
        fp: File path.
        nx: Number of states.
    
    Returns:
        arr: (nt, nx) population matrix.
    """
    arr = np.loadtxt(fp, dtype=float)
    if arr.ndim == 1:
        assert arr.size % nx == 0
        arr = arr.reshape(nx, arr.size // nx)
    return arr.T if arr.shape[0] == nx else arr


def load_history_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a history file (time, control variables).
    """
    # --- NEW: automatically detect whether a header exists (numeric first token means data) ---
    with open(path, "r") as f:
        first = f.readline().strip()

    first_token = ""
    if first:
        # Extract only the first column, whether CSV or whitespace-separated
        first_token = first.split(",")[0].split()[0]

    try:
        float(first_token)
        skip = 0
    except ValueError:
        skip = 1
    # -------------------------------------------------------------

    data = np.loadtxt(path, dtype=float, skiprows=skip)  # <-- only this line changed
    if data.ndim == 1:
        data = data.reshape(1, -1)  # Important: make one-line files (1, ncol)

    if data.shape[1] >= 3:
        t = data[:, 0]
        U = data[:, 1:3]  # T, density
    else:
        t = np.arange(len(data))
        U = data

    return t, U



def guess_history_path(pop_path: str) -> Optional[str]:
    """
    Infer the history file path from a population file path.
    
    density_population_seg{N}.txt -> historyfile_seg{N}.txt
    """
    p = Path(pop_path)
    m = re.search(r'seg(\d+)', p.name)
    if m:
        hist_name = f"historyfile_seg{m.group(1)}.txt"
        hist_path = p.parent / hist_name
        if hist_path.exists():
            return str(hist_path)
    return None


def align_controls(t_h: np.ndarray, U_h: np.ndarray, t_seg: np.ndarray) -> np.ndarray:
    """
    Align control variables to the segment time axis by interpolation.
    
    Args:
        t_h: History time.
        U_h: History control variables.
        t_seg: Segment time.
    
    Returns:
        U_aligned: (L, mu) aligned control variables.
    """
    from scipy.interpolate import interp1d
    
    L = len(t_seg)
    mu = U_h.shape[1]
    U_aligned = np.zeros((L, mu), dtype=np.float64)
    
    for j in range(mu):
        f = interp1d(t_h, U_h[:, j], kind='linear', fill_value='extrapolate')
        U_aligned[:, j] = f(t_seg)
    
    return U_aligned


# =============================================================================
# Cache
# =============================================================================

def _np_cache_path(base_dir: str, data_files: List[str], nx: int) -> Path:
    """Create a cache file path with a unique hash for the file combination."""
    nfiles = len(data_files)
    cache_dir = Path(base_dir) / "np_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(",".join(sorted(data_files)).encode()).hexdigest()[:8]
    return cache_dir / f"pops_{nfiles}_nx{nx}_{key}.npz"


def load_or_build_pops(data_files: List[str], nx: int, base_dir: str) -> List[np.ndarray]:
    """
    Load population data using the cache.
    
    Args:
        data_files: List of population file paths.
        nx: Number of states.
        base_dir: Base directory.
    
    Returns:
        pops: list of (nt_i, nx) arrays
    """
    cache_path = _np_cache_path(base_dir, data_files, nx)
    
    # Try reading the cache
    if cache_path.exists():
        try:
            d = np.load(cache_path, allow_pickle=True)
            cached_nx = int(d["nx"]) if "nx" in d.files else nx
            cached_nseg = int(d["nseg"]) if "nseg" in d.files else None
            
            pops_obj = d["pops"]
            pops = pops_obj.tolist() if isinstance(pops_obj, np.ndarray) else list(pops_obj)
            pops = [np.asarray(p, dtype=np.float64) for p in pops]
            
            if cached_nseg is not None and cached_nseg != len(data_files):
                print(f"[cache] file count mismatch ({cached_nseg}→{len(data_files)}) → rebuild")
                raise ValueError("cache mismatch")
            if cached_nx != nx:
                print(f"[cache] nx mismatch ({cached_nx}→{nx}) → rebuild")
                raise ValueError("cache mismatch")
            
            for i, a in enumerate(pops):
                if a.ndim != 2 or a.shape[1] != nx:
                    raise ValueError(f"bad cached shape pops[{i}]={a.shape}")
            
            print(f"[cache] loaded {cache_path.name} (segments={len(pops)}, nx={nx})")
            return pops
        except Exception as e:
            print(f"[warn] failed to read cache: {e} → rebuild")
    
    # Rebuild cache
    print(f"[cache] rebuilding pops cache (reading {len(data_files)} txt files)...")
    pops: List[np.ndarray] = []
    
    for f in data_files:
        arr = load_pop_matrix_auto(Path(f), nx=nx)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != nx:
            raise ValueError(f"{Path(f).name}: expected (?,{nx}), got {arr.shape}")
        pops.append(arr)
    
    try:
        np.savez_compressed(
            cache_path,
            pops=np.array(pops, dtype=object),
            nseg=np.int64(len(pops)),
            nx=np.int64(nx),
        )
        print(f"[cache] saved → {cache_path.name}")
    except Exception as e:
        print(f"[warn] failed to save cache: {e}")
    
    return pops


# =============================================================================
# Segment/slice management
# =============================================================================

def build_segment_slices(pops: List[np.ndarray]) -> List[slice]:
    """
    Create slices for each segment.
    
    Args:
        pops: list of (nt_i, nx) arrays
    
    Returns:
        slices: list of slice objects
    """
    slices = []
    start = 0
    for p in pops:
        L = p.shape[0]
        slices.append(slice(start, start + L))
        start += L
    return slices


def build_segment_spans(pops: List[np.ndarray]) -> List[Tuple[int, int]]:
    """Return (start, end) spans for each segment."""
    spans = []
    start = 0
    for p in pops:
        L = p.shape[0]
        spans.append((start, start + L))
        start += L
    return spans


def parse_selection_spec(spec: str, nseg: int) -> List[int]:
    """
    Parse a visualization segment selection spec.
    
    Example: "6, 148, 175" -> [5, 147, 174] (0-indexed)
    Example: "1:10" -> [0, 1, ..., 9]
    """
    if not spec or str(spec).strip().lower() == "all":
        return list(range(nseg))
    
    out = set()
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    
    for p in parts:
        if ":" in p:
            a, b = p.split(":")
            a, b = int(a), int(b)
            if a <= 0 or b <= 0:
                continue
            lo, hi = min(a, b), max(a, b)
            for k in range(lo, hi + 1):
                if 1 <= k <= nseg:
                    out.add(k - 1)
        else:
            k = int(p)
            if 1 <= k <= nseg:
                out.add(k - 1)
    
    return sorted(out)


# =============================================================================
# Train/Val ë¶"í• 
# =============================================================================

def split_train_val_random_segments(segment_slices: List[slice], 
                                     val_ratio: float,
                                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                               List[slice], List[slice],
                                                               List[int], List[int]]:
    """
    Random train/validation split by segment.
    
    Returns:
        train_idx, val_idx: Global indices.
        train_slices, val_slices: Slice lists.
        train_seg_ids, val_seg_ids: Segment ID lists.
    """
    import random
    
    n_seg = len(segment_slices)
    all_seg_ids = list(range(n_seg))
    
    rng = random.Random(seed)
    rng.shuffle(all_seg_ids)
    
    n_val_seg = max(1, int(round(n_seg * val_ratio)))
    val_seg_ids = sorted(all_seg_ids[:n_val_seg])
    train_seg_ids = sorted(all_seg_ids[n_val_seg:])
    
    # Collect indices
    train_idx = []
    val_idx = []
    train_slices = []
    val_slices = []
    
    for sid in train_seg_ids:
        sl = segment_slices[sid]
        train_idx.extend(range(sl.start, sl.stop))
        train_slices.append(sl)
    
    for sid in val_seg_ids:
        sl = segment_slices[sid]
        val_idx.extend(range(sl.start, sl.stop))
        val_slices.append(sl)
    
    return (np.array(train_idx), np.array(val_idx),
            train_slices, val_slices,
            train_seg_ids, val_seg_ids)


# =============================================================================
# Mini-batch utilities
# =============================================================================

def iter_case_minibatches(all_slices: List[slice], batch_size: int):
    """Case-wise minibatch generator."""
    n_cases = len(all_slices)
    for i in range(0, n_cases, batch_size):
        yield all_slices[i:i + batch_size]


def build_local_indices_for_batch(slices_batch: List[slice]) -> Tuple[np.ndarray, List[slice]]:
    """
    Build local indices for a minibatch.
    
    Returns:
        idx_all: ì "ì²´ time index (1D numpy array)
        local_slices: Local slices within the batch.
    """
    idx_list = []
    local_slices = []
    offset = 0
    
    for sl in slices_batch:
        s, e = sl.start, sl.stop
        L = e - s
        idx_list.append(np.arange(s, e, dtype=np.int64))
        local_slices.append(slice(offset, offset + L))
        offset += L
    
    idx_all = np.concatenate(idx_list, axis=0)
    return idx_all, local_slices


# =============================================================================
# Steady-state data
# =============================================================================

def load_steady_pair(pop_path: str, hist_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a steady-state data pair.
    
    Returns:
        P: (nx, K) population
        Tn: (K, 2) [T, n] control
    """
    P = np.loadtxt(pop_path, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if P.shape[0] != P.shape[1]:
        if P.shape[1] > P.shape[0]:
            P = P.T

    # --- Replaced this part only to fix header/time handling ---
    t, U = load_history_file(hist_path)   # U: (K, mu)
    # ------------------------------------------------

    # Keep the existing logic: use only [T, n]
    if U.ndim == 1:
        U = U.reshape(1, -1)

    if U.shape[1] >= 2:
        Tn = U[:, :2]
    else:
        Tn = U

    return P, Tn



class SteadyStateData:
    """Steady-state data manager."""
    
    def __init__(self, pop_hist_pairs: List[Tuple[str, str]], 
                 pop_scaler, ctrl_scaler,
                 random_pick: bool = False,
                 num_samples: int = 200,
                 seed: int = 42,
                 pop_lim: float = 1e-50):
        """
        Args:
            pop_hist_pairs: List of (pop_path, hist_path) pairs.
            pop_scaler: PopulationScaler
            ctrl_scaler: ControlScaler
            random_pick: Whether to sample randomly.
            num_samples: Number of samples.
            seed: Random seed.
            pop_lim: Lower population bound.
        """
        self.pop_scaler = pop_scaler
        self.ctrl_scaler = ctrl_scaler
        self.pop_lim = pop_lim
        
        rng = np.random.default_rng(seed)
        pairs = pop_hist_pairs[:]
        
        if random_pick:
            rng.shuffle(pairs)
            pairs = pairs[:num_samples]
        
        self.W_list = []
        self.U_list = []
        self.P_list = []
        self.Uraw_list = []
        
        for pop_path, hist_path in pairs:
            P, Tn = load_steady_pair(pop_path, hist_path)
            K = P.shape[1]
            
            for k in range(K):
                pk = P[:, k]
                # Apply pop_lim
                pk = np.clip(pk, self.pop_lim, None)
                pk_norm = pk / (pk.sum() + 1e-300)
                
                # W transform
                Wk = pop_scaler.transform(pk_norm.reshape(1, -1)).flatten()
                
                # U transform
                Uk_raw = Tn[k, :].reshape(1, -1)
                Uk = ctrl_scaler.transform(Uk_raw).flatten()
                
                self.P_list.append(pk)
                self.W_list.append(Wk)
                self.U_list.append(Uk)
                self.Uraw_list.append(Tn[k, :])
        
        if len(self.W_list) > 0:
            self.W_all = np.stack(self.W_list, axis=0)
            self.U_all = np.stack(self.U_list, axis=0)
            self.P_all = np.stack(self.P_list, axis=0)
            self.Uraw_all = np.stack(self.Uraw_list, axis=0)
            self.enabled = True
            print(f"[SteadyState] loaded {len(self.W_list)} samples")
        else:
            self.enabled = False
            print("[SteadyState] no valid samples")
    
    def to_torch(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors."""
        W_t = torch.tensor(self.W_all, dtype=dtype, device=device)
        U_t = torch.tensor(self.U_all, dtype=dtype, device=device)
        return W_t, U_t


# =============================================================================
# State Names
# =============================================================================

def load_state_names(path: str, nx: int) -> Optional[List[str]]:
    """Load a state-name file."""
    p = Path(path)
    if not p.exists():
        print(f"[info] names_file '{path}' not found")
        return None
    
    try:
        with open(p, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[info] failed to read names_file: {e}")
        return None
    
    if len(names) != nx:
        print(f"[info] len(state_names)={len(names)} != nx={nx}")
        return None
    
    return names


def load_label_subset(path: str) -> Optional[List[str]]:
    """Load a label subset file."""
    p = Path(path)
    if not p.exists():
        return None
    
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]
