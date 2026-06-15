# -*- coding: utf-8 -*-
"""
SINDy Module
============
Utilities for SINDyC (SINDy with Control).

Dynamics model: dZ/dt = a + A·Z + B·U
- a: (nz,) bias vector
- A: (nz, nz) state matrix
- B: (nz, mu) control input matrix
- Z: (N, nz) latent state
- U: (N, mu) control variables (T, density)

Future extensions:
- A = A(U): nonlinear dynamics depending on control variables.
- Support for other basis functions.
"""

from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F


def split_coefs_torch(coef_vec: torch.Tensor, nz: int, mu: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a SINDyC coefficient vector into a, A, and B (PyTorch).
    
    coef_vec structure: flattened [a (nz), A (nz*nz), B (nz*mu)].
    
    Args:
        coef_vec: (p,) or (p, nz) coefficient vector.
        nz: latent dimension
        mu: control dimension
    
    Returns:
        a: (nz,) bias
        A: (nz, nz) state matrix
        B: (nz, mu) control matrix
    """
    if coef_vec.dim() == 1:
        C = coef_vec.view(-1, nz)
    elif coef_vec.dim() == 2:
        C = coef_vec
    else:
        raise ValueError(f"coef_vec shape {tuple(coef_vec.shape)} not supported")
    
    p = C.size(0)
    if p < (1 + nz + mu):
        raise ValueError(f"coeff length {p} < 1+nz+mu")

    a = C[0, :]                        # (nz,)
    A = C[1:1+nz, :].t().contiguous()  # (nz, nz)
    
    if mu > 0:
        B = C[1+nz:1+nz+mu, :].t().contiguous()  # (nz, mu)
    else:
        B = torch.zeros(nz, 0, dtype=C.dtype, device=C.device)
    
    return a, A, B


def zstar_exact_torch(U_t: torch.Tensor, a_t: torch.Tensor, 
                      A_t: torch.Tensor, B_t: torch.Tensor,
                      cond_warn: float = 1e8) -> Tuple[torch.Tensor, float]:
    """
    Compute steady-state Z*.
    
    dZ/dt = 0 → a + A·Z* + B·U = 0
    → Z* = -A^(-1) · (a + B·U)
    
    Args:
        U_t: (N, mu) control variables
        a_t: (nz,) bias
        A_t: (nz, nz) state matrix
        B_t: (nz, mu) control matrix
        cond_warn: Condition-number warning threshold.
    
    Returns:
        Z_star: (N, nz) steady-state latent vectors
        condA:  float, condition number of A (inf if calculation fails)
    """
    # rhs = -(a + B·U) → (N, nz)
    rhs = -(a_t.unsqueeze(0) + U_t @ B_t.T)

    # Compute cond(A)
    condA = float('inf')
    try:
        with torch.no_grad():
            s = torch.linalg.svdvals(A_t)
            s_min = s.min()
            if s_min.abs() > 0:
                condA = (s.max() / s_min).item()
            if condA > cond_warn:
                print(f"[zstar_exact] Warning: cond(A) = {condA:.2e} (possible numerical instability)")
    except Exception as e:
        print(f"[zstar_exact] Failed to compute cond(A): {e}")

    # A·Z* = rhs → Z* = A^(-1)·rhs
    Z_star = torch.linalg.solve(A_t, rhs.T).T
    return Z_star, condA


def eig_extrema(A: torch.Tensor) -> Tuple[float, float]:
    """
    Return the maximum and minimum real parts of eigenvalues of matrix A.
    
    Args:
        A: (nz, nz) matrix.
    
    Returns:
        (max_real, min_real)
    """
    ev = torch.linalg.eigvals(A)
    max_real = float(ev.real.max().item())
    min_real = float(ev.real.min().item())
    return max_real, min_real


def hurwitz_gate_check(max_real: float, min_real: float, 
                       min_real_threshold: float = 0.0) -> Tuple[bool, bool]:
    """
    Check the Hurwitz stability gate.
    
    Args:
        max_real: Maximum eigenvalue real part.
        min_real: Minimum eigenvalue real part.
        min_real_threshold: Minimum eigenvalue threshold.
    
    Returns:
        (hurwitz_ok, strong_ok)
        - hurwitz_ok: all Re(λ) < 0
        - strong_ok: min Re(λ) < threshold
    """
    hurwitz_ok = (max_real < 0.0)
    strong_ok = (min_real < min_real_threshold)
    return hurwitz_ok, strong_ok


def hurwitz_penalty_symmetric(A: torch.Tensor, margin: float = 0.0,
                               gate_enable: bool = False,
                               gate_min_real: float = 0.0) -> torch.Tensor:
    """
    Hurwitz stability penalty using eigenvalues of the symmetrized matrix.
    
    penalty = ReLU(λ_max((A+A^T)/2) + margin)
    
    Args:
        A: (nz, nz) or (B, nz, nz) state matrix.
        margin: Stability margin.
        gate_enable: Whether to add a minimum-eigenvalue constraint.
        gate_min_real: Minimum eigenvalue threshold.
    
    Returns:
        penalty: scalar
    """
    S = 0.5 * (A + A.transpose(-1, -2))
    evals = torch.linalg.eigvalsh(S)  # Real-symmetric -> real eigenvalues
    lam_max = evals[..., -1]
    pen = F.relu(lam_max + margin)
    
    if gate_enable:
        lam_min = evals[..., 0]
        pen_min = F.relu(lam_min - gate_min_real)
        pen = pen + pen_min
    
    if pen.dim() > 0:
        return pen.mean()
    return pen


class SINDyLossCalculator:
    """
    SINDy loss calculator.
    
    Handles SINDy loss computation for minibatches.
    """
    
    def __init__(self, ld, nz: int, dt_eff: float, device: torch.device,
                 dtype: torch.dtype = torch.float64, use_cpu: bool = False):
        """
        Args:
            ld: SINDyC latent dynamics object.
            nz: latent dimension
            dt_eff: effective dt
            device: torch device
            dtype: torch dtype
            use_cpu: Whether to run calibration on CPU.
        """
        self.ld = ld
        self.nz = nz
        self.dt_eff = dt_eff
        self.device = device
        self.dtype = dtype
        self.use_cpu = use_cpu
        self._warned_lengths = set()
    
    def compute_loss_from_precomputed(self, Z_store: torch.Tensor,
                                       U_batch: torch.Tensor,
                                       local_slices: List[slice],
                                       require_grad: bool = True,
                                       reduce: str = "sum") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SINDy loss from precomputed Z.
        
        Args:
            Z_store: (T_batch, nz) latent vectors
            U_batch: (T_batch, mu) control variables
            local_slices: Local index slices for each case.
            require_grad: Whether gradients are required.
            reduce: "sum" or "mean"
        
        Returns:
            (L_sindy, L_coef)
        """
        ctx = (torch.enable_grad() if require_grad else torch.no_grad())
        
        with ctx:
            tot_si = torch.zeros((), dtype=self.dtype, device=self.device)
            tot_cf = torch.zeros((), dtype=self.dtype, device=self.device)
            
            Z_flat = Z_store.squeeze(1) if Z_store.dim() == 3 else Z_store
            Z_flat = Z_flat.to(device=self.device, dtype=self.dtype)
            
            if isinstance(U_batch, torch.Tensor):
                U_full = U_batch.to(device=Z_flat.device, dtype=Z_flat.dtype)
            else:
                U_full = torch.as_tensor(U_batch, device=Z_flat.device, dtype=Z_flat.dtype)
            
            # Group by length
            len2Z_list, len2U_list = {}, {}
            
            for sl in local_slices:
                if isinstance(sl, tuple):
                    sl = slice(int(sl[0]), int(sl[1]))
                
                L = sl.stop - sl.start
                Z_seg = Z_flat[sl.start:sl.stop]
                U_seg = U_full[sl.start:sl.stop]
                
                len2Z_list.setdefault(L, []).append(Z_seg)
                len2U_list.setdefault(L, []).append(U_seg)
            
            for L, Z_list in len2Z_list.items():
                if not Z_list:
                    continue
                
                ZB = torch.stack(Z_list, dim=0)  # (B, L, nz)
                UB = torch.stack(len2U_list[L], dim=0)  # (B, L, mu)
                B = ZB.size(0)
                
                # Update FD operator
                if getattr(self.ld, "nt", None) != L:
                    self.ld.nt = L
                    self.ld.fd_oper, _, _ = self.ld.fd.getOperators(L)
                
                # Select CPU/GPU
                if self.use_cpu:
                    ZB_c = ZB.to("cpu")
                    UB_c = UB.to("cpu")
                    if hasattr(self.ld, "fd_oper") and isinstance(self.ld.fd_oper, torch.Tensor):
                        self.ld.fd_oper = self.ld.fd_oper.to(device="cpu", dtype=ZB_c.dtype)
                    target_Z, target_U = ZB_c, UB_c
                else:
                    if hasattr(self.ld, "fd_oper") and isinstance(self.ld.fd_oper, torch.Tensor):
                        self.ld.fd_oper = self.ld.fd_oper.to(device=ZB.device, dtype=ZB.dtype)
                    target_Z, target_U = ZB, UB
                
                try:
                    _, Ls, Lc = self.ld.calibrate(target_Z, target_U, float(self.dt_eff),
                                                   compute_loss=True, numpy=False)
                    if reduce == "mean":
                        tot_si = tot_si + (Ls.to(self.device) / max(1, B))
                        tot_cf = tot_cf + (Lc.to(self.device) / max(1, B))
                    else:
                        tot_si = tot_si + Ls.to(self.device)
                        tot_cf = tot_cf + Lc.to(self.device)
                        
                except (RuntimeError, torch.linalg.LinAlgError):
                    if L not in self._warned_lengths:
                        print(f"[SINDyC] length={L}: batch calibrate failed, falling back to per-segment loop")
                        self._warned_lengths.add(L)
                    
                    # Per-segment fallback
                    si_acc = torch.zeros((), dtype=self.dtype, device=self.device)
                    cf_acc = torch.zeros((), dtype=self.dtype, device=self.device)
                    
                    for Zi, Ui in zip(target_Z, target_U):
                        _, Ls_i, Lc_i = self.ld.calibrate(Zi, Ui, float(self.dt_eff),
                                                          compute_loss=True, numpy=False)
                        si_acc = si_acc + Ls_i.to(self.device)
                        cf_acc = cf_acc + Lc_i.to(self.device)
                    
                    if reduce == "mean" and B > 0:
                        tot_si = tot_si + si_acc / B
                        tot_cf = tot_cf + cf_acc / B
                    else:
                        tot_si = tot_si + si_acc
                        tot_cf = tot_cf + cf_acc
            
            return tot_si, tot_cf
    
    def compute_global_coefs(self, Z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Compute global SINDy coefficients.
        
        Args:
            Z: (N, nz) latent vectors
            U: (N, mu) control variables
        
        Returns:
            coef_vec: Coefficient vector.
        """
        ZB = Z.unsqueeze(0)  # (1, N, nz)
        UB = U.unsqueeze(0)  # (1, N, mu)
        
        coefsB = self.ld.calibrate(ZB, UB, float(self.dt_eff), 
                                    compute_loss=False, numpy=False)
        return coefsB.squeeze(0)
    
    def compute_hurwitz_penalty(self, Z: torch.Tensor, U: torch.Tensor,
                                 margin: float = 0.0,
                                 gate_enable: bool = False,
                                 gate_min_real: float = 0.0) -> Tuple[torch.Tensor, float, float]:
        """
        Compute the Hurwitz penalty.
        
        Returns:
            (penalty, max_real, min_real)
        """
        coef_vec = self.compute_global_coefs(Z, U)
        mu = U.size(1)
        a, A, B = split_coefs_torch(coef_vec, self.nz, mu)
        
        penalty = hurwitz_penalty_symmetric(A, margin, gate_enable, gate_min_real)
        max_real, min_real = eig_extrema(A)
        
        return penalty, max_real, min_real


def save_sindy_coefs(path: str, coefs_list: List[np.ndarray], 
                     nz: int, mu: int, segment_labels: Optional[List[str]] = None):
    """
    Save SINDy coefficients.
    
    Args:
        path: Save path.
        coefs_list: Coefficient list.
        nz: latent dimension
        mu: control dimension
        segment_labels: Segment labels.
    """
    if segment_labels is None:
        segment_labels = [f"seg{i+1}" for i in range(len(coefs_list))]
    
    np.savez(path,
             coefs_all=np.array(coefs_list, dtype=object),
             nz=np.int64(nz),
             mu=np.int64(mu),
             segment_labels=np.array(segment_labels))


def load_sindy_coefs(path: str) -> Tuple[List[np.ndarray], int, int]:
    """
    Load SINDy coefficients.
    
    Returns:
        (coefs_list, nz, mu)
    """
    data = np.load(path, allow_pickle=True)
    coefs_all = data['coefs_all'].tolist()
    nz = int(data['nz'])
    mu = int(data['mu'])
    return coefs_all, nz, mu
