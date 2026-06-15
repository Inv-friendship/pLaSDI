# -*- coding: utf-8 -*-
"""
Atomic Physics Module
=====================
Compute CSD (Charge State Distribution), FSE (Fraction Sum Error), and Zbar (mean charge).

Physics utilities for computing ion distributions and mean charge from population fractions.
"""

import re
from typing import Optional, List, Tuple

import numpy as np
import torch


class AtomicPhysics:
    """
    Atomic-physics calculation class.
    
    From population fractions:
    - CSD (Charge State Distribution): distribution by charge state.
    - Zbar (Mean Charge): mean charge state.
    - FSE (Fraction Sum Error): fraction sum error.
    """
    
    def __init__(self, state_names: Optional[List[str]], nx: int, dtype: torch.dtype = torch.float64):
        """
        Args:
            state_names: List of state names (for example, ['1s2', '2s', ..., '29+']).
            nx: Number of states.
            dtype: PyTorch dtype
        """
        self.nx = nx
        self.dtype = dtype
        self.state_names = state_names
        
        # Build ionization index
        self.ion_available = False
        self.onehot_q_np = None
        self.onehot_q_t = None
        self.q_vec_t = None
        self.Z0 = None
        self.charge_idx = None
        
        if state_names is not None:
            try:
                self._build_charge_index(state_names)
                self.ion_available = True
            except Exception as e:
                print(f"[AtomicPhysics] Failed to build ionization index: {e}")
    
    def _prefix2(self, name: str) -> str:
        """Extract the prefix from a state name."""
        if re.fullmatch(r'\d{2}\+', name):
            return name
        return name[:2].lower()
    
    def _build_charge_index(self, state_names: List[str]):
        """Automatically build charge indices from state names."""
        # Find the 'NN+' pattern (fully ionized state)
        plus_pos, Z0 = None, None
        for i in range(len(state_names) - 1, -1, -1):
            m = re.fullmatch(r'(\d{2})\+', state_names[i])
            if m:
                plus_pos, Z0 = i, int(m.group(1))
                break
        
        if plus_pos is None:
            raise ValueError("Could not find an 'NN+' pattern (for example, '29+') in name_total.")
        
        # Determine element order
        element_order, seen = [], set()
        for i in range(plus_pos - 1, -1, -1):
            key = self._prefix2(state_names[i])
            if key == f"{Z0:02d}+":
                continue
            if key not in seen:
                seen.add(key)
                element_order.append(key)
        
        elem_to_E = {key: idx + 1 for idx, key in enumerate(element_order)}
        
        # Create charge-index array
        charge_idx = np.empty(len(state_names), dtype=int)
        unknown = set()
        for i, nm in enumerate(state_names):
            if re.fullmatch(r'\d{2}\+', nm):
                charge_idx[i] = Z0
                continue
            key = self._prefix2(nm)
            if key not in elem_to_E:
                unknown.add(key)
                charge_idx[i] = -1
            else:
                charge_idx[i] = Z0 - elem_to_E[key]
        
        if unknown:
            raise ValueError(f"Prefixes could not be inferred automatically: {sorted(unknown)}")
        
        self.Z0 = Z0
        self.charge_idx = charge_idx
        
        # Create one-hot matrix
        nq = Z0 + 1
        onehot_q = np.zeros((self.nx, nq), dtype=np.float64)
        for i, qi in enumerate(charge_idx):
            if 0 <= qi < nq:
                onehot_q[i, qi] = 1.0
        
        self.onehot_q_np = onehot_q
        self.onehot_q_t = torch.tensor(onehot_q, dtype=self.dtype)
        self.q_vec_t = torch.arange(nq, dtype=self.dtype)
        self.nq = nq
    
    def compute_csd_numpy(self, fraction: np.ndarray) -> np.ndarray:
        """
        Compute Charge State Distribution (NumPy).
        
        Args:
            fraction: (N, nx) population fraction
        
        Returns:
            csd: (N, nq) charge state distribution
        """
        if not self.ion_available:
            raise RuntimeError("Ion index not available")
        return fraction @ self.onehot_q_np
    
    def compute_csd_torch(self, fraction: torch.Tensor) -> torch.Tensor:
        """
        Compute Charge State Distribution (PyTorch).
        
        Args:
            fraction: (N, nx) or (N, 1, 1, nx) population fraction
        
        Returns:
            csd: (N, nq) charge state distribution
        """
        if not self.ion_available:
            raise RuntimeError("Ion index not available")
        
        # Shape handling
        if fraction.dim() == 4:
            fraction = fraction.view(fraction.size(0), -1)
        elif fraction.dim() == 3:
            fraction = fraction.view(fraction.size(0), -1)
        
        onehot = self.onehot_q_t.to(device=fraction.device, dtype=fraction.dtype)
        return fraction @ onehot
    
    def compute_zbar_numpy(self, fraction: np.ndarray) -> np.ndarray:
        """
        Compute mean charge (NumPy).
        
        Args:
            fraction: (N, nx) population fraction
        
        Returns:
            zbar: (N,) mean charge.
        """
        csd = self.compute_csd_numpy(fraction)
        q_vec = np.arange(self.nq, dtype=np.float64)
        return np.sum(csd * q_vec, axis=1)
    
    def compute_zbar_torch(self, fraction: torch.Tensor) -> torch.Tensor:
        """
        Compute mean charge (PyTorch).
        
        Args:
            fraction: (N, nx) or (N, 1, 1, nx) population fraction
        
        Returns:
            zbar: (N,) mean charge.
        """
        csd = self.compute_csd_torch(fraction)
        q_vec = self.q_vec_t.to(device=csd.device, dtype=csd.dtype)
        return torch.sum(csd * q_vec, dim=1)
    
    def compute_fse_torch(self, fraction_sum: torch.Tensor, target: float = 1.0) -> torch.Tensor:
        """
        Compute Fraction Sum Error (PyTorch).
        sum(fraction) should be 1.
        
        Args:
            fraction_sum: (N, 1) sum of fractions.
            target: Target value (default 1.0).
        
        Returns:
            fse: scalar, MSE
        """
        return torch.mean((fraction_sum - target) ** 2)


class PhysicsLoss:
    """
    Physics loss calculation class.
    
    L_frac: Fraction reconstruction loss (values above tau only).
    L_ion:  Ion distribution loss.
    L_zbar: Mean charge loss.
    L_fse:  Fraction sum error loss.
    L_rate_W: W-space rate equation loss.
    L_rate_N: N-space rate equation loss.
    
    Rate equation losses use the same SBP FD operator as SINDy.
    """
    
    def __init__(self, atomic_physics: AtomicPhysics, tau: float = 1e-100,
                 fd_type: str = 'sbp12', dt_eff: Optional[float] = None):
        """
        Args:
            atomic_physics: AtomicPhysics instance.
            tau: fraction mask threshold
            fd_type: SBP FD operator type ('sbp12', 'sbp24', 'sbp36', 'sbp48').
            dt_eff: Effective dt for time derivatives (if None, use the dt passed at call time).
        """
        self.ap = atomic_physics
        self.tau = tau
        self.fd_type = fd_type
        self.dt_eff = dt_eff
        
        # Initialize FD operator (regenerated later to match the sequence length)
        self._fd_class = None
        self._fd_oper = None
        self._fd_nt = None
        self._init_fd_operator()
    
    def _init_fd_operator(self):
        """Load the FD operator class."""
        try:
            from .fd import FDdict
        except ImportError:
            from fd import FDdict
        if self.fd_type in FDdict:
            self._fd_class = FDdict[self.fd_type]
        else:
            raise ValueError(f"Unknown fd_type: {self.fd_type}. Available: {list(FDdict.keys())}")
    
    def _get_fd_operator(self, nt: int, device, dtype):
        """Return the FD operator for the sequence length (cached)."""
        if self._fd_nt != nt:
            self._fd_oper, _, _ = self._fd_class.getOperators(nt)
            self._fd_nt = nt
        
        fd = self._fd_oper.to(device=device, dtype=dtype)
        return fd
    
    def compute_time_derivative(self, X: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute time derivatives with the SBP FD operator (same method as SINDy).
        
        Args:
            X: (L, nx) time-series data.
            dt: Time interval.
        
        Returns:
            dX_dt: (L, nx) time derivative.
        """
        L = X.size(0)
        if L < 2:
            return torch.zeros_like(X)
        
        fd = self._get_fd_operator(L, X.device, X.dtype)
        
        if fd.is_sparse:
            dX = torch.sparse.mm(fd, X) / float(dt)
        else:
            dX = (fd @ X) / float(dt)
        
        return dX
    
    @staticmethod
    def central_diff(X: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute time derivatives with central differences (legacy/reference only).
        
        Args:
            X: (L, nx) time-series data.
            dt: Time interval.
        
        Returns:
            dX_dt: (L-2, nx) time derivative (excluding first/last points).
        """
        # Central difference: (X[i+1] - X[i-1]) / (2*dt)
        dX_dt = (X[2:] - X[:-2]) / (2.0 * dt)
        return dX_dt
    
    def _get_dt(self, dt: float) -> float:
        """Use dt_eff if set; otherwise use the dt argument."""
        return self.dt_eff if self.dt_eff is not None else dt
    
    def compute_rate_equation_loss_W(self, pred_W: torch.Tensor, 
                                      truth_W: torch.Tensor,
                                      dt: float) -> torch.Tensor:
        """
        W-space rate equation loss: ||dW/dt_pred - dW/dt_truth||²
        
        Args:
            pred_W: (L, nx) predicted W
            truth_W: (L, nx) truth W
            dt: Time step (ignored if dt_eff is set).
        
        Returns:
            loss: scalar MSE
        """
        if pred_W.size(0) < 2:
            return torch.tensor(0.0, device=pred_W.device, dtype=pred_W.dtype)
        
        dt_use = self._get_dt(dt)
        dW_truth = self.compute_time_derivative(truth_W, dt_use)
        dW_pred = self.compute_time_derivative(pred_W, dt_use)
        
        return torch.mean((dW_pred - dW_truth) ** 2)
    
    def compute_rate_equation_loss_N(self, pred_frac: torch.Tensor,
                                      truth_frac: torch.Tensor,
                                      nA: torch.Tensor,
                                      dt: float) -> torch.Tensor:
        """
        N-space rate equation loss: ||dN/dt_pred - dN/dt_truth||²
        
        Args:
            pred_frac: (L, nx) predicted fraction
            truth_frac: (L, nx) truth fraction  
            nA: (L, 1) or (L,) total population per timestep
            dt: Time step (ignored if dt_eff is set).
        
        Returns:
            loss: scalar MSE
        """
        if pred_frac.size(0) < 2:
            return torch.tensor(0.0, device=pred_frac.device, dtype=pred_frac.dtype)
        
        # Match nA shape
        if nA.dim() == 1:
            nA = nA.unsqueeze(1)
        
        # N = fraction * nA
        N_truth = truth_frac * nA
        N_pred = pred_frac * nA
        
        dt_use = self._get_dt(dt)
        dN_truth = self.compute_time_derivative(N_truth, dt_use)
        dN_pred = self.compute_time_derivative(N_pred, dt_use)
        
        return torch.mean((dN_pred - dN_truth) ** 2)
    
    def compute_rate_equation_loss_CSD(self, pred_frac: torch.Tensor,
                                        truth_frac: torch.Tensor,
                                        dt: float) -> torch.Tensor:
        """
        CSD rate equation loss: ||dCSD/dt_pred - dCSD/dt_truth||²
        
        Args:
            pred_frac: (L, nx) predicted fraction
            truth_frac: (L, nx) truth fraction
            dt: Time step (ignored if dt_eff is set).
        
        Returns:
            loss: scalar MSE
        """
        if not self.ap.ion_available:
            return torch.tensor(0.0, device=pred_frac.device, dtype=pred_frac.dtype)
        
        if pred_frac.size(0) < 2:
            return torch.tensor(0.0, device=pred_frac.device, dtype=pred_frac.dtype)
        
        # Compute CSD
        CSD_truth = self.ap.compute_csd_torch(truth_frac)
        CSD_pred = self.ap.compute_csd_torch(pred_frac)
        
        dt_use = self._get_dt(dt)
        dCSD_truth = self.compute_time_derivative(CSD_truth, dt_use)
        dCSD_pred = self.compute_time_derivative(CSD_pred, dt_use)
        
        return torch.mean((dCSD_pred - dCSD_truth) ** 2)
    
    def compute_rate_equation_loss_Zbar(self, pred_frac: torch.Tensor,
                                         truth_frac: torch.Tensor,
                                         dt: float) -> torch.Tensor:
        """
        Zbar rate equation loss: ||dZbar/dt_pred - dZbar/dt_truth||²
        
        Args:
            pred_frac: (L, nx) predicted fraction
            truth_frac: (L, nx) truth fraction
            dt: Time step (ignored if dt_eff is set).
        
        Returns:
            loss: scalar MSE
        """
        if not self.ap.ion_available:
            return torch.tensor(0.0, device=pred_frac.device, dtype=pred_frac.dtype)
        
        if pred_frac.size(0) < 2:
            return torch.tensor(0.0, device=pred_frac.device, dtype=pred_frac.dtype)
        
        # Compute Zbar
        Zbar_truth = self.ap.compute_zbar_torch(truth_frac)
        Zbar_pred = self.ap.compute_zbar_torch(pred_frac)
        
        # (L,) -> (L, 1) for time derivative
        Zbar_truth = Zbar_truth.unsqueeze(1)
        Zbar_pred = Zbar_pred.unsqueeze(1)
        
        dt_use = self._get_dt(dt)
        dZbar_truth = self.compute_time_derivative(Zbar_truth, dt_use)
        dZbar_pred = self.compute_time_derivative(Zbar_pred, dt_use)
        
        return torch.mean((dZbar_pred - dZbar_truth) ** 2)
    
    def compute_fraction_loss(self, pred_frac: torch.Tensor, 
                               truth_frac: torch.Tensor) -> torch.Tensor:
        """
        Fraction reconstruction loss using only values above tau.
        
        Args:
            pred_frac: (N, nx) predicted fractions.
            truth_frac: (N, nx) true fractions.
        
        Returns:
            loss: scalar
        """
        # Flatten if needed
        if pred_frac.dim() > 2:
            pred_frac = pred_frac.view(pred_frac.size(0), -1)
        if truth_frac.dim() > 2:
            truth_frac = truth_frac.view(truth_frac.size(0), -1)
        
        mask = (truth_frac >= self.tau).to(truth_frac.dtype)
        diff = (pred_frac - truth_frac) * mask
        se = diff.pow(2)
        valid = mask.sum(dim=1)
        loss = (se.sum(dim=1) / valid.clamp_min(1)).mean()
        return loss
    
    def compute_ion_loss(self, pred_frac: torch.Tensor, 
                          truth_frac: torch.Tensor) -> torch.Tensor:
        """
        Ion distribution loss based on CSD.
        
        Args:
            pred_frac: (N, nx) predicted fractions.
            truth_frac: (N, nx) true fractions.
        
        Returns:
            loss: scalar MSE
        """
        if not self.ap.ion_available:
            return torch.tensor(0.0, device=pred_frac.device, dtype=pred_frac.dtype)
        
        pred_csd = self.ap.compute_csd_torch(pred_frac)
        truth_csd = self.ap.compute_csd_torch(truth_frac)
        return torch.mean((pred_csd - truth_csd) ** 2)
    
    def compute_zbar_loss(self, pred_frac: torch.Tensor, 
                           truth_frac: torch.Tensor) -> torch.Tensor:
        """
        Mean charge loss.
        
        Args:
            pred_frac: (N, nx) predicted fractions.
            truth_frac: (N, nx) true fractions.
        
        Returns:
            loss: scalar MSE
        """
        if not self.ap.ion_available:
            return torch.tensor(0.0, device=pred_frac.device, dtype=pred_frac.dtype)
        
        pred_zbar = self.ap.compute_zbar_torch(pred_frac)
        truth_zbar = self.ap.compute_zbar_torch(truth_frac)
        return torch.mean((pred_zbar - truth_zbar) ** 2)
    
    def compute_all(self, pred_frac: torch.Tensor, truth_frac: torch.Tensor,
                    pred_sum: torch.Tensor, truth_sum: torch.Tensor) -> dict:
        """
        Compute all physics losses.
        
        Returns:
            dict with keys: 'frac', 'ion', 'zbar', 'fse'
        """
        return {
            'frac': self.compute_fraction_loss(pred_frac, truth_frac),
            'ion': self.compute_ion_loss(pred_frac, truth_frac),
            'zbar': self.compute_zbar_loss(pred_frac, truth_frac),
            'fse': torch.mean((pred_sum - truth_sum) ** 2)
        }
