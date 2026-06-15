# -*- coding: utf-8 -*-
"""
SINDyC Module
=============
Sparse Identification of Nonlinear Dynamics with Control

Linear SINDyC model:
    dZ/dt = a + A·Z + B·U

Where:
    Z: (L, nz) or (B, L, nz) latent state
    U: (L, mu) or (B, L, mu) control input
    a: (nz,) bias vector
    A: (nz, nz) state matrix
    B: (nz, mu) control matrix
"""

import numpy as np
import torch
from scipy.integrate import odeint
from typing import Optional, Tuple, Union, Dict, Any

from .fd import FDdict


class SINDyC:
    """
    Linear SINDyC dynamics model
    
    Learns coefficients for: dZ/dt = a + A·Z + B·U
    """
    
    def __init__(self, dim: int, nt: int,
                 fd_type: str = 'sbp12',
                 coef_norm_order: int = 2,
                 use_global_coefs: bool = False):
        """
        Args:
            dim: Latent dimension (nz)
            nt: Number of time steps
            fd_type: Finite difference type ('sbp12', 'sbp24', 'sbp36', 'sbp48')
            coef_norm_order: Norm order for coefficient regularization (1 or 2)
            use_global_coefs: If True, compute single global coefficients for all batches
        """
        self.dim = dim
        self.nt = nt
        self.fd_type = fd_type
        self.coef_norm_order = coef_norm_order
        self.use_global_coefs = use_global_coefs
        
        # FD operator
        if fd_type not in FDdict:
            raise ValueError(f"Unknown fd_type: {fd_type}. Available: {list(FDdict.keys())}")
        self.fd = FDdict[fd_type]
        self.fd_oper, _, _ = self.fd.getOperators(nt)
        
        # Control input dimension (set on first calibrate)
        self.mu = None
        self.ncoefs = None  # = nz * (1 + nz + mu)
        
        self.MSE = torch.nn.MSELoss()
    
    def _set_mu(self, mu: int):
        """Set control input dimension"""
        if self.mu is None:
            self.mu = int(mu)
            self.ncoefs = self.dim * (1 + self.dim + self.mu)
        else:
            assert mu == self.mu, f"Inconsistent mu: got {mu}, expected {self.mu}"
    
    def _update_fd_operator(self, L: int):
        """Update FD operator for new sequence length"""
        if self.nt != L:
            self.nt = L
            self.fd_oper, _, _ = self.fd.getOperators(L)
    
    def compute_time_derivative(self, Z: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute time derivative using SBP finite difference
        
        Args:
            Z: (L, nz) or (B, L, nz) state tensor
            dt: Time step
        
        Returns:
            dZ/dt: Same shape as Z
        """
        assert dt > 0, "dt must be positive"
        
        if Z.dim() == 2:
            # Single sequence: (L, nz)
            L = Z.size(0)
            self._update_fd_operator(L)
            
            fd = self.fd_oper.to(device=Z.device, dtype=Z.dtype)
            if fd.is_sparse:
                dZ = torch.sparse.mm(fd, Z) / float(dt)
            else:
                dZ = (fd @ Z) / float(dt)
            return dZ
        
        elif Z.dim() == 3:
            # Batch: (B, L, nz)
            B, L, nz = Z.shape
            self._update_fd_operator(L)
            
            D = self.fd_oper.to(device=Z.device, dtype=Z.dtype)
            if D.is_sparse:
                D = D.to_dense()
            D = D / float(dt)  # (L, L)
            
            # Batch matrix multiply: (L, L) @ (B, L, nz) -> (B, L, nz)
            return torch.einsum('ij, bjk -> bik', D, Z)
        
        else:
            raise ValueError(f"Z must be 2D or 3D, got {Z.dim()}D")
    
    def calibrate(self, Z: torch.Tensor, U: torch.Tensor, dt: float,
                  compute_loss: bool = True,
                  numpy: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, float, float]]:
        """
        Calibrate SINDyC coefficients via least squares
        
        Args:
            Z: (L, nz) or (B, L, nz) latent states
            U: (L, mu) or (B, L, mu) control inputs
            dt: Time step
            compute_loss: Whether to compute losses
            numpy: Return numpy arrays instead of tensors
        
        Returns:
            If compute_loss:
                (coefs, loss_sindy, loss_coef)
            Else:
                coefs
        """
        # Batch path
        if Z.dim() == 3:
            return self._calibrate_batch(Z, U, dt, compute_loss, numpy)
        
        # Single sequence path
        assert Z.dim() == 2 and U.dim() == 2, "Z:(L,nz), U:(L,mu) expected"
        L, nz = Z.shape
        assert U.size(0) == L, "Z, U length mismatch"
        mu = U.size(1)
        self._set_mu(mu)
        
        # Compute dZ/dt
        dZdt = self.compute_time_derivative(Z, dt)  # (L, nz)
        
        # Build feature matrix: [1, Z, U]
        ones = torch.ones(L, 1, dtype=Z.dtype, device=Z.device)
        Phi = torch.cat([ones, Z, U], dim=1)  # (L, 1+nz+mu)
        
        # Least squares: Phi @ C = dZdt
        C = torch.linalg.lstsq(Phi, dZdt).solution  # (1+nz+mu, nz)
        
        if compute_loss:
            resid = dZdt - Phi @ C
            loss_sindy = self.MSE(resid, torch.zeros_like(resid))
            loss_coef = torch.norm(C, self.coef_norm_order)
        
        coefs = C.flatten()
        
        if numpy:
            coefs = coefs.cpu().detach().numpy()
            if compute_loss:
                return coefs, float(loss_sindy.item()), float(loss_coef.item())
            return coefs
        
        if compute_loss:
            return coefs, loss_sindy, loss_coef
        return coefs
    
    def _calibrate_batch(self, Z: torch.Tensor, U: torch.Tensor, dt: float,
                         compute_loss: bool, numpy: bool):
        """Batch calibration"""
        B, L, nz = Z.shape
        assert U.dim() == 3 and U.size(0) == B and U.size(1) == L
        mu = U.size(2)
        self._set_mu(mu)
        
        device, dtype = Z.device, Z.dtype
        
        # Compute dZ/dt
        dZ = self.compute_time_derivative(Z, dt)  # (B, L, nz)
        
        # Build feature matrix
        ones = torch.ones(B, L, 1, device=device, dtype=dtype)
        Phi = torch.cat([ones, Z, U], dim=-1)  # (B, L, 1+nz+mu)
        p = Phi.size(-1)
        
        if self.use_global_coefs:
            # Global coefficients: single C for all batches
            Phi_all = Phi.reshape(B * L, p)  # (B·L, p)
            dZ_all = dZ.reshape(B * L, nz)   # (B·L, nz)
            
            sol = torch.linalg.lstsq(Phi_all, dZ_all)
            C = sol.solution  # (p, nz)
            
            if compute_loss:
                pred = Phi @ C  # (B, L, nz) via broadcasting
                loss_sindy_per = torch.mean((pred - dZ) ** 2, dim=(1, 2))
                loss_sindy = loss_sindy_per.sum()
                loss_coef = C.abs().sum() if self.coef_norm_order == 1 else C.norm(p=self.coef_norm_order)
            
            coefs_flat = C.reshape(-1)
            coefs = coefs_flat.unsqueeze(0).expand(B, -1).contiguous()
        
        else:
            # Local coefficients: separate C for each batch
            sol = torch.linalg.lstsq(Phi, dZ)  # (B, L, p) \ (B, L, nz)
            C = sol.solution  # (B, p, nz)
            
            if compute_loss:
                pred = Phi @ C  # (B, L, nz)
                loss_sindy_per = torch.mean((pred - dZ) ** 2, dim=(1, 2))
                loss_coef_per = C.abs().sum(dim=(1, 2)) if self.coef_norm_order == 1 else C.norm(p=self.coef_norm_order, dim=(1, 2))
                loss_sindy = loss_sindy_per.sum()
                loss_coef = loss_coef_per.sum()
            
            coefs = C.reshape(B, -1).contiguous()
        
        if numpy:
            coefs_np = coefs.detach().cpu().numpy()
            if compute_loss:
                return coefs_np, float(loss_sindy.item()), float(loss_coef.item())
            return coefs_np
        
        if compute_loss:
            return coefs, loss_sindy, loss_coef
        return coefs
    
    def simulate(self, coefs: Union[np.ndarray, torch.Tensor],
                 z0: Union[np.ndarray, torch.Tensor],
                 t_grid: np.ndarray,
                 U: Union[np.ndarray, callable]) -> np.ndarray:
        """
        Simulate dynamics using scipy odeint
        
        Args:
            coefs: Flattened coefficients or (p, nz) matrix
            z0: Initial latent state
            t_grid: Time points
            U: Control input array (L, mu) or callable U(t)
        
        Returns:
            Z: (L, nz) simulated trajectory
        """
        # Convert to numpy
        if isinstance(coefs, torch.Tensor):
            cmat = coefs.detach().cpu().numpy()
        else:
            cmat = np.asarray(coefs, dtype=np.float64)
        
        nz = self.dim
        if cmat.ndim == 1:
            assert self.mu is not None, "mu unknown; call calibrate first"
            p = 1 + nz + self.mu
            cmat = cmat.reshape(p, nz)
        elif cmat.ndim == 2:
            p = cmat.shape[0]
            mu = p - 1 - nz
            if self.mu is None:
                self.mu = int(mu)
                self.ncoefs = nz * (1 + nz + self.mu)
        
        # Extract a, A, B
        a = cmat[0, :]           # (nz,)
        A = cmat[1:1+nz, :]      # (nz, nz)
        B = cmat[1+nz:, :]       # (mu, nz)
        
        # Initial condition
        z0 = z0.detach().cpu().numpy() if isinstance(z0, torch.Tensor) else np.asarray(z0, dtype=np.float64)
        t = np.asarray(t_grid, dtype=np.float64)
        
        # Control interpolation
        if callable(U):
            u_of_t = U
        else:
            U = np.asarray(U, dtype=np.float64)
            assert U.ndim == 2 and U.shape[0] == t.shape[0]
            
            def u_of_t(tnow):
                if U.shape[0] == 1:
                    return U[0]
                return np.array([np.interp(tnow, t, U[:, j]) for j in range(U.shape[1])], dtype=np.float64)
        
        # ODE right-hand side
        def rhs(z, tau):
            u = u_of_t(tau) if self.mu and self.mu > 0 else 0.0
            term_z = z @ A
            term_u = (u @ B) if isinstance(u, np.ndarray) else 0.0
            return term_z + term_u + a
        
        # Integrate
        Z = odeint(rhs, z0, t)
        return Z
    
    def export(self) -> Dict[str, Any]:
        """Export configuration"""
        return {
            'dim': self.dim,
            'nt': self.nt,
            'fd_type': self.fd_type,
            'coef_norm_order': self.coef_norm_order,
            'use_global_coefs': self.use_global_coefs,
            'mu': self.mu,
        }


def split_coefs(coefs: Union[np.ndarray, torch.Tensor],
                nz: int, mu: int) -> Tuple:
    """
    Split flattened coefficients into a, A, B
    
    Args:
        coefs: Flattened coefficients
        nz: Latent dimension
        mu: Control dimension
    
    Returns:
        (a, A, B) where a:(nz,), A:(nz,nz), B:(nz,mu)
    """
    if isinstance(coefs, torch.Tensor):
        coefs = coefs.detach().cpu().numpy()
    
    coefs = np.asarray(coefs).flatten()
    p = 1 + nz + mu
    C = coefs.reshape(p, nz)
    
    a = C[0, :]
    A = C[1:1+nz, :]
    B = C[1+nz:, :]
    
    return a, A, B


# NOTE: split_coefs_torch is defined in sindy_utils.py (including A transpose).
# It used to be defined here too, but the A-transpose convention differed and could cause bugs.
# Usage: from .sindy_utils import split_coefs_torch
