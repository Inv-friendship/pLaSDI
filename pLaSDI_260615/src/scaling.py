# -*- coding: utf-8 -*-
"""
Scaling Module
==============
Data scaling utilities (log-space transform, min-max scaling, etc.).

Future extensions:
- Separate scalers can be applied to each U component (T, density).
- Independent scaling strategies can be used for populations and control variables.
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ScaleParams:
    """Scale transform parameters."""
    eps: float
    W_min: float
    W_max: float
    nA: np.ndarray  # normalization factor
    axis: int
    normalize: bool
    
    def to_dict(self) -> dict:
        return {
            'eps': self.eps,
            'W_min': self.W_min,
            'W_max': self.W_max,
            'nA': self.nA,
            'axis': self.axis,
            'normalize': self.normalize
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'ScaleParams':
        return ScaleParams(**d)


class PopulationScaler:
    """
    Population data scaler.
    
    X (original space) <-> W (scaled space)
    
    Transform: W = (1 - log(X + eps) - W_min) / (W_max - W_min)
    Inverse transform: X = exp(1 - W_tilde) - eps, where W_tilde = W * (W_max - W_min) + W_min
    """
    
    def __init__(self, eps: float = 1e-50, normalize: bool = True):
        """
        Args:
            eps: Minimum epsilon value.
            normalize: Whether to normalize populations so the sum is 1.
        """
        self.eps = eps
        self.normalize = normalize
        self.params: Optional[ScaleParams] = None
        self._fitted = False
    
    def fit_transform(self, N: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Fit the scaler and transform the data.
        
        Args:
            N: (N_samples, nx) population data.
            axis: Normalization axis.
        
        Returns:
            W: Transformed data in [0, 1].
        """
        nA = np.sum(N, axis=axis, keepdims=True)
        X = N / (nA + self.eps) if self.normalize else N
        W_tilde = 1.0 - np.log(X + self.eps)

        if self.normalize:
            # Fixed physical range: X=1 -> W=0, X=1e-100 -> W≈1
            X_min, X_max = 1e-50, 1.0
            W_min = 1.0 - np.log(X_max + self.eps)
            W_max = 1.0 - np.log(X_min + self.eps)
        else:
            W_min, W_max = float(np.nanmin(W_tilde)), float(np.nanmax(W_tilde))

        W = (W_tilde - W_min) / (W_max - W_min + self.eps)
        
        self.params = ScaleParams(
            eps=self.eps,
            W_min=W_min,
            W_max=W_max,
            nA=nA.copy(),
            axis=axis,
            normalize=self.normalize
        )
        self._fitted = True
        
        return W
    
    def transform(self, N: np.ndarray) -> np.ndarray:
        """Transform using already fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")
        
        nA = np.sum(N, axis=self.params.axis, keepdims=True)
        X = N / (nA + self.eps) if self.normalize else N
        W_tilde = 1.0 - np.log(X + self.eps)
        W = (W_tilde - self.params.W_min) / (self.params.W_max - self.params.W_min + self.eps)
        return W
    
    def inverse_transform(self, W: np.ndarray) -> np.ndarray:
        """Inverse transform (W -> X)."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")
        
        p = self.params
        W_tilde = W * (p.W_max - p.W_min) + p.W_min
        X_tilde = np.exp(1.0 - W_tilde) - p.eps
        X_tilde = np.clip(X_tilde, 0.0, None)
        # Normalize so the sum is 1
        X_tilde = X_tilde / np.sum(X_tilde, axis=1, keepdims=True)
        return (X_tilde * p.nA) if p.normalize else X_tilde
    
    def get_params(self) -> ScaleParams:
        """Return parameters."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted.")
        return self.params

    def save_state(self) -> dict:
        """Return a serializable state for checkpoint saving."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted.")
        return {
            'eps': self.eps,
            'normalize': self.normalize,
            'params': self.params.to_dict(),
        }

    @classmethod
    def from_state(cls, state: dict) -> 'PopulationScaler':
        """Restore from a saved state."""
        scaler = cls(eps=state['eps'], normalize=state['normalize'])
        scaler.params = ScaleParams.from_dict(state['params'])
        scaler._fitted = True
        return scaler


class ControlScaler:
    """
    Control variable scaler (U: Temperature, Density).
    
    Scaling is applied independently to each column.
    
    Future extensions:
    - Different scaling strategies can be applied to T and density.
    - For example, T can use log-scale and density can use linear scaling.
    """
    
    def __init__(self, eps: float = 1e-300):
        self.eps = eps
        self.col_params: list = []
        self._fitted = False
    
    def fit_transform(self, U: np.ndarray) -> np.ndarray:
        """
        Scale each column independently.
        
        Args:
            U: (N, mu) control variables.
        
        Returns:
            U_scaled: (N, mu) scaled control variables.
        """
        mu = U.shape[1]
        U_scaled_cols = []
        self.col_params = []
        
        for j in range(mu):
            col = U[:, j:j+1]
            W_tilde = 1.0 - np.log(col + self.eps)
            W_min, W_max = float(np.nanmin(W_tilde)), float(np.nanmax(W_tilde))
            W = (W_tilde - W_min) / (W_max - W_min + self.eps)
            
            self.col_params.append({
                'eps': self.eps,
                'W_min': W_min,
                'W_max': W_max
            })
            U_scaled_cols.append(W)
        
        self._fitted = True
        return np.hstack(U_scaled_cols).astype(np.float64)
    
    def transform(self, U: np.ndarray) -> np.ndarray:
        """Transform using already fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted.")
        
        mu = U.shape[1]
        U_scaled_cols = []
        
        for j in range(mu):
            col = U[:, j:j+1]
            p = self.col_params[j]
            W_tilde = 1.0 - np.log(col + p['eps'])
            W = (W_tilde - p['W_min']) / (p['W_max'] - p['W_min'] + p['eps'])
            U_scaled_cols.append(W)
        
        return np.hstack(U_scaled_cols).astype(np.float64)
    
    def inverse_transform_single(self, W_val: float, col_idx: int) -> float:
        """Inverse-transform a single value."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted.")
        
        p = self.col_params[col_idx]
        W_tilde = W_val * (p['W_max'] - p['W_min']) + p['W_min']
        return np.exp(1.0 - W_tilde) - p['eps']

    def save_state(self) -> dict:
        """Return a serializable state for checkpoint saving."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted.")
        return {
            'eps': self.eps,
            'col_params': self.col_params,
        }

    @classmethod
    def from_state(cls, state: dict) -> 'ControlScaler':
        """Restore from a saved state."""
        scaler = cls(eps=state['eps'])
        scaler.col_params = state['col_params']
        scaler._fitted = True
        return scaler


# =============================================================================
# PyTorch transform functions
# =============================================================================

def W_to_fraction_torch(W_tensor: torch.Tensor, 
                        W_min: float, W_max: float, eps: float,
                        dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Convert from W-space to fractions (PyTorch, autograd-compatible).
    
    Args:
        W_tensor: (N, ..., nx) W-space data.
        W_min, W_max, eps: Scale parameters.
        dtype: Data type.
    
    Returns:
        fraction: (N, ..., nx) normalized fractions (sum=1).
    """
    device = W_tensor.device
    W_min_t = torch.tensor(W_min, dtype=dtype, device=device)
    W_max_t = torch.tensor(W_max, dtype=dtype, device=device)
    eps_t = torch.tensor(eps, dtype=dtype, device=device)
    
    Wt = W_tensor * (W_max_t - W_min_t) + W_min_t
    Ft = torch.exp(torch.tensor(1.0, dtype=dtype, device=device) - Wt) - eps_t
    Ft = torch.clamp(Ft, min=0.0)
    # Normalize so the sum is 1
    Ft = Ft / Ft.sum(dim=-1, keepdim=True)
    return Ft


def sum_before_normalize_torch(W_tensor: torch.Tensor,
                                W_min: float, W_max: float, eps: float,
                                dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Compute the pre-normalization sum for FSE loss.
    
    Returns:
        sum: (N, ..., 1) pre-normalization sum.
    """
    device = W_tensor.device
    W_min_t = torch.tensor(W_min, dtype=dtype, device=device)
    W_max_t = torch.tensor(W_max, dtype=dtype, device=device)
    eps_t = torch.tensor(eps, dtype=dtype, device=device)
    
    Wt = W_tensor * (W_max_t - W_min_t) + W_min_t
    Ft = torch.exp(torch.tensor(1.0, dtype=dtype, device=device) - Wt) - eps_t
    Ft = torch.clamp(Ft, min=0.0)
    return Ft.sum(dim=-1, keepdim=True)


class TorchScaleHelper:
    """PyTorch transform helper with cached scalar tensors."""
    
    def __init__(self, pop_scaler: PopulationScaler, dtype: torch.dtype = torch.float64):
        self.params = pop_scaler.get_params()
        self.dtype = dtype
        
        # Cache scalar tensors to avoid recreating them for every batch
        self._cached_device = None
        self._W_min_t = None
        self._W_max_t = None
        self._eps_t = None
        self._one_t = None
    
    def _ensure_cache(self, device):
        """Create or move cached tensors as needed."""
        if self._cached_device != device:
            p = self.params
            self._W_min_t = torch.tensor(p.W_min, dtype=self.dtype, device=device)
            self._W_max_t = torch.tensor(p.W_max, dtype=self.dtype, device=device)
            self._eps_t = torch.tensor(p.eps, dtype=self.dtype, device=device)
            self._one_t = torch.tensor(1.0, dtype=self.dtype, device=device)
            self._cached_device = device
    
    def W_to_fraction(self, W: torch.Tensor) -> torch.Tensor:
        """Convert W -> Fraction using cached scalars."""
        self._ensure_cache(W.device)
        Wt = W * (self._W_max_t - self._W_min_t) + self._W_min_t
        Ft = torch.exp(self._one_t - Wt) - self._eps_t
        Ft = torch.clamp(Ft, min=0.0)
        Ft = Ft / Ft.sum(dim=-1, keepdim=True)
        return Ft
    
    def get_sum_before_normalize(self, W: torch.Tensor) -> torch.Tensor:
        """Pre-normalization sum using cached scalars."""
        self._ensure_cache(W.device)
        Wt = W * (self._W_max_t - self._W_min_t) + self._W_min_t
        Ft = torch.exp(self._one_t - Wt) - self._eps_t
        Ft = torch.clamp(Ft, min=0.0)
        return Ft.sum(dim=-1, keepdim=True)
