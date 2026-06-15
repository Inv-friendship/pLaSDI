# -*- coding: utf-8 -*-
"""
Common Module
=============
Constants and utilities shared across the project.

- ACT_DICT: activation function dictionary with lowercase keys.
- compute_time_derivative: shared FD-based time derivative helper.
"""

import torch
import torch.nn as nn

from .fd import FDdict


# =============================================================================
# Activation function dictionary (lowercase keys)
# =============================================================================
# This was previously defined separately in autoencoder.py and sindyc_adaptive.py.
# Different key casing could make only one side match depending on config values.
# Define it once here and import it from both places.
#
# Usage: ACT_DICT["mish"]() -> nn.Mish()

ACT_DICT = {
    'elu':          nn.ELU,
    'hardshrink':   nn.Hardshrink,
    'hardsigmoid':  nn.Hardsigmoid,
    'hardtanh':     nn.Hardtanh,
    'hardswish':    nn.Hardswish,
    'leakyrelu':    nn.LeakyReLU,
    'logsigmoid':   nn.LogSigmoid,
    'prelu':        nn.PReLU,
    'relu':         nn.ReLU,
    'relu6':        nn.ReLU6,
    'rrelu':        nn.RReLU,
    'selu':         nn.SELU,
    'celu':         nn.CELU,
    'gelu':         nn.GELU,
    'sigmoid':      nn.Sigmoid,
    'silu':         nn.SiLU,
    'mish':         nn.Mish,
    'softplus':     nn.Softplus,
    'softshrink':   nn.Softshrink,
    'tanh':         nn.Tanh,
    'tanhshrink':   nn.Tanhshrink,
}


def get_activation(name: str) -> nn.Module:
    """
    Return an activation function instance (case-insensitive).
    
    Args:
        name: Activation function name (for example, "mish", "Mish", and "MISH" all work).
    
    Returns:
        nn.Module instance.
    
    Raises:
        ValueError: Unknown activation function.
    """
    key = name.lower()
    if key not in ACT_DICT:
        raise ValueError(
            f"Unknown activation: '{name}'. "
            f"Available: {sorted(ACT_DICT.keys())}"
        )
    return ACT_DICT[key]()


# =============================================================================
# Shared FD-based time derivative helper
# =============================================================================
# Nearly identical code previously existed in SINDyC, AdaptiveSINDyC, and PhysicsLoss.
# Define it once here.

class FDOperatorCache:
    """
    FD operator cache manager.
    
    Caches FD operators by sequence length (nt) to avoid repeated construction.
    """
    
    def __init__(self, fd_type: str = 'sbp12'):
        if fd_type not in FDdict:
            raise ValueError(f"Unknown fd_type: {fd_type}. Available: {list(FDdict.keys())}")
        self.fd = FDdict[fd_type]
        self.fd_type = fd_type
        self._cache = {}  # {nt: fd_operator}
    
    def get_operator(self, nt: int, device=None, dtype=None) -> torch.Tensor:
        """
        Return the cached FD operator for length nt.
        
        Args:
            nt: Sequence length.
            device: target device
            dtype: target dtype
        
        Returns:
            FD operator tensor
        """
        if nt not in self._cache:
            fd_oper, _, _ = self.fd.getOperators(nt)
            self._cache[nt] = fd_oper
        
        fd = self._cache[nt]
        if device is not None or dtype is not None:
            fd = fd.to(device=device, dtype=dtype)
        return fd
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()


def compute_time_derivative(Z: torch.Tensor, dt: float,
                            fd_cache: FDOperatorCache) -> torch.Tensor:
    """
    Compute time derivatives with an FD operator (shared helper).
    
    Shared by SINDyC, AdaptiveSINDyC, and PhysicsLoss.
    
    Args:
        Z: (L, D) or (B, L, D) tensor.
        dt: Time interval (positive).
        fd_cache: FDOperatorCache instance.
    
    Returns:
        dZ/dt with the same shape as Z.
    """
    assert dt > 0, "dt must be positive"
    
    if Z.dim() == 2:
        # Single sequence: (L, D)
        L = Z.size(0)
        if L < 2:
            return torch.zeros_like(Z)
        
        D = fd_cache.get_operator(L, device=Z.device, dtype=Z.dtype)
        if D.is_sparse:
            return torch.sparse.mm(D, Z) / float(dt)
        return (D @ Z) / float(dt)
    
    elif Z.dim() == 3:
        # Batch: (B, L, D)
        B, L, nz = Z.shape
        if L < 2:
            return torch.zeros_like(Z)
        
        D = fd_cache.get_operator(L, device=Z.device, dtype=Z.dtype)
        if D.is_sparse:
            D = D.to_dense()
        D = D / float(dt)  # (L, L)
        
        # Batch matrix multiply
        return torch.einsum('ij, bjk -> bik', D, Z)
    
    else:
        raise ValueError(f"Z must be 2D or 3D, got {Z.dim()}D")
