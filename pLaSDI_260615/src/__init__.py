# -*- coding: utf-8 -*-
"""
LaSDIc Source Package
=====================
Latent-space SINDy with Control for atomic population dynamics
"""

from .autoencoder import Autoencoder, create_autoencoder
from .sindyc import SINDyC, split_coefs
from .sindy_utils import split_coefs_torch  # Use the unified sindy_utils version only (includes A transpose)
from .sindyc_adaptive import AdaptiveSINDyC
from .fd import FDdict
from .atomic_physics import AtomicPhysics, PhysicsLoss
from .scaling import PopulationScaler, ControlScaler, TorchScaleHelper
from .common import ACT_DICT, get_activation, FDOperatorCache, compute_time_derivative

# Note: LaSDIcTrainer is not imported here to avoid circular dependency
# Import it directly: from src.trainer import LaSDIcTrainer

__all__ = [
    # Models
    'Autoencoder',
    'create_autoencoder',
    'SINDyC',
    'AdaptiveSINDyC',
    'split_coefs',
    'split_coefs_torch',
    
    # Physics
    'AtomicPhysics',
    'PhysicsLoss',
    'FDdict',
    
    # Scaling
    'PopulationScaler',
    'ControlScaler',
    'TorchScaleHelper',
    
    # Common
    'ACT_DICT',
    'get_activation',
    'FDOperatorCache',
    'compute_time_derivative',
]
