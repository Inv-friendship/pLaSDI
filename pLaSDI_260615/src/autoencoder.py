# -*- coding: utf-8 -*-
"""
Autoencoder Module
==================
Autoencoder for LaSDIc with external library dependencies removed.

Autoencoder that compresses population data into latent space.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any

from .common import get_activation



class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron with configurable activation and reshape
    """
    
    def __init__(self, layer_sizes: List[int],
                 act_type: str = 'sigmoid',
                 reshape_index: Optional[int] = None,
                 reshape_shape: Optional[List[int]] = None,
                 threshold: float = 0.1,
                 value: float = 0.0):
        super(MultiLayerPerceptron, self).__init__()
        
        # Layer configuration
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        
        # Linear layers
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[k], layer_sizes[k + 1])
            for k in range(self.n_layers - 1)
        ])
        self._init_weights()
        
        # Reshape configuration
        assert reshape_index is None or reshape_index in [0, -1]
        if reshape_shape is not None:
            assert np.prod(reshape_shape) == layer_sizes[reshape_index]
        self.reshape_index = reshape_index
        self.reshape_shape = reshape_shape
        
        # Activation function
        self.act_type = act_type
        if act_type == "threshold":
            self.act = nn.Threshold(threshold, value)
        else:
            self.act = get_activation(act_type)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input reshape (for encoder)
        if self.reshape_index == 0 and self.reshape_shape is not None:
            assert list(x.shape[-len(self.reshape_shape):]) == self.reshape_shape
            x = x.view(list(x.shape[:-len(self.reshape_shape)]) + [self.layer_sizes[0]])
        
        # Hidden layers with activation
        for i in range(self.n_layers - 2):
            x = self.fcs[i](x)
            x = self.act(x)
        
        # Output layer (no activation)
        x = self.fcs[-1](x)
        
        # Output sigmoid for decoder (ensure output in [0, 1])
        if self.reshape_index == -1:
            x = torch.sigmoid(x)
        
        # Output reshape (for decoder)
        if self.reshape_index == -1 and self.reshape_shape is not None:
            x = x.view(list(x.shape[:-1]) + self.reshape_shape)
        
        return x
    
    def _init_weights(self):
        """Xavier uniform initialization"""
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)


class Autoencoder(nn.Module):
    """
    Autoencoder for population data compression
    
    Input: (B, 1, 1, nx) population data in W-space
    Latent: (B, nz) latent representation
    Output: (B, 1, 1, nx) reconstructed population
    """
    
    def __init__(self, nx: int, latent_dim: int,
                 hidden_units: List[int],
                 activation: str = 'sigmoid'):
        """
        Args:
            nx: Input dimension (number of states)
            latent_dim: Latent space dimension
            hidden_units: List of hidden layer sizes
            activation: Activation function name
        """
        super(Autoencoder, self).__init__()
        
        self.nx = nx
        self.n_z = latent_dim
        self.qgrid_size = [1, 1, nx]  # (qdim=1, 1, nx) for compatibility
        self.space_dim = nx
        
        # Layer sizes: input -> hidden -> latent
        layer_sizes = [self.space_dim] + hidden_units + [latent_dim]
        
        # Encoder: (B, 1, 1, nx) -> (B, nz)
        self.encoder = MultiLayerPerceptron(
            layer_sizes,
            act_type=activation,
            reshape_index=0,
            reshape_shape=self.qgrid_size
        )
        
        # Decoder: (B, nz) -> (B, 1, 1, nx)
        self.decoder = MultiLayerPerceptron(
            layer_sizes[::-1],
            act_type=activation,
            reshape_index=-1,
            reshape_shape=self.qgrid_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode
        
        Args:
            x: (B, 1, 1, nx) input
        
        Returns:
            (B, 1, 1, nx) reconstruction
        """
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def export(self) -> Dict[str, Any]:
        """Export model state"""
        return {
            'autoencoder_param': self.cpu().state_dict(),
            'nx': self.nx,
            'n_z': self.n_z,
        }
    
    def load(self, dict_: Dict[str, Any]):
        """Load model state"""
        self.load_state_dict(dict_['autoencoder_param'])


def create_autoencoder(nx: int, latent_dim: int,
                       hidden_units: List[int],
                       activation: str = 'sigmoid',
                       device: torch.device = None,
                       dtype: torch.dtype = torch.float64) -> Autoencoder:
    """
    Factory function to create Autoencoder
    
    Args:
        nx: Input dimension
        latent_dim: Latent space dimension
        hidden_units: Hidden layer sizes
        activation: Activation function
        device: Target device
        dtype: Data type
    
    Returns:
        Configured Autoencoder
    """
    ae = Autoencoder(nx, latent_dim, hidden_units, activation)
    
    if device is not None:
        ae = ae.to(device)
    if dtype is not None:
        ae = ae.to(dtype)
    
    return ae
