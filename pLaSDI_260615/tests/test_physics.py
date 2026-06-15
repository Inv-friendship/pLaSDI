# -*- coding: utf-8 -*-
"""Tests for atomic_physics.py and fd.py"""
import numpy as np
import torch
import pytest

from src.atomic_physics import AtomicPhysics, PhysicsLoss
from src.fd import FDdict
from src.common import FDOperatorCache, compute_time_derivative


class TestAtomicPhysics:
    @pytest.fixture
    def ap(self):
        # ab→charge 0, cd→charge 1, 02+→charge 2
        return AtomicPhysics(["ab_0_0", "cd_0_0", "02+"], nx=3, dtype=torch.float64)
    
    def test_ion_available(self, ap):
        assert ap.ion_available and ap.Z0 == 2 and ap.nq == 3
    
    def test_csd_sum_one(self, ap):
        csd = ap.compute_csd_numpy(np.array([[0.3, 0.5, 0.2]]))
        np.testing.assert_allclose(csd.sum(axis=1), 1.0, atol=1e-12)
    
    def test_zbar_range(self, ap):
        frac = np.random.dirichlet([1, 1, 1], size=100)
        zbar = ap.compute_zbar_numpy(frac)
        assert np.all(zbar >= -1e-10) and np.all(zbar <= ap.Z0 + 1e-10)
    
    def test_zbar_extreme(self, ap):
        np.testing.assert_allclose(ap.compute_zbar_numpy(np.array([[1, 0, 0]])), 0.0, atol=1e-12)
        np.testing.assert_allclose(ap.compute_zbar_numpy(np.array([[0, 0, 1]])), ap.Z0, atol=1e-12)


class TestFDOperator:
    @pytest.mark.parametrize("fd_type", ["sbp12", "sbp24", "sbp36", "sbp48"])
    def test_linear_derivative(self, fd_type):
        L, dt = 100, 0.01
        t = np.arange(L) * dt
        Z = torch.tensor((2.0 * t + 1.0).reshape(-1, 1), dtype=torch.float64)
        dZ = compute_time_derivative(Z, dt, FDOperatorCache(fd_type))
        np.testing.assert_allclose(dZ[10:-10, 0].numpy(), 2.0, atol=1e-8)
    
    def test_batch(self):
        Z = torch.randn(3, 50, 2, dtype=torch.float64)
        dZ = compute_time_derivative(Z, 0.01, FDOperatorCache('sbp12'))
        assert dZ.shape == Z.shape and torch.isfinite(dZ).all()
    
    def test_fd_dtype_float64(self):
        D, _, _ = FDdict['sbp12'].getOperators(50)
        assert D.dtype == torch.float64


class TestPhysicsLoss:
    def test_fraction_loss_zero(self):
        ap = AtomicPhysics(["ab_0_0", "cd_0_0", "02+"], nx=3, dtype=torch.float64)
        pl = PhysicsLoss(ap, tau=1e-100)
        f = torch.tensor([[0.3, 0.5, 0.2]], dtype=torch.float64)
        assert pl.compute_fraction_loss(f, f).item() < 1e-15
    
    def test_ion_loss_zero(self):
        ap = AtomicPhysics(["ab_0_0", "cd_0_0", "02+"], nx=3, dtype=torch.float64)
        pl = PhysicsLoss(ap, tau=1e-100)
        f = torch.tensor([[0.3, 0.5, 0.2]], dtype=torch.float64)
        assert pl.compute_ion_loss(f, f).item() < 1e-15
