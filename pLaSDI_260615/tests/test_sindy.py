# -*- coding: utf-8 -*-
"""Tests for SINDy modules"""
import numpy as np
import torch
import pytest

from src.sindy_utils import split_coefs_torch
from src.sindyc import SINDyC, split_coefs
from src.sindyc_adaptive import AdaptiveSINDyC


class TestSplitCoefs:
    def test_roundtrip_flat(self):
        nz, mu = 3, 2
        p = 1 + nz + mu
        C_orig = torch.randn(p, nz, dtype=torch.float64)
        a, A, B = split_coefs_torch(C_orig.flatten(), nz, mu)
        assert a.shape == (nz,) and A.shape == (nz, nz) and B.shape == (nz, mu)
        C_rec = torch.zeros_like(C_orig)
        C_rec[0, :] = a
        C_rec[1:1+nz, :] = A.t()
        C_rec[1+nz:, :] = B.t()
        torch.testing.assert_close(C_rec, C_orig)
    
    def test_numpy_torch_consistency(self):
        nz, mu = 3, 2
        C = np.random.randn(1 + nz + mu, nz)
        a_np, A_np, _ = split_coefs(C.flatten(), nz, mu)
        a_t, A_t, _ = split_coefs_torch(torch.tensor(C.flatten(), dtype=torch.float64), nz, mu)
        np.testing.assert_allclose(a_np, a_t.numpy(), atol=1e-12)
        np.testing.assert_allclose(A_np, A_t.numpy().T, atol=1e-12)


class TestAdaptiveSINDyCHurwitz:
    @pytest.fixture
    def model(self):
        return AdaptiveSINDyC(nz=3, mu=2, hidden_dims=[16, 16],
                              activation='mish', fd_type='sbp12',
                              eps=0.01, symmetric=True, head_gain=1.0).to(torch.float64)
    
    @pytest.fixture
    def model_asym(self):
        return AdaptiveSINDyC(nz=3, mu=2, hidden_dims=[16, 16],
                              activation='mish', fd_type='sbp12',
                              eps=0.01, symmetric=False, head_gain=1.0).to(torch.float64)
    
    def test_eigenvalues_negative_symmetric(self, model):
        U = torch.randn(10, 2, dtype=torch.float64)
        with torch.no_grad():
            for i in range(U.size(0)):
                ev = model.get_eigenvalues(U[i])
                assert ev.real.max().item() < -model.eps + 1e-10
    
    def test_eigenvalues_negative_asymmetric(self, model_asym):
        U = torch.randn(10, 2, dtype=torch.float64)
        with torch.no_grad():
            for i in range(U.size(0)):
                ev = model_asym.get_eigenvalues(U[i])
                assert ev.real.max().item() < -model_asym.eps + 1e-10
    
    def test_equilibrium_finite(self, model):
        U = torch.randn(5, 2, dtype=torch.float64)
        with torch.no_grad():
            Z_star = model.get_equilibrium_batch(U)
        assert Z_star.shape == (5, 3) and torch.isfinite(Z_star).all()


class TestSINDyCCalibrate:
    def test_calibrate_2d(self):
        nz, mu, L = 3, 2, 50
        s = SINDyC(dim=nz, nt=L, fd_type='sbp12', use_global_coefs=True)
        c, ls, lc = s.calibrate(torch.randn(L, nz, dtype=torch.float64),
                                 torch.randn(L, mu, dtype=torch.float64), dt=1e-3)
        assert c.numel() == nz * (1 + nz + mu) and ls >= 0
    
    def test_calibrate_batch(self):
        nz, mu, B, L = 3, 2, 4, 50
        s = SINDyC(dim=nz, nt=L, fd_type='sbp12', use_global_coefs=True)
        c, _, _ = s.calibrate(torch.randn(B, L, nz, dtype=torch.float64),
                               torch.randn(B, L, mu, dtype=torch.float64), dt=1e-3)
        assert c.shape[0] == B
