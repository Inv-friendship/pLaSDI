# -*- coding: utf-8 -*-
"""Tests for src/scaling.py"""
import numpy as np
import torch
import pytest

from src.scaling import PopulationScaler, ControlScaler, TorchScaleHelper


class TestPopulationScaler:
    def _make_pop(self, n=50, nx=100, seed=42):
        rng = np.random.default_rng(seed)
        pop = rng.exponential(1.0, size=(n, nx))
        pop[:, -10:] = 1e-40
        return pop
    
    def test_fit_transform_shape(self):
        pop = self._make_pop()
        s = PopulationScaler(eps=1e-50, normalize=True)
        W = s.fit_transform(pop, axis=1)
        assert W.shape == pop.shape
    
    def test_W_range_01(self):
        pop = self._make_pop()
        s = PopulationScaler(eps=1e-50, normalize=True)
        W = s.fit_transform(pop, axis=1)
        assert np.all(W >= -0.01) and np.all(W <= 1.01)
    
    def test_roundtrip_fraction(self):
        pop = self._make_pop(nx=50)
        s = PopulationScaler(eps=1e-50, normalize=True)
        W = s.fit_transform(pop, axis=1)
        rec = s.inverse_transform(W)
        nA = np.sum(pop, axis=1, keepdims=True)
        f_orig = pop / (nA + 1e-50)
        f_rec = rec / (np.sum(rec, axis=1, keepdims=True) + 1e-50)
        mask = f_orig > 1e-10
        if mask.any():
            rel_err = np.abs(f_orig[mask] - f_rec[mask]) / (f_orig[mask] + 1e-30)
            assert np.mean(rel_err) < 0.01
    
    def test_fraction_sum_one(self):
        pop = self._make_pop(nx=50)
        s = PopulationScaler(eps=1e-50, normalize=True)
        W = s.fit_transform(pop, axis=1)
        rec = s.inverse_transform(W)
        f_rec = rec / (np.sum(rec, axis=1, keepdims=True) + 1e-300)
        np.testing.assert_allclose(np.sum(f_rec, axis=1), 1.0, atol=1e-10)


class TestTorchScaleHelper:
    def _setup(self):
        pop = np.random.exponential(1.0, size=(20, 50))
        s = PopulationScaler(eps=1e-50, normalize=True)
        W = s.fit_transform(pop, axis=1)
        h = TorchScaleHelper(s, dtype=torch.float64)
        return h, torch.tensor(W, dtype=torch.float64)
    
    def test_fraction_sum_one(self):
        h, W_t = self._setup()
        F = h.W_to_fraction(W_t)
        torch.testing.assert_close(F.sum(dim=-1), torch.ones(F.size(0), dtype=torch.float64), atol=1e-10, rtol=0)
    
    def test_non_negative(self):
        h, W_t = self._setup()
        assert (h.W_to_fraction(W_t) >= 0).all()
    
    def test_cache_consistency(self):
        h, W_t = self._setup()
        torch.testing.assert_close(h.W_to_fraction(W_t), h.W_to_fraction(W_t))


class TestControlScaler:
    def test_shape(self):
        U = np.array([[1e4, 1e20], [2e4, 5e20]])
        s = ControlScaler(eps=1e-300)
        assert s.fit_transform(U).shape == U.shape
    
    def test_range(self):
        U = np.random.uniform(1e3, 1e5, size=(100, 2))
        s = ControlScaler(eps=1e-300)
        U_s = s.fit_transform(U)
        assert U_s.min() >= -0.1 and U_s.max() <= 1.1
