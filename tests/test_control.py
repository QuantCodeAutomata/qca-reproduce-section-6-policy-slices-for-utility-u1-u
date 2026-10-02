"""
Tests for src/control.py — Kbar evaluation and z-optimization.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grid import Grid
from src.control import compute_sup_Kbar, compute_Kbar_at_z, post_trade_states
from src.utils import Ubar


@pytest.fixture
def small_grid():
    """Minimal 5×5 grid for control tests."""
    return Grid(m_max=3.0, x_max=3.0, N_m=10, N_x=10, N_z=20,
                eps_buy=0.05, eps_sell=0.05)


def test_post_trade_z0(small_grid):
    """z=0 -> full sell: m_z=0."""
    g = small_grid
    mz, xz = post_trade_states(
        np.array([1.0]), np.array([1.0]),
        np.array([0.0]),
        g.eps_buy, g.eps_sell
    )
    assert abs(mz[0]) < 1e-12
    # x_z = x + (1-eps_sell)*m = 1 + 0.95*1 = 1.95
    assert abs(xz[0] - (1.0 + 0.95 * 1.0)) < 1e-10


def test_post_trade_z1(small_grid):
    """z=1 -> maximum buy: m_z = x/(1+eps_buy) + m."""
    g = small_grid
    m, x = 1.0, 1.0
    mz, xz = post_trade_states(
        np.array([m]), np.array([x]),
        np.array([1.0]),
        g.eps_buy, g.eps_sell
    )
    expected_mz = x / (1 + g.eps_buy) + m
    assert abs(mz[0] - expected_mz) < 1e-10
    # x_z = x - (1+eps_buy)*(mz - m) = x - (1+eps_buy)*(x/(1+eps_buy)) = 0
    assert abs(xz[0]) < 1e-9


def test_post_trade_no_trade(small_grid):
    """At z* such that m_z=m, no cost is paid: x_z=x."""
    g = small_grid
    # The z value that keeps m_z = m is z = m / (x/(1+eps_buy) + m)
    m, x = 1.0, 1.0
    z_notrade = m / (x / (1 + g.eps_buy) + m)
    mz, xz = post_trade_states(
        np.array([m]), np.array([x]),
        np.array([z_notrade]),
        g.eps_buy, g.eps_sell
    )
    assert abs(mz[0] - m) < 1e-9
    assert abs(xz[0] - x) < 1e-9


def test_Kbar_constant_nu(small_grid):
    """If nu is constant c, then Kbar = c * Ubar_ratio."""
    g = small_grid
    c = 2.5
    nu = np.full((g.N_m + 1, g.N_x + 1), c)
    # For any (i,j,z), Kbar = c * Ubar(w_z) / Ubar(w)
    # The sup over z is c * max_z(Ubar(w_z)/Ubar(w))
    sup_val, _, _ = compute_sup_Kbar(nu, g)
    # Must equal c * max_z Ubar_ratio
    expected_max = c * g.Ubar_ratio.max(axis=2)
    np.testing.assert_allclose(sup_val, expected_max, rtol=1e-8)


def test_sup_Kbar_shape(small_grid):
    """sup_Kbar should return arrays of shape (N_m+1, N_x+1)."""
    g = small_grid
    nu = np.random.rand(g.N_m + 1, g.N_x + 1)
    sup_val, z_hat, kbar_all = compute_sup_Kbar(nu, g)
    assert sup_val.shape == (g.N_m + 1, g.N_x + 1)
    assert z_hat.shape == (g.N_m + 1, g.N_x + 1)
    assert kbar_all.shape == (g.N_m + 1, g.N_x + 1, g.N_z)


def test_sup_Kbar_nonneg(small_grid):
    """Kbar values should be non-negative for non-negative nu."""
    g = small_grid
    nu = np.abs(np.random.rand(g.N_m + 1, g.N_x + 1))
    sup_val, _, kbar_all = compute_sup_Kbar(nu, g)
    assert np.all(kbar_all >= -1e-12)


def test_z_hat_in_range(small_grid):
    """z_hat must be in [0, 1]."""
    g = small_grid
    nu = np.random.rand(g.N_m + 1, g.N_x + 1)
    _, z_hat, _ = compute_sup_Kbar(nu, g)
    assert np.all(z_hat >= -1e-12)
    assert np.all(z_hat <= 1 + 1e-12)


def test_Kbar_at_specific_z(small_grid):
    """compute_Kbar_at_z should match manual computation."""
    g = small_grid
    i, j, k = 5, 5, 10  # grid indices
    m = g.m_grid[i]
    x = g.x_grid[j]
    z = g.z_grid[k]
    mz = g.mz_table[i, j, k]
    xz = g.xz_table[i, j, k]
    w = m + x
    w_z = mz + xz

    nu = np.ones((g.N_m + 1, g.N_x + 1)) * 3.0
    val = compute_Kbar_at_z(nu, g, i, j, k)
    expected = 3.0 * Ubar(w_z) / Ubar(w)
    assert abs(val - expected) < 1e-8
