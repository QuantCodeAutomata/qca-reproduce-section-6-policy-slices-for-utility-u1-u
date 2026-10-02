"""
Tests for src/grid.py — Grid construction and precomputed tables.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grid import Grid, make_grid


@pytest.fixture
def small_grid():
    """A small grid suitable for fast testing."""
    return Grid(m_max=3.0, x_max=3.0, N_m=10, N_x=10, N_z=20,
                eps_buy=0.05, eps_sell=0.05)


def test_grid_shapes(small_grid):
    """Grid arrays should have correct shapes."""
    g = small_grid
    assert g.m_grid.shape == (g.N_m + 1,)
    assert g.x_grid.shape == (g.N_x + 1,)
    assert g.z_grid.shape == (g.N_z,)
    assert g.mz_table.shape == (g.N_m + 1, g.N_x + 1, g.N_z)
    assert g.xz_table.shape == (g.N_m + 1, g.N_x + 1, g.N_z)
    assert g.Ubar_ratio.shape == (g.N_m + 1, g.N_x + 1, g.N_z)


def test_grid_bounds(small_grid):
    """Grid should span [0, m_max] and [0, x_max]."""
    g = small_grid
    assert abs(g.m_grid[0]) < 1e-12
    assert abs(g.m_grid[-1] - g.m_max) < 1e-12
    assert abs(g.x_grid[0]) < 1e-12
    assert abs(g.x_grid[-1] - g.x_max) < 1e-12


def test_z_grid_range(small_grid):
    """z-grid should span [0, 1]."""
    g = small_grid
    assert abs(g.z_grid[0]) < 1e-12
    assert abs(g.z_grid[-1] - 1.0) < 1e-12
    assert np.all(g.z_grid >= 0) and np.all(g.z_grid <= 1)


def test_post_trade_admissibility(small_grid):
    """Post-trade states must be non-negative."""
    g = small_grid
    assert np.all(g.mz_table >= -1e-10)
    assert np.all(g.xz_table >= -1e-10)


def test_post_trade_z0_no_trade(small_grid):
    """At z=0, risky holding becomes 0: m_z = 0, x_z = x + (1-eps_sell)*m."""
    g = small_grid
    # z=0 means mz = 0 * (x/(1+eps_buy) + m) = 0
    mz_at_z0 = g.mz_table[:, :, 0]
    np.testing.assert_allclose(mz_at_z0, 0.0, atol=1e-12)


def test_post_trade_z1_full_risky(small_grid):
    """At z=1, m_z = x/(1+eps_buy) + m (maximum risky holding)."""
    g = small_grid
    m = g.m_grid[:, None]
    x = g.x_grid[None, :]
    mz_expected = x / (1 + g.eps_buy) + m
    mz_at_z1 = g.mz_table[:, :, -1]
    np.testing.assert_allclose(mz_at_z1, mz_expected, atol=1e-10)


def test_wealth_conservation(small_grid):
    """
    Total wealth is not conserved after trading (costs apply),
    but post-trade wealth should be <= pre-trade wealth (costs reduce wealth).
    """
    g = small_grid
    w_pre = g.m_grid[:, None, None] + g.x_grid[None, :, None]
    w_post = g.mz_table + g.xz_table
    # Post-trade wealth <= pre-trade wealth (transaction costs)
    assert np.all(w_post <= w_pre + 1e-10)


def test_Ubar_ratio_positive(small_grid):
    """Ubar_ratio must be strictly positive."""
    g = small_grid
    assert np.all(g.Ubar_ratio > 0)


def test_make_grid_factory():
    """make_grid should produce correct domain for each utility label."""
    g1 = make_grid("U1", 0.05, 0.05, N_m=10, N_x=10, N_z=5)
    assert g1.m_max == 4.0
    g2 = make_grid("U2", 0.02, 0.02, N_m=10, N_x=10, N_z=5)
    assert g2.m_max == 2.0
    g3 = make_grid("U3", 0.05, 0.05, N_m=10, N_x=10, N_z=5)
    assert g3.m_max == 4.0


def test_make_grid_override_bounds():
    """Custom bounds should override defaults."""
    g = make_grid("U1", 0.05, 0.05, N_m=5, N_x=5, N_z=5, m_max=6.0, x_max=7.0)
    assert g.m_max == 6.0
    assert g.x_max == 7.0


def test_uniform_spacing(small_grid):
    """Grid should have uniform spacing."""
    g = small_grid
    diffs_m = np.diff(g.m_grid)
    diffs_x = np.diff(g.x_grid)
    np.testing.assert_allclose(diffs_m, g.dm, rtol=1e-12)
    np.testing.assert_allclose(diffs_x, g.dx, rtol=1e-12)
