"""
Tests for src/interpolation.py — bilinear interpolation on regular grids.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interpolation import bilinear_interp


@pytest.fixture
def simple_grid():
    """2×2 grid for analytic verification."""
    m_grid = np.array([0.0, 1.0, 2.0])
    x_grid = np.array([0.0, 1.0, 2.0])
    # f(m, x) = m + 2*x
    nu = np.outer(m_grid, np.ones(3)) + 2 * np.outer(np.ones(3), x_grid)
    return m_grid, x_grid, nu


def test_bilinear_at_nodes(simple_grid):
    """Interpolation at grid nodes must be exact."""
    m_grid, x_grid, nu = simple_grid
    for i, m in enumerate(m_grid):
        for j, x in enumerate(x_grid):
            val = bilinear_interp(nu, m_grid, x_grid, np.array([m]), np.array([x]))[0]
            assert abs(val - nu[i, j]) < 1e-12


def test_bilinear_midpoint(simple_grid):
    """Interpolation at midpoint of a cell."""
    m_grid, x_grid, nu = simple_grid
    # f(0.5, 0.5) = 0.5 + 2*0.5 = 1.5
    val = bilinear_interp(nu, m_grid, x_grid, np.array([0.5]), np.array([0.5]))[0]
    assert abs(val - 1.5) < 1e-12


def test_bilinear_affine_exact(simple_grid):
    """Bilinear interpolation is exact for linear functions."""
    m_grid, x_grid, nu = simple_grid
    # f(m, x) = m + 2*x is linear (affine), so bilinear must be exact everywhere
    test_pts_m = np.array([0.3, 0.7, 1.1, 1.9])
    test_pts_x = np.array([0.6, 0.2, 1.5, 0.9])
    expected = test_pts_m + 2 * test_pts_x
    result = bilinear_interp(nu, m_grid, x_grid, test_pts_m, test_pts_x)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_bilinear_clipping():
    """Points outside the grid should be clipped to boundary."""
    m_grid = np.array([0.0, 1.0])
    x_grid = np.array([0.0, 1.0])
    nu = np.array([[0.0, 2.0], [1.0, 3.0]])
    # Point beyond x_max=1 should be clipped to x=1
    val = bilinear_interp(nu, m_grid, x_grid, np.array([0.5]), np.array([1.5]))[0]
    val_at_boundary = bilinear_interp(nu, m_grid, x_grid, np.array([0.5]), np.array([1.0]))[0]
    assert abs(val - val_at_boundary) < 1e-10


def test_bilinear_vectorised():
    """bilinear_interp should handle large arrays efficiently."""
    N = 10000
    m_grid = np.linspace(0, 3, 50)
    x_grid = np.linspace(0, 3, 50)
    nu = np.random.rand(50, 50)

    pts_m = np.random.uniform(0, 3, N)
    pts_x = np.random.uniform(0, 3, N)
    result = bilinear_interp(nu, m_grid, x_grid, pts_m, pts_x)
    assert result.shape == (N,)
    assert np.all(np.isfinite(result))


def test_bilinear_constant():
    """Bilinear interp of a constant function equals the constant."""
    m_grid = np.linspace(0, 5, 20)
    x_grid = np.linspace(0, 5, 20)
    nu = np.full((20, 20), 7.0)
    pts_m = np.random.uniform(0, 5, 200)
    pts_x = np.random.uniform(0, 5, 200)
    result = bilinear_interp(nu, m_grid, x_grid, pts_m, pts_x)
    np.testing.assert_allclose(result, 7.0, atol=1e-12)


def test_bilinear_output_range():
    """Bilinear interpolation should not extrapolate beyond node values."""
    m_grid = np.linspace(0, 2, 10)
    x_grid = np.linspace(0, 2, 10)
    nu = np.sin(m_grid[:, None] + x_grid[None, :])
    pts_m = np.random.uniform(0, 2, 500)
    pts_x = np.random.uniform(0, 2, 500)
    result = bilinear_interp(nu, m_grid, x_grid, pts_m, pts_x)
    assert np.all(result >= nu.min() - 1e-12)
    assert np.all(result <= nu.max() + 1e-12)
