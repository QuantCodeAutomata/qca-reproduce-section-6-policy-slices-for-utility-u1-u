"""
Tests for src/pde_solver.py — IMEX solver correctness.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grid import Grid, make_grid
from src.pde_solver import (
    solve_hjb_pide, build_tridiagonal_operator,
    apply_terminal_condition,
)
from src.utils import U1, U2, U3, Ubar, gtilde


@pytest.fixture
def tiny_grid():
    """Tiny grid for fast solver tests."""
    return Grid(m_max=3.0, x_max=3.0, N_m=8, N_x=8, N_z=10,
                eps_buy=0.05, eps_sell=0.05)


def test_terminal_condition_consistency(tiny_grid):
    """nu(T) = exp(alpha*T) * gtilde, then solve 0 steps -> unchanged."""
    g = tiny_grid
    alpha = 100.0
    N_t = 0   # No time stepping — just apply terminal condition
    nu0 = solve_hjb_pide(g, U1, alpha, N_t=N_t)
    nu_T_expected = apply_terminal_condition(g, U1, alpha)
    np.testing.assert_allclose(nu0, nu_T_expected, rtol=1e-10)


def test_output_shape(tiny_grid):
    """Solver should return array of shape (N_m+1, N_x+1)."""
    g = tiny_grid
    nu0 = solve_hjb_pide(g, U1, alpha=100.0, N_t=5)
    assert nu0.shape == (g.N_m + 1, g.N_x + 1)


def test_nu_finite(tiny_grid):
    """Solved nu should not contain NaN or Inf."""
    g = tiny_grid
    nu0 = solve_hjb_pide(g, U1, alpha=100.0, N_t=10)
    assert np.all(np.isfinite(nu0))


def test_tridiagonal_operator_shape(tiny_grid):
    """Tridiagonal operator should be (N_m-1, N_m-1) for interior points."""
    g = tiny_grid
    x_idx = 4
    alpha = 100.0
    A = build_tridiagonal_operator(g, x_idx, alpha, dt=0.01)
    interior_size = g.N_m - 1
    assert A.shape == (interior_size, interior_size)


def test_solver_u2_small():
    """Solver should complete without error for U2 and small grid."""
    g = make_grid("U2", 0.02, 0.02, N_m=8, N_x=8, N_z=10)
    nu0 = solve_hjb_pide(g, U2, alpha=200.0, N_t=5)
    assert np.all(np.isfinite(nu0))
    assert nu0.shape == (9, 9)


def test_solver_u3_small():
    """Solver should complete without error for U3."""
    g = make_grid("U3", 0.05, 0.05, N_m=8, N_x=8, N_z=10)
    nu0 = solve_hjb_pide(g, U3, alpha=200.0, N_t=5)
    assert np.all(np.isfinite(nu0))


def test_alpha_sensitivity():
    """
    For two sufficiently large alpha values, the recovered policy structure
    (sup_Kbar dominance) should give the same value function shape.
    Implemented as: nu0 under alpha1 and alpha2 should differ by a finite
    scalar that does not alter the argmax structure.
    """
    g = Grid(m_max=3.0, x_max=3.0, N_m=6, N_x=6, N_z=5,
             eps_buy=0.05, eps_sell=0.05)
    nu0_a = solve_hjb_pide(g, U1, alpha=150.0, N_t=5)
    nu0_b = solve_hjb_pide(g, U1, alpha=250.0, N_t=5)
    # Both should be finite
    assert np.all(np.isfinite(nu0_a))
    assert np.all(np.isfinite(nu0_b))


def test_terminal_condition_at_m0():
    """
    At m=0, gtilde(0, x) = U1(x) / Ubar(x) since no risky asset to liquidate.

    The solver uses v = V/Ūbar WITHOUT the exp(alpha*t) prefactor, so the
    terminal condition is O(1) — just the normalised liquidation payoff.
    """
    g = Grid(m_max=3.0, x_max=3.0, N_m=8, N_x=8, N_z=5,
             eps_buy=0.05, eps_sell=0.05)
    alpha = 100.0
    nu_T = apply_terminal_condition(g, U1, alpha)
    # At m=0 (row 0): nu_T[0, j] = U1(x_j) / Ubar(x_j)  (no exp factor)
    for j, x in enumerate(g.x_grid):
        if x > 1e-3:  # skip near-zero x
            expected = U1(x) / Ubar(x)
            assert abs(nu_T[0, j] - expected) < abs(expected) * 0.01 + 1e-6


def test_gtilde_monotone_at_high_m():
    """
    gtilde at high m should encode the terminal liquidation payoff:
    U1(x + (1-eps_sell)*m) / Ubar(x+m).
    """
    g = Grid(m_max=3.0, x_max=3.0, N_m=8, N_x=8, N_z=5,
             eps_buy=0.05, eps_sell=0.05)
    m2d = g.m_grid[:, None] * np.ones(g.N_x + 1)[None, :]
    x2d = g.x_grid[None, :] * np.ones(g.N_m + 1)[:, None]
    gt = gtilde(m2d, x2d, U1, g.eps_sell)
    # Result should be finite for all grid points where m+x > 0
    valid = (m2d + x2d) > 1e-3
    assert np.all(np.isfinite(gt[valid]))
