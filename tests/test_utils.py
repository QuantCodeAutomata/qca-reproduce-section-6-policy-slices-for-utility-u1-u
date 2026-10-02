"""
Tests for src/utils.py — verifying mathematical properties of utility functions
and model coefficient computations.
"""

import pytest
import numpy as np
from scipy.special import ndtr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    U1, U2, U3, Ubar, dUbar, d2Ubar, phi,
    mu_tilde, gamma_coeff, gtilde, compute_alpha_lower_bound,
    MU, SIGMA, LAMBDA,
)


# ---------------------------------------------------------------------------
# phi / normal CDF
# ---------------------------------------------------------------------------

def test_phi_range():
    """phi must return values in [0, 1]."""
    w = np.linspace(-5, 5, 100)
    vals = phi(w)
    assert np.all(vals >= 0) and np.all(vals <= 1)


def test_phi_at_zero():
    """phi(0) = 0.5."""
    assert abs(phi(0.0) - 0.5) < 1e-10


def test_phi_monotone():
    """phi must be non-decreasing."""
    x = np.linspace(-3, 3, 50)
    vals = phi(x)
    assert np.all(np.diff(vals) >= 0)


# ---------------------------------------------------------------------------
# Ubar and derivatives
# ---------------------------------------------------------------------------

def test_Ubar_positive():
    """Ubar(w) > 0 for all w > 0."""
    w = np.linspace(0.01, 5, 200)
    assert np.all(Ubar(w) > 0)


def test_Ubar_formula():
    """Ubar(w) = 1 + w^2 + w^(-2)."""
    w = 1.5
    expected = 1 + w**2 + w**(-2)
    assert abs(Ubar(w) - expected) < 1e-12


def test_dUbar_formula():
    """Ubar'(w) = 2w - 2w^(-3)."""
    w = 2.0
    expected = 2 * w - 2 * w**(-3)
    assert abs(dUbar(w) - expected) < 1e-12


def test_d2Ubar_formula():
    """Ubar''(w) = 2 + 6w^(-4)."""
    w = 1.8
    expected = 2 + 6 * w**(-4)
    assert abs(d2Ubar(w) - expected) < 1e-12


def test_dUbar_numerical_consistency():
    """Verify dUbar against finite-difference approximation."""
    w = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    h = 1e-6
    fd = (Ubar(w + h) - Ubar(w - h)) / (2 * h)
    analytical = dUbar(w)
    # atol needed: at w=1 dUbar=0 exactly, so relative error is ill-defined
    np.testing.assert_allclose(analytical, fd, rtol=1e-5, atol=1e-8)


def test_d2Ubar_numerical_consistency():
    """Verify d2Ubar against finite-difference approximation."""
    w = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    h = 1e-5
    fd2 = (Ubar(w + h) - 2 * Ubar(w) + Ubar(w - h)) / h**2
    analytical = d2Ubar(w)
    np.testing.assert_allclose(analytical, fd2, rtol=1e-4)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def test_U1_at_known_points():
    """U1(w) = -1/w + 6*Phi((w-1.2)/0.1) + 3.5*Phi((w-1.8)/0.2)."""
    w = 1.5
    expected = -1.0 / w + 6 * ndtr((w - 1.2) / 0.1) + 3.5 * ndtr((w - 1.8) / 0.2)
    assert abs(U1(w) - expected) < 1e-12


def test_U2_at_known_points():
    """U2(w) = -1/w^2 + 8*Phi((w-1.8)/0.15)."""
    w = 0.9
    expected = -1.0 / w**2 + 8 * ndtr((w - 1.8) / 0.15)
    assert abs(U2(w) - expected) < 1e-12


def test_U3_at_known_points():
    """U3(w) = -1/w + 6*Phi((w-1.6)/0.02) + 12*Phi((w-2.1)/0.05)."""
    w = 1.7
    expected = -1.0 / w + 6 * ndtr((w - 1.6) / 0.02) + 12 * ndtr((w - 2.1) / 0.05)
    assert abs(U3(w) - expected) < 1e-12


def test_utilities_vectorised():
    """All utilities should accept arrays."""
    w = np.linspace(0.5, 3.0, 50)
    for fn in [U1, U2, U3]:
        result = fn(w)
        assert result.shape == w.shape


# ---------------------------------------------------------------------------
# PDE coefficients
# ---------------------------------------------------------------------------

def test_mu_tilde_formula():
    """mu_tilde = mu - sigma^2/2 + sigma^2 * m * Ubar'(w) / Ubar(w)."""
    m, w = 1.0, 2.0
    sig2 = SIGMA**2
    expected = MU - sig2 / 2 + sig2 * m * dUbar(w) / Ubar(w)
    assert abs(mu_tilde(np.array([m]), np.array([w]))[0] - expected) < 1e-12


def test_gamma_coeff_formula():
    """Check gamma formula against direct computation."""
    m, w = 0.8, 1.5
    sig2 = SIGMA**2
    Ub = Ubar(w)
    dUb = dUbar(w)
    d2Ub = d2Ubar(w)
    expected = (sig2 * (m * dUb + m**2 * d2Ub) / (2 * Ub)
                + (MU - sig2 / 2) * m * dUb / Ub)
    result = gamma_coeff(np.array([m]), np.array([w]))[0]
    assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# Terminal condition
# ---------------------------------------------------------------------------

def test_gtilde_scalar():
    """gtilde(m=0, x=2) = U1(2*(1-eps_sell)) / Ubar(2)."""
    m, x = 0.5, 1.5
    eps_sell = 0.05
    m2d = np.array([[m]])
    x2d = np.array([[x]])
    w_sell = x + (1 - eps_sell) * m
    expected = U1(w_sell) / Ubar(m + x)
    result = gtilde(m2d, x2d, U1, eps_sell)[0, 0]
    assert abs(result - expected) < 1e-10


def test_gtilde_m_zero():
    """When m=0, gtilde(0, x) = U1(x) / Ubar(x)."""
    x = 1.5
    m2d = np.array([[0.0]])
    x2d = np.array([[x]])
    expected = U1(x) / Ubar(x)
    result = gtilde(m2d, x2d, U1, 0.05)[0, 0]
    assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# Alpha lower bound
# ---------------------------------------------------------------------------

def test_alpha_lb_positive():
    """Alpha must be strictly positive."""
    alpha = compute_alpha_lower_bound(0.05, gamma_inf_estimate=10.0)
    assert alpha > 0


def test_alpha_lb_increases_with_eps():
    """Larger transaction costs -> higher alpha requirement."""
    a1 = compute_alpha_lower_bound(0.02, gamma_inf_estimate=10.0)
    a2 = compute_alpha_lower_bound(0.10, gamma_inf_estimate=10.0)
    assert a2 > a1


def test_alpha_lb_theoretical_bound():
    """Alpha must exceed c0 = 2*lambda*c_U + lambda + gamma_inf."""
    eps = 0.05
    gamma_inf = 10.0
    p = 2
    c_U = (1 + eps)**p + (1 - eps)**(-p)
    c0 = 2 * LAMBDA * c_U + LAMBDA + gamma_inf
    alpha = compute_alpha_lower_bound(eps, gamma_inf)
    assert alpha > c0, f"alpha={alpha:.2f} should exceed c0={c0:.2f}"
