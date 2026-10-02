"""
Utility functions and model parameters for the reduced HJB/PIDE experiments.

Utilities U1, U2, U3 are non-concave and defined as in the paper (Section 6).
Ubar(w) = 1 + w^2 + w^(-2) is the normalisation weight with p=2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF  # Custom — no library wrapper needed


# ---------------------------------------------------------------------------
# Model constants (Section 6, all experiments)
# ---------------------------------------------------------------------------
MU: float = 0.06        # drift
SIGMA: float = 0.35     # volatility
LAMBDA: float = 20.0    # Poisson trading-opportunity arrival rate
T_HORIZON: float = 1.0  # horizon
T: float = T_HORIZON    # shorthand alias used in tests


def phi(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF Phi(x). Uses scipy.special.ndtr."""
    return ndtr(x)


# ---------------------------------------------------------------------------
# Non-concave utility functions (paper Section 6)
# ---------------------------------------------------------------------------

def U1(w: np.ndarray | float) -> np.ndarray | float:
    """
    Utility U1: U1(w) = -1/w + 6*Phi((w-1.2)/0.1) + 3.5*Phi((w-1.8)/0.2).
    """
    w = np.asarray(w, dtype=float)
    return -1.0 / w + 6.0 * phi((w - 1.2) / 0.1) + 3.5 * phi((w - 1.8) / 0.2)


def U2(w: np.ndarray | float) -> np.ndarray | float:
    """
    Utility U2: U2(w) = -1/w^2 + 8*Phi((w-1.8)/0.15).
    """
    w = np.asarray(w, dtype=float)
    return -1.0 / w**2 + 8.0 * phi((w - 1.8) / 0.15)


def U3(w: np.ndarray | float) -> np.ndarray | float:
    """
    Utility U3: U3(w) = -1/w + 6*Phi((w-1.6)/0.02) + 12*Phi((w-2.1)/0.05).
    """
    w = np.asarray(w, dtype=float)
    return -1.0 / w + 6.0 * phi((w - 1.6) / 0.02) + 12.0 * phi((w - 2.1) / 0.05)


# ---------------------------------------------------------------------------
# Normalisation weight Ubar and its derivatives (paper Appendix / Section 5)
# ---------------------------------------------------------------------------

def Ubar(w: np.ndarray | float) -> np.ndarray | float:
    """Ubar(w) = 1 + w^2 + w^(-2), the normalisation weight with p=2."""
    w = np.asarray(w, dtype=float)
    return 1.0 + w**2 + w**(-2)


def dUbar(w: np.ndarray | float) -> np.ndarray | float:
    """First derivative: Ubar'(w) = 2w - 2w^(-3)."""
    w = np.asarray(w, dtype=float)
    return 2.0 * w - 2.0 * w**(-3)


def d2Ubar(w: np.ndarray | float) -> np.ndarray | float:
    """Second derivative: Ubar''(w) = 2 + 6w^(-4)."""
    w = np.asarray(w, dtype=float)
    return 2.0 + 6.0 * w**(-4)


# ---------------------------------------------------------------------------
# Reduced PDE coefficients
# ---------------------------------------------------------------------------

def mu_tilde(m: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    mu_tilde = mu - sigma^2/2 + sigma^2 * m * Ubar'(w) / Ubar(w).
    Drift correction for the reduced state variable.
    """
    sig2 = SIGMA**2
    return MU - sig2 / 2.0 + sig2 * m * dUbar(w) / Ubar(w)


def gamma_coeff(m: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    gamma = [sigma^2 * (m*Ubar'(w) + m^2*Ubar''(w)) / 2] / Ubar(w)
            + (mu - sigma^2/2) * m * Ubar'(w) / Ubar(w).
    Zeroth-order coefficient in the reduced PDE.
    """
    sig2 = SIGMA**2
    Ub = Ubar(w)
    dUb = dUbar(w)
    d2Ub = d2Ubar(w)
    term1 = sig2 * (m * dUb + m**2 * d2Ub) / (2.0 * Ub)
    term2 = (MU - sig2 / 2.0) * m * dUb / Ub
    return term1 + term2


# ---------------------------------------------------------------------------
# Alpha lower-bound computation
# ---------------------------------------------------------------------------

def compute_alpha_lower_bound(eps_max: float, gamma_inf_estimate: float = 10.0) -> float:
    """
    Compute a conservative lower bound for alpha.

    Theory: alpha > c0 = 2*lambda*c_U + lambda + ||gamma||_inf
    where c_U = (1 + eps_M)^p + (1 - eps_M)^(-p) with p=2.

    We use eps_max = max(eps_buy, eps_sell) and add margin.

    Parameters
    ----------
    eps_max : float
        Maximum transaction cost fraction.
    gamma_inf_estimate : float
        Estimated supremum of |gamma| over the domain (conservative).

    Returns
    -------
    float
        Recommended alpha value (with 20% safety margin above the lower bound).
    """
    p = 2
    c_U = (1.0 + eps_max)**p + (1.0 - eps_max)**(-p)
    c0 = 2.0 * LAMBDA * c_U + LAMBDA + gamma_inf_estimate
    alpha_lower = c0
    return 1.2 * alpha_lower  # 20% safety margin


# ---------------------------------------------------------------------------
# Terminal liquidation payoff (normalised)
# ---------------------------------------------------------------------------

def gtilde(
    m: np.ndarray,
    x: np.ndarray,
    utility_fn,
    eps_sell: float,
    w_floor: float = 1e-8,
) -> np.ndarray:
    """
    Compute the normalised terminal payoff:
        gtilde(m, x) = U(x + (1 - eps_sell)*m) / Ubar(x + m)

    Parameters
    ----------
    m, x : np.ndarray
        2D grids of risky holdings and cash (shapes must match).
    utility_fn : callable
        One of U1, U2, U3.
    eps_sell : float
        Proportional selling cost.
    w_floor : float
        Minimum wealth floor for numerical safety.
    """
    w_sell = np.maximum(x + (1.0 - eps_sell) * m, w_floor)
    w_pre = np.maximum(x + m, w_floor)
    return utility_fn(w_sell) / Ubar(w_pre)
