"""
Reduced intervention operator Kbar and z-optimal control computation.

Kbar(phi)(t, m, x, z) = phi(t, m_z, x_z) * Ubar(w_z) / Ubar(w)

The supremum over z in [0, 1] is evaluated on a discrete z-grid with optional
local refinement around the top candidates.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from src.grid import Grid
from src.interpolation import bilinear_interp_3d


# ---------------------------------------------------------------------------
# Low-level helpers: post-trade state maps
# ---------------------------------------------------------------------------

def post_trade_states(
    m: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    eps_buy: float,
    eps_sell: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute post-trade (m_z, x_z) from pre-trade state and control z.

    m_z = z * (x / (1+eps_buy) + m)
    x_z = x - (1+eps_buy)*(m_z-m)^+ + (1-eps_sell)*(m_z-m)^-

    Parameters
    ----------
    m, x : np.ndarray
        Pre-trade risky holdings and cash (broadcastable with z).
    z : np.ndarray
        Control in [0, 1].
    eps_buy, eps_sell : float
        Proportional transaction costs.

    Returns
    -------
    mz, xz : np.ndarray
        Post-trade state, clipped to [0, inf).
    """
    mz = z * (x / (1.0 + eps_buy) + m)
    delta = mz - m
    xz = x - (1.0 + eps_buy) * np.maximum(delta, 0.0) + (1.0 - eps_sell) * np.maximum(-delta, 0.0)
    mz = np.maximum(mz, 0.0)
    xz = np.maximum(xz, 0.0)
    return mz, xz


# ---------------------------------------------------------------------------
# Kbar at a single grid point / z index
# ---------------------------------------------------------------------------

def compute_Kbar_at_z(
    nu: np.ndarray,
    grid: Grid,
    i_m: int,
    j_x: int,
    k_z: int,
) -> float:
    """
    Compute Kbar(nu) at grid point (i_m, j_x) for control index k_z.

    Returns
    -------
    float
        Kbar value = nu_interp(mz, xz) * Ubar(wz) / Ubar(w).
    """
    from src.interpolation import bilinear_interp_batch
    mz = np.array([grid.mz_table[i_m, j_x, k_z]])
    xz = np.array([grid.xz_table[i_m, j_x, k_z]])
    nu_post = bilinear_interp_batch(nu, grid.m_grid, grid.x_grid, mz, xz)[0]
    return float(nu_post * grid.Ubar_ratio[i_m, j_x, k_z])


# ---------------------------------------------------------------------------
# Full grid sup-Kbar (internal fast version used by solver)
# ---------------------------------------------------------------------------

def sup_Kbar(
    nu: np.ndarray,
    grid: Grid,
    return_z_idx: bool = False,
    top_k: int = 2,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sup_{z in [0,1]} Kbar(nu) at every grid point.

    Parameters
    ----------
    nu : np.ndarray, shape (N_m+1, N_x+1)
    grid : Grid
    return_z_idx : bool
    top_k : int

    Returns
    -------
    sup_val : np.ndarray, shape (N_m+1, N_x+1)
    z_hat_idx : np.ndarray (if return_z_idx), shape (N_m+1, N_x+1)
    top_kbar : np.ndarray (if return_z_idx), shape (N_m+1, N_x+1, top_k)
    """
    nu_at_post = bilinear_interp_3d(
        nu, grid.m_grid, grid.x_grid, grid.mz_table, grid.xz_table
    )
    kbar_vals = nu_at_post * grid.Ubar_ratio  # (N_m+1, N_x+1, N_z)
    sup_val = np.max(kbar_vals, axis=2)

    if not return_z_idx:
        return sup_val

    N_z = grid.N_z
    kbar_flip = kbar_vals[:, :, ::-1]
    argmax_flip = np.argmax(kbar_flip, axis=2)
    z_hat_idx = (N_z - 1) - argmax_flip

    if top_k > 1 and top_k <= N_z:
        sorted_kbar = -np.partition(-kbar_vals, top_k - 1, axis=2)[:, :, :top_k]
        top_kbar = np.sort(sorted_kbar, axis=2)[:, :, ::-1]
    else:
        top_kbar = sup_val[:, :, None]

    return sup_val, z_hat_idx, top_kbar


# ---------------------------------------------------------------------------
# Public compute_sup_Kbar — returns full kbar_all tensor for tests/diagnostics
# ---------------------------------------------------------------------------

def compute_sup_Kbar(
    nu: np.ndarray,
    grid: Grid,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the sup of Kbar(nu) over z, the argmax z value, and full Kbar tensor.

    Parameters
    ----------
    nu : np.ndarray, shape (N_m+1, N_x+1)
    grid : Grid

    Returns
    -------
    sup_val : np.ndarray, shape (N_m+1, N_x+1)
        Maximum Kbar value at each grid point.
    z_hat : np.ndarray, shape (N_m+1, N_x+1), float
        Argmax z value (from z_grid); tie-break = largest z.
    kbar_all : np.ndarray, shape (N_m+1, N_x+1, N_z)
        Full Kbar tensor over all z values.
    """
    nu_at_post = bilinear_interp_3d(
        nu, grid.m_grid, grid.x_grid, grid.mz_table, grid.xz_table
    )
    kbar_all = nu_at_post * grid.Ubar_ratio  # (N_m+1, N_x+1, N_z)
    sup_val = np.max(kbar_all, axis=2)

    N_z = grid.N_z
    kbar_flip = kbar_all[:, :, ::-1]
    argmax_flip = np.argmax(kbar_flip, axis=2)
    z_hat_idx = (N_z - 1) - argmax_flip          # largest maximising z index
    z_hat = grid.z_grid[z_hat_idx]               # float values in [0,1]

    return sup_val, z_hat, kbar_all


# ---------------------------------------------------------------------------
# Policy slice extraction
# ---------------------------------------------------------------------------

def extract_policy_slice(
    nu: np.ndarray,
    grid: Grid,
    wealth_target: float,
    pi_grid: np.ndarray,
    w_floor: float = 1e-10,
    ntrade_dz_frac: float = 0.52,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover the optimal policy pi-hat(pi) for a fixed target wealth.

    No-trade identification:
    For each pi, we compute z_nt — the control z that leaves the portfolio
    unchanged (mz = m_pre, xz = x_pre).  If the argmax z_hat is within
    ``ntrade_dz_frac * dz`` of z_nt, the discrete solver is selecting the
    identity intervention, so we declare no-trade and set pi_hat = pi.
    This eliminates the staircase artifact that arises when z_nt does not
    fall exactly on the z-grid.

    Parameters
    ----------
    nu : np.ndarray, shape (N_m+1, N_x+1)
    grid : Grid
    wealth_target : float
    pi_grid : np.ndarray
    w_floor : float
    ntrade_dz_frac : float
        Multiple of dz used for no-trade proximity test (default 0.52, just
        over half a grid step).

    Returns
    -------
    pi_hat : np.ndarray
        Post-trade risky weight; equals pi for no-trade points.
    z_hat : np.ndarray
        Argmax control; equals z_nt for no-trade points.
    top2_kbar : np.ndarray, shape (len(pi_grid), 2)
    kbar_all : np.ndarray, shape (len(pi_grid), N_z)
    """
    from src.interpolation import bilinear_interp_batch
    from src.utils import Ubar

    w = wealth_target
    N_pi = len(pi_grid)
    N_z = grid.N_z

    m_pre = w * pi_grid
    x_pre = w * (1.0 - pi_grid)

    z = grid.z_grid[None, :]
    m_p = m_pre[:, None]
    x_p = x_pre[:, None]

    mz, xz = post_trade_states(m_p, x_p, z, grid.eps_buy, grid.eps_sell)

    w_pre = np.maximum(m_pre + x_pre, w_floor)
    w_post = np.maximum(mz + xz, w_floor)
    ubar_ratio = Ubar(w_post) / Ubar(w_pre[:, None])

    nu_post = bilinear_interp_batch(nu, grid.m_grid, grid.x_grid, mz, xz)
    kbar_all = nu_post * ubar_ratio

    # Argmax with tie-break = largest z index
    kbar_flip = kbar_all[:, ::-1]
    argmax_flip = np.argmax(kbar_flip, axis=1)
    z_hat_idx = (N_z - 1) - argmax_flip
    z_hat = grid.z_grid[z_hat_idx]

    mz_hat = mz[np.arange(N_pi), z_hat_idx]
    xz_hat = xz[np.arange(N_pi), z_hat_idx]
    w_hat = np.maximum(mz_hat + xz_hat, w_floor)
    pi_hat_raw = mz_hat / w_hat

    # ---- No-trade identification via z-proximity -------------------------
    # z_nt satisfies mz(z_nt) = m_pre, i.e. z_nt = m_pre / M_cap
    # where M_cap = x_pre/(1+eps_buy) + m_pre  (≥ 0).
    # If z_hat is within ntrade_dz_frac * dz of z_nt, the discrete argmax
    # is selecting the identity control → declare no-trade, set pi_hat = pi.
    M_cap = x_pre / (1.0 + grid.eps_buy) + m_pre         # (N_pi,)
    z_nt = m_pre / np.maximum(M_cap, 1e-15)              # (N_pi,)
    dz = 1.0 / max(N_z - 1, 1)
    is_notrade = np.abs(z_hat - z_nt) <= ntrade_dz_frac * dz

    pi_hat = np.where(is_notrade, pi_grid, pi_hat_raw)
    # -----------------------------------------------------------------------

    # Top-2 Kbar (safe: only when N_z >= 2)
    if N_z >= 2:
        top2 = -np.partition(-kbar_all, 1, axis=1)[:, :2]
        top2_kbar = np.sort(top2, axis=1)[:, ::-1]
    else:
        top2_kbar = kbar_all

    return pi_hat, z_hat, top2_kbar, kbar_all


# ---------------------------------------------------------------------------
# Action classification (kept here for backward compat; canonical in policy.py)
# ---------------------------------------------------------------------------

def classify_actions(
    pi_hat: np.ndarray,
    pi_grid: np.ndarray,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Classify each pre-trade pi as 'buy', 'sell', or 'notrade'.

    Parameters
    ----------
    pi_hat : np.ndarray  (post-trade weights)
    pi_grid : np.ndarray (pre-trade weights)
    tol : float

    Returns
    -------
    np.ndarray of str
    """
    diff = pi_hat - pi_grid
    return np.where(diff > tol, "buy", np.where(diff < -tol, "sell", "notrade"))
