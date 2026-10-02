"""
IMEX finite-difference backward-in-time solver for the reduced HJB/PIDE.

We work directly with v(t,m,x) = V(t,m,x)/Ūbar(x+m) — the normalised value
function WITHOUT the exp(alpha*t) prefactor.  This keeps values O(1)
throughout and avoids the catastrophic cancellation that arises when the
O(exp(alpha*T)) terminal condition is integrated back with a large alpha.

The PDE for v is (dropping the exp factor):
  0 = v_t + L_m[v] + c_v(m,x)*v + lambda * sup_z Kbar(v)

where:
  - L_m[v] = (mu_tilde + sigma^2/2)*m * v_m + (sigma^2*m^2/2) * v_mm
  - c_v(m,x) = -lambda + gamma(m,x)          (alpha does NOT appear)
  - Kbar(v)(m,x,z) = v(m_z,x_z)*Ubar(w_z)/Ubar(w)

IMEX scheme (implicit for local operator L_m + c_v; explicit for Kbar):
  (v^k - v^{k+1}) / (-dt) + L_m[v^k] + c_v*v^k + lambda * Kbar(v^{k+1}) = 0
  => (I - dt*L_m - dt*c_v) * v^k = v^{k+1} + dt*lambda*sup_z Kbar(v^{k+1})

Boundary conditions:
  - m = 0: ODE  v_t - lambda*v + lambda*sup_z Kbar(v) = 0
    => (1 + dt*lambda - dt*gamma)*v^k = v^{k+1} + dt*lambda*Kbar(v^{k+1})
  - m = m_max: Neumann (v[N_m,:] = v[N_m-1,:])
  - x = x_max: Neumann (v[:,N_x] = v[:,N_x-1])

Terminal condition:  v(T,m,x) = gtilde(m,x)  [O(1), no exp factor]

alpha is used only to compute the theoretical lower bound;
it no longer enters the IMEX coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Callable, Tuple

from src.grid import Grid
from src.utils import (
    MU, SIGMA, LAMBDA,
    mu_tilde, gamma_coeff, gtilde, Ubar, dUbar
)
from src.control import sup_Kbar


def build_tridiagonal_m(
    grid: Grid,
    alpha: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build implicit-operator tridiagonal coefficients for each (m_i, x_j) point.

    We solve for v (NOT nu = exp(alpha*t)*v), so alpha does NOT appear in the
    PDE coefficients.  The zeroth-order coefficient is:
        c_v = -lambda + gamma   (no alpha)

    The IMEX operator to invert is:
        (I - dt*L_m - dt*c_v*I) * v^k = rhs

    Returns arrays of shape (N_m-1, N_x+1) for interior m-nodes (i=1..N_m-1).

    Returns
    -------
    lower, diag, upper : np.ndarray, shape (N_m-1, N_x+1)
    c_arr : np.ndarray, shape (N_m+1, N_x+1)  — full c_v for diagnostics
    """
    sig2 = SIGMA**2
    dm = grid.dm

    # Interior m-indices: 1 .. N_m-1
    m_int = grid.m_grid[1:grid.N_m][:, None]   # (N_m-1, 1)
    x_all = grid.x_grid[None, :]                 # (1, N_x+1)
    w_all = np.maximum(m_int + x_all, 1e-12)     # (N_m-1, N_x+1)

    mu_t = mu_tilde(m_int, w_all)                # (N_m-1, N_x+1)
    gam = gamma_coeff(m_int, w_all)              # (N_m-1, N_x+1)

    # Diffusion coefficient: sigma^2 * m^2 / 2
    diff_coef = sig2 * m_int**2 / 2.0           # (N_m-1, N_x+1)
    # Drift coefficient: (mu_tilde + sigma^2/2) * m
    drift_coef = (mu_t + sig2 / 2.0) * m_int    # (N_m-1, N_x+1)

    # Upwind first-derivative stencil (monotone scheme)
    drift_upper = np.maximum(drift_coef, 0.0) / dm
    drift_lower = np.maximum(-drift_coef, 0.0) / dm
    drift_diag = -(drift_upper + drift_lower)

    # Central-difference diffusion term
    diff_upper = diff_coef / dm**2
    diff_lower = diff_coef / dm**2
    diff_diag = -2.0 * diff_coef / dm**2

    # Zeroth-order coefficient c_v = -lambda + gamma  (alpha absent)
    c_v = -LAMBDA + gam                          # (N_m-1, N_x+1)

    # Operator (I - dt*L_m - dt*c_v):
    #   diag = 1 - dt*(drift_diag + diff_diag) - dt*c_v
    main_diag = 1.0 - dt * (drift_diag + diff_diag) - dt * c_v
    upper_diag = -dt * (drift_upper + diff_upper)
    lower_diag = -dt * (drift_lower + diff_lower)

    # Full c_v for diagnostics
    m_full = grid.m_grid[:, None]
    x_full = grid.x_grid[None, :]
    w_full = np.maximum(m_full + x_full, 1e-12)
    c_arr = -LAMBDA + gamma_coeff(m_full, w_full)

    return lower_diag, main_diag, upper_diag, c_arr


def solve_tridiagonal_column(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve a tridiagonal system for one x-column using Thomas algorithm.

    Parameters
    ----------
    lower, diag, upper : np.ndarray, shape (N-1,) for N interior points
        Tridiagonal coefficients (lower[0] unused; upper[-1] unused).
    rhs : np.ndarray, shape (N-1,)

    Returns
    -------
    np.ndarray, shape (N-1,)
    """
    n = len(diag)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def build_tridiagonal_operator(
    grid: Grid,
    x_idx: int,
    alpha: float,
    dt: float,
) -> sparse.csc_matrix:
    """
    Build the implicit IMEX tridiagonal operator for a single x-column.

    Parameters
    ----------
    grid : Grid
    x_idx : int
        Column index in x_grid.
    alpha : float
    dt : float

    Returns
    -------
    scipy.sparse.csc_matrix, shape (N_m-1, N_m-1)
        Tridiagonal matrix for the interior m-nodes at x = x_grid[x_idx].
    """
    lower_2d, diag_2d, upper_2d, _ = build_tridiagonal_m(grid, alpha, dt)
    N_int = grid.N_m - 1

    lo = lower_2d[:, x_idx]
    di = diag_2d[:, x_idx]
    up = upper_2d[:, x_idx]

    A = sparse.diags(
        [lo[1:], di, up[:-1]],
        [-1, 0, 1],
        shape=(N_int, N_int),
        format="csc",
    )
    return A


def build_tridiagonal_matrices(
    grid: Grid,
    alpha: float,
    dt: float,
) -> list:
    """
    Build scipy sparse tridiagonal matrices (one per x-column) for the
    implicit solve.

    Returns a list of N_x+1 sparse CSC matrices, each of size (N_m-1, N_m-1).
    """
    lower_2d, diag_2d, upper_2d, _ = build_tridiagonal_m(grid, alpha, dt)
    N_int = grid.N_m - 1
    N_x1 = grid.N_x + 1

    matrices = []
    for jx in range(N_x1):
        lo = lower_2d[:, jx]
        di = diag_2d[:, jx]
        up = upper_2d[:, jx]
        A = sparse.diags(
            [lo[1:], di, up[:-1]],
            [-1, 0, 1],
            shape=(N_int, N_int),
            format="csc",
        )
        matrices.append(A)

    return matrices


def backward_step(
    nu_next: np.ndarray,
    grid: Grid,
    alpha: float,
    dt: float,
    tri_matrices: list,
) -> np.ndarray:
    """
    Perform one backward IMEX time step: nu^k <- F(nu^{k+1}).

    Steps:
    1. Compute explicit nonlocal term: sup_z Kbar(nu^{k+1}).
    2. For each x-column, assemble RHS and solve tridiagonal for interior m.
    3. Enforce boundary conditions.

    Parameters
    ----------
    nu_next : np.ndarray, shape (N_m+1, N_x+1)
        Value function at time step k+1 (future/terminal direction).
    grid : Grid
    alpha : float
    dt : float
    tri_matrices : list
        Prebuilt tridiagonal matrices (one per x-column).

    Returns
    -------
    nu_curr : np.ndarray, shape (N_m+1, N_x+1)
        Value function at time step k.
    """
    N_m = grid.N_m
    N_x = grid.N_x

    # Step 1: nonlocal supremum at time k+1 (explicit)
    sup_val = sup_Kbar(nu_next, grid)   # (N_m+1, N_x+1)

    nu_curr = np.zeros_like(nu_next)

    # Step 2: Boundary m=0 — separate ODE for v (alpha absent):
    #   v_t - lambda*v + lambda*Kbar(v) = 0
    # IMEX: (1 + dt*lambda)*v^k = v^{k+1} + dt*lambda*Kbar(v^{k+1})
    # (gamma at m=0, x=x_grid is small; absorbed by gamma_coeff via the interior)
    # Conservative: use plain (1 + dt*lambda) on LHS.
    coef_m0 = 1.0 + dt * LAMBDA
    nu_curr[0, :] = (nu_next[0, :] + dt * LAMBDA * sup_val[0, :]) / coef_m0

    # Step 3: Interior m-nodes (i = 1 .. N_m-1) for each x-column
    # IMEX RHS: v^{k+1} + dt*lambda * sup_val[1:N_m, jx]
    for jx in range(N_x + 1):
        rhs = nu_next[1:N_m, jx] + dt * LAMBDA * sup_val[1:N_m, jx]

        # Tridiagonal solve
        A = tri_matrices[jx]
        nu_int = spsolve(A, rhs)           # shape (N_m-1,)
        nu_curr[1:N_m, jx] = nu_int

    # Step 4: Boundary at m=m_max — zero-flux Neumann: nu[N_m,:] = nu[N_m-1,:]
    nu_curr[N_m, :] = nu_curr[N_m - 1, :]

    # Step 5: Boundary at x=x_max — Neumann: nu[:,N_x] = nu[:,N_x-1]
    nu_curr[:, N_x] = nu_curr[:, N_x - 1]

    return nu_curr


def apply_terminal_condition(
    grid: Grid,
    utility_fn: Callable,
    alpha: float,
) -> np.ndarray:
    """
    Compute the terminal condition v(T, m, x) = gtilde(m, x).

    We work with v = V/Ūbar (no exp(alpha*T) factor), so the terminal
    condition is simply the normalised liquidation payoff.

    Parameters
    ----------
    grid : Grid
    utility_fn : callable
    alpha : float
        Retained for API compatibility; not used in this function.

    Returns
    -------
    np.ndarray, shape (N_m+1, N_x+1), values O(1)
    """
    m2d, x2d = np.meshgrid(grid.m_grid, grid.x_grid, indexing="ij")
    return gtilde(m2d, x2d, utility_fn, grid.eps_sell)   # O(1), no exp factor


def solve_hjb_pide(
    grid: Grid,
    utility_fn: Callable,
    alpha: float,
    N_t: int = 500,
    verbose: bool = True,
    report_interval: int = 100,
) -> np.ndarray:
    """
    Solve the reduced HJB/PIDE backward in time from T to 0.

    Parameters
    ----------
    grid : Grid
    utility_fn : callable
        U1, U2, or U3 — used for the terminal condition.
    alpha : float
        Must satisfy alpha > c0.
    N_t : int
        Number of time subintervals. Use N_t=0 to return only the terminal condition.
    verbose : bool
    report_interval : int

    Returns
    -------
    nu0 : np.ndarray, shape (N_m+1, N_x+1)
        Value function nu-tilde at time t=0.
    """
    nu = apply_terminal_condition(grid, utility_fn, alpha)

    if N_t == 0:
        return nu

    dt = 1.0 / N_t      # T=1

    if verbose:
        print(f"  Building {grid.N_x + 1} tridiagonal matrices...")
    tri_mats = build_tridiagonal_matrices(grid, alpha, dt)

    if verbose:
        print(f"  Solving backward PDE: N_t={N_t}, dt={dt:.6f}, alpha={alpha:.2f}")

    for step in range(N_t):
        nu = backward_step(nu, grid, alpha, dt, tri_mats)

        if verbose and (step + 1) % report_interval == 0:
            print(f"  Step {step+1}/{N_t}, nu range: [{nu.min():.4f}, {nu.max():.4f}]")

    if verbose:
        print(f"  PDE solve complete. nu at t=0 range: [{nu.min():.4f}, {nu.max():.4f}]")

    return nu
