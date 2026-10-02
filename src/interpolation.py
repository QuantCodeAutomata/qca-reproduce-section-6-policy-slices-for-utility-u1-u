"""
Vectorized bilinear interpolation on a 2-D regular (m, x) grid.

The function bilinear_interp_batch is the performance-critical routine
called at every time step to evaluate the value function at off-grid
post-trade states (m_z, x_z).
"""

from __future__ import annotations

import numpy as np


def bilinear_interp_batch(
    nu: np.ndarray,
    m_grid: np.ndarray,
    x_grid: np.ndarray,
    pts_m: np.ndarray,
    pts_x: np.ndarray,
) -> np.ndarray:
    """
    Vectorized bilinear interpolation of nu on a regular (m, x) grid.

    Parameters
    ----------
    nu : np.ndarray, shape (N_m+1, N_x+1)
        Value function on the grid.
    m_grid : np.ndarray, shape (N_m+1,)
        Uniform m-axis grid points.
    x_grid : np.ndarray, shape (N_x+1,)
        Uniform x-axis grid points.
    pts_m : np.ndarray, arbitrary shape
        Query points along m-axis.
    pts_x : np.ndarray, same shape as pts_m
        Query points along x-axis.

    Returns
    -------
    np.ndarray
        Interpolated values, same shape as pts_m.
    """
    shape = pts_m.shape
    pts_m = pts_m.ravel()
    pts_x = pts_x.ravel()

    N_m = len(m_grid) - 1
    N_x = len(x_grid) - 1

    dm = m_grid[1] - m_grid[0]
    dx = x_grid[1] - x_grid[0]

    # Find left-neighbour indices (uniform grid allows direct index computation)
    i = np.floor(pts_m / dm).astype(np.int64)
    j = np.floor(pts_x / dx).astype(np.int64)

    # Clip to valid index range
    i = np.clip(i, 0, N_m - 1)
    j = np.clip(j, 0, N_x - 1)

    # Fractional weights within each cell
    alpha = (pts_m - m_grid[i]) / dm   # in [0, 1]
    beta = (pts_x - x_grid[j]) / dx    # in [0, 1]
    alpha = np.clip(alpha, 0.0, 1.0)
    beta = np.clip(beta, 0.0, 1.0)

    # Four corners of the bilinear cell
    f00 = nu[i,     j    ]
    f10 = nu[i + 1, j    ]
    f01 = nu[i,     j + 1]
    f11 = nu[i + 1, j + 1]

    # Bilinear blend
    result = (
        (1.0 - alpha) * (1.0 - beta) * f00
        + alpha * (1.0 - beta) * f10
        + (1.0 - alpha) * beta * f01
        + alpha * beta * f11
    )

    return result.reshape(shape)


def bilinear_interp_3d(
    nu: np.ndarray,
    m_grid: np.ndarray,
    x_grid: np.ndarray,
    mz_table: np.ndarray,
    xz_table: np.ndarray,
) -> np.ndarray:
    """
    Evaluate bilinear_interp_batch for the full (N_m+1, N_x+1, N_z) table.

    Parameters
    ----------
    nu : np.ndarray, shape (N_m+1, N_x+1)
    m_grid, x_grid : grids
    mz_table : np.ndarray, shape (N_m+1, N_x+1, N_z)
    xz_table : np.ndarray, shape (N_m+1, N_x+1, N_z)

    Returns
    -------
    np.ndarray, shape (N_m+1, N_x+1, N_z)
        nu evaluated at (mz_table, xz_table).
    """
    return bilinear_interp_batch(nu, m_grid, x_grid, mz_table, xz_table)


# Alias used in tests and external callers
bilinear_interp = bilinear_interp_batch
