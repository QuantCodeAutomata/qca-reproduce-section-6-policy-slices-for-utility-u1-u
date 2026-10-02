"""
Grid construction and precomputed intervention-state tables.

The computational domain is [0, m_max] x [0, x_max] with uniform spacing.
For each grid point (m_i, x_j) and control z in [0, 1], the post-trade states
(m_z, x_z) are precomputed once and reused at every time step.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

from src.utils import Ubar


@dataclass
class Grid:
    """
    Holds the discretisation of the (m, x) domain and the z-control grid,
    plus precomputed post-trade state tables.

    Attributes
    ----------
    m_max, x_max : float
        Domain truncation bounds.
    N_m, N_x : int
        Number of subintervals in m and x.
    N_z : int
        Number of z-control points.
    eps_buy, eps_sell : float
        Transaction costs.
    """

    m_max: float
    x_max: float
    N_m: int
    N_x: int
    N_z: int
    eps_buy: float
    eps_sell: float

    # Computed fields (populated in __post_init__)
    m_grid: np.ndarray = field(init=False, repr=False)
    x_grid: np.ndarray = field(init=False, repr=False)
    z_grid: np.ndarray = field(init=False, repr=False)
    dm: float = field(init=False)
    dx: float = field(init=False)

    # Post-trade state tables: shape (N_m+1, N_x+1, N_z)
    mz_table: np.ndarray = field(init=False, repr=False)
    xz_table: np.ndarray = field(init=False, repr=False)
    Ubar_ratio: np.ndarray = field(init=False, repr=False)  # Ubar(w_z) / Ubar(w)

    def __post_init__(self) -> None:
        self.m_grid = np.linspace(0.0, self.m_max, self.N_m + 1)
        self.x_grid = np.linspace(0.0, self.x_max, self.N_x + 1)
        self.z_grid = np.linspace(0.0, 1.0, self.N_z)
        self.dm = self.m_grid[1] - self.m_grid[0]
        self.dx = self.x_grid[1] - self.x_grid[0]
        self._precompute_tables()

    def _precompute_tables(self) -> None:
        """Precompute m_z, x_z, and Ubar ratio for all (m, x, z) triples."""
        # Expand to 3D: shapes (N_m+1, 1, 1), (1, N_x+1, 1), (1, 1, N_z)
        m = self.m_grid[:, None, None]
        x = self.x_grid[None, :, None]
        z = self.z_grid[None, None, :]
        eps_buy = self.eps_buy
        eps_sell = self.eps_sell

        # Post-trade risky holding: m_z = z * (x/(1+eps_buy) + m)
        mz = z * (x / (1.0 + eps_buy) + m)

        # Post-trade cash:
        # x_z = x - (1+eps_buy)*(m_z - m)^+ + (1-eps_sell)*(m_z - m)^-
        delta = mz - m
        buy_part = np.maximum(delta, 0.0)    # buying risky: cash decreases
        sell_part = np.maximum(-delta, 0.0)  # selling risky: cash increases
        xz = x - (1.0 + eps_buy) * buy_part + (1.0 - eps_sell) * sell_part

        # Enforce admissibility (should hold by construction; clip for safety)
        mz = np.maximum(mz, 0.0)
        xz = np.maximum(xz, 0.0)

        self.mz_table = mz  # (N_m+1, N_x+1, N_z)
        self.xz_table = xz

        # Pre-trade and post-trade wealth
        w_pre = np.maximum(m + x, 1e-12)           # (N_m+1, N_x+1, 1)
        w_post = np.maximum(mz + xz, 1e-12)        # (N_m+1, N_x+1, N_z)

        self.Ubar_ratio = Ubar(w_post) / Ubar(w_pre)  # (N_m+1, N_x+1, N_z)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.N_m + 1, self.N_x + 1)


def make_grid(
    utility_label: str,
    eps_buy: float,
    eps_sell: float,
    N_m: int = 60,
    N_x: int = 60,
    N_z: int = 100,
    m_max: float | None = None,
    x_max: float | None = None,
) -> Grid:
    """
    Factory function for Grid objects.

    Domain bounds are chosen conservatively based on the utility scenario if
    not provided:
    - U1 / U3 (target w ~ 2): m_max = x_max = 4.0
    - U2 (target w < 1): m_max = x_max = 2.0

    Parameters
    ----------
    utility_label : str
        'U1', 'U2', or 'U3' — used to select default domain bounds.
    eps_buy, eps_sell : float
        Transaction costs.
    N_m, N_x : int
        Number of subintervals per axis.
    N_z : int
        Number of z control grid points.
    m_max, x_max : float, optional
        Override default domain bounds.
    """
    default_bound = 2.0 if utility_label == "U2" else 4.0
    m_max = m_max if m_max is not None else default_bound
    x_max = x_max if x_max is not None else default_bound
    return Grid(
        m_max=m_max,
        x_max=x_max,
        N_m=N_m,
        N_x=N_x,
        N_z=N_z,
        eps_buy=eps_buy,
        eps_sell=eps_sell,
    )
