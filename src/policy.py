"""
Policy recovery, switching-point detection, and robustness diagnostics.

Key public functions
--------------------
classify_actions    : label each pi as buy / sell / notrade
detect_intervals    : list connected action intervals with shared boundaries
detect_switches     : find near-tied objective points with pi-hat jumps
full_policy_diagnostics : all-in-one diagnostics for one (nu, wealth) pair
intervals_to_dataframe  : convert intervals to DataFrame
robustness_compare  : aggregate statistics across refinement runs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict

from src.control import extract_policy_slice


# ---------------------------------------------------------------------------
# Action classification
# ---------------------------------------------------------------------------

def classify_actions(
    pi: np.ndarray,
    pi_hat: np.ndarray,
    tol: float = 1e-3,
) -> List[str]:
    """
    Classify each pre-trade pi as 'buy', 'sell', or 'notrade'.

    Parameters
    ----------
    pi : np.ndarray
        Pre-trade risky weight grid.
    pi_hat : np.ndarray
        Optimal post-trade risky weights.
    tol : float
        Tolerance: |pi_hat - pi| <= tol => notrade.

    Returns
    -------
    list of str
        Action labels ('buy', 'sell', 'notrade').
    """
    pi = np.asarray(pi, dtype=float)
    pi_hat = np.asarray(pi_hat, dtype=float)
    diff = pi_hat - pi
    return list(np.where(diff > tol, "buy", np.where(diff < -tol, "sell", "notrade")))


# ---------------------------------------------------------------------------
# Interval detection (shared boundary convention)
# ---------------------------------------------------------------------------

def detect_intervals(
    pi_grid: np.ndarray,
    actions: List[str],
) -> List[Dict]:
    """
    List connected action intervals along the pi axis.

    Boundaries are shared: pi_end of interval k equals pi_start of interval k+1.
    The last interval ends at pi_grid[-1].

    Parameters
    ----------
    pi_grid : np.ndarray, shape (N,)
    actions : list of str, length N
        Each element is 'buy', 'sell', or 'notrade'.

    Returns
    -------
    List of dicts with keys: 'action', 'pi_start', 'pi_end', 'pi_mid', 'count'.
    """
    n = len(actions)
    if n == 0:
        return []

    intervals = []
    current = actions[0]
    start_idx = 0

    for i in range(1, n + 1):
        if i == n or actions[i] != current:
            # Determine pi_end: the start of the next interval for sharing
            if i < n:
                pi_end = float(pi_grid[i])     # start of next = end of this
            else:
                pi_end = float(pi_grid[n - 1]) # last point

            pi_start = float(pi_grid[start_idx])
            intervals.append({
                "action": current,
                "pi_start": pi_start,
                "pi_end": pi_end,
                "pi_mid": 0.5 * (pi_start + pi_end),
                "count": i - start_idx,
            })

            if i < n:
                current = actions[i]
                start_idx = i

    return intervals


# Keep backward-compatible alias
def action_intervals(pi_grid: np.ndarray, labels: np.ndarray) -> List[Dict]:
    """Alias for detect_intervals (backward compatibility)."""
    return detect_intervals(pi_grid, list(labels))


# ---------------------------------------------------------------------------
# Switching-point detection (near-tied objective + pi-hat jump)
# ---------------------------------------------------------------------------

def detect_switches(
    pi: np.ndarray,
    pi_hat: np.ndarray,
    kbar_top2: np.ndarray,
    obj_tol: float = 0.01,
    pi_hat_jump_tol: float = 0.1,
) -> List[Dict]:
    """
    Detect switching points where competing z-values are nearly tied
    AND pi_hat shows an abrupt local jump.

    A switch is flagged at index i when both conditions hold:
    (a) (kbar_top2[i,0] - kbar_top2[i,1]) / max(|kbar_top2[i,0]|, 1e-12) < obj_tol
    (b) |pi_hat[i+1] - pi_hat[i-1]| > pi_hat_jump_tol  (centred difference)

    Parameters
    ----------
    pi : np.ndarray, shape (N,)
    pi_hat : np.ndarray, shape (N,)
    kbar_top2 : np.ndarray, shape (N, 2)
        Top-2 Kbar values at each pi point (descending order).
    obj_tol : float
        Relative tolerance for near-tie classification.
    pi_hat_jump_tol : float
        Minimum centred pi-hat difference to flag a jump.

    Returns
    -------
    List of dicts, one per switch, with keys:
        'pi', 'pi_hat', 'kbar_gap', 'relative_gap', 'pi_hat_jump'.
    """
    pi = np.asarray(pi, dtype=float)
    pi_hat = np.asarray(pi_hat, dtype=float)
    kbar_top2 = np.asarray(kbar_top2, dtype=float)

    N = len(pi)
    gap = kbar_top2[:, 0] - kbar_top2[:, 1]
    ref = np.maximum(np.abs(kbar_top2[:, 0]), 1e-12)
    near_tied = (gap / ref) < obj_tol

    switches = []
    for i in range(N):
        if not near_tied[i]:
            continue
        lo = max(0, i - 1)
        hi = min(N - 1, i + 1)
        jump = abs(pi_hat[hi] - pi_hat[lo])
        if jump > pi_hat_jump_tol:
            switches.append({
                "pi": float(pi[i]),
                "pi_hat": float(pi_hat[i]),
                "kbar_gap": float(gap[i]),
                "relative_gap": float(gap[i] / ref[i]),
                "pi_hat_jump": float(jump),
            })

    return switches


# ---------------------------------------------------------------------------
# Legacy switching-point finder (uses full kbar_all profile)
# ---------------------------------------------------------------------------

def find_switching_points(
    pi_grid: np.ndarray,
    kbar_all: np.ndarray,
    pi_hat: np.ndarray,
    obj_tol: float = 0.01,
    pi_hat_tol: float = 0.05,
) -> List[Dict]:
    """
    Detect near-tie switching points using the full Kbar profile.

    Parameters
    ----------
    pi_grid, kbar_all, pi_hat : as returned by extract_policy_slice / full_policy_diagnostics
    obj_tol : relative gap tolerance
    pi_hat_tol : minimum pi-hat jump to declare a switch

    Returns
    -------
    List of dicts with diagnostic information.
    """
    N_pi = len(pi_grid)
    N_z = kbar_all.shape[1] if kbar_all.ndim == 2 else 1
    switches = []

    for k in range(N_pi):
        vals = kbar_all[k]
        order = np.argsort(vals)[::-1]
        best_val = vals[order[0]]
        second_val = vals[order[1]] if N_z > 1 else best_val
        gap = abs(best_val - second_val)
        ref = max(abs(best_val), 1e-12)

        if gap / ref < obj_tol:
            lo = max(0, k - 1)
            hi = min(N_pi - 1, k + 1)
            jump = abs(pi_hat[hi] - pi_hat[lo])
            switches.append({
                "pi": float(pi_grid[k]),
                "pi_hat": float(pi_hat[k]),
                "top2_kbar_gap": gap,
                "relative_gap": gap / ref,
                "pi_hat_jump": float(jump),
                "best_z_idx": int(order[0]),
                "second_z_idx": int(order[1]) if N_z > 1 else int(order[0]),
            })

    return switches


# ---------------------------------------------------------------------------
# Full diagnostics entry point
# ---------------------------------------------------------------------------

def full_policy_diagnostics(
    nu: np.ndarray,
    grid,
    wealth: float,
    pi_grid: np.ndarray,
    tol_action: float = 1e-3,
    obj_tol: float = 0.01,
    label: str = "",
    ntrade_dz_frac: float = 0.52,
) -> Dict:
    """
    Complete policy diagnostics for one (nu, wealth) pair.

    Parameters
    ----------
    nu : np.ndarray, shape (N_m+1, N_x+1)
    grid : Grid
    wealth : float
        Target wealth slice.
    pi_grid : np.ndarray
        Dense pre-trade risky weight grid in [0, 1].
    tol_action : float
        No-trade tolerance on |pi_hat - pi|.
    obj_tol : float
        Near-tie tolerance for switching-point detection.
    label : str
    ntrade_dz_frac : float
        Passed to extract_policy_slice: z-proximity fraction for no-trade
        suppression of staircase artifacts (default 0.52, just over half dz).

    Returns
    -------
    dict with keys:
        'label', 'wealth', 'pi_grid', 'pi_hat', 'z_hat', 'actions', 'labels',
        'intervals', 'switches', 'top2_kbar', 'kbar_all',
        'n_buy_intervals', 'n_sell_intervals', 'n_notrade_intervals'.
    """
    pi_hat, z_hat, top2_kbar, kbar_all = extract_policy_slice(
        nu, grid, wealth, pi_grid, ntrade_dz_frac=ntrade_dz_frac
    )
    actions = classify_actions(pi_grid, pi_hat, tol=tol_action)
    intervals = detect_intervals(pi_grid, actions)
    switches = find_switching_points(pi_grid, kbar_all, pi_hat, obj_tol=obj_tol)

    n_buy = sum(1 for iv in intervals if iv["action"] == "buy")
    n_sell = sum(1 for iv in intervals if iv["action"] == "sell")
    n_nt = sum(1 for iv in intervals if iv["action"] == "notrade")

    return {
        "label": label,
        "wealth": wealth,
        "pi_grid": pi_grid,
        "pi_hat": pi_hat,
        "z_hat": z_hat,
        "actions": actions,
        "labels": actions,       # backward-compat alias
        "intervals": intervals,
        "switches": switches,
        "top2_kbar": top2_kbar,
        "kbar_all": kbar_all,
        "n_buy_intervals": n_buy,
        "n_sell_intervals": n_sell,
        "n_notrade_intervals": n_nt,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def intervals_to_dataframe(intervals: List[Dict]) -> pd.DataFrame:
    """Convert list of interval dicts to a formatted DataFrame."""
    if not intervals:
        return pd.DataFrame(columns=["action", "pi_start", "pi_end"])
    rows = [
        {
            "action": iv.get("action", ""),
            "pi_start": iv.get("pi_start", np.nan),
            "pi_end": iv.get("pi_end", np.nan),
            "pi_mid": iv.get("pi_mid", np.nan),
            "count": iv.get("count", np.nan),
        }
        for iv in intervals
    ]
    return pd.DataFrame(rows)


def robustness_compare(
    results_list: List[Dict],
    config_labels: List[str],
) -> pd.DataFrame:
    """
    Compare policy interval structures across different solver configurations.

    Parameters
    ----------
    results_list : list of dicts from full_policy_diagnostics
    config_labels : list of str

    Returns
    -------
    pd.DataFrame — one row per configuration.
    """
    rows = []
    for cfg, diag in zip(config_labels, results_list):
        first_switch_pi = (
            diag["switches"][0]["pi"] if diag.get("switches") else np.nan
        )
        rows.append({
            "config": cfg,
            "wealth": diag.get("wealth", np.nan),
            "n_buy": diag.get("n_buy_intervals", 0),
            "n_sell": diag.get("n_sell_intervals", 0),
            "n_notrade": diag.get("n_notrade_intervals", 0),
            "n_intervals_total": len(diag.get("intervals", [])),
            "first_switch_pi": first_switch_pi,
            "n_switches": len(diag.get("switches", [])),
            "N_m": diag.get("N_m", np.nan),
            "N_x": diag.get("N_x", np.nan),
            "N_t": diag.get("N_t", np.nan),
            "N_z": diag.get("N_z", np.nan),
            "m_max": diag.get("m_max", np.nan),
            "x_max": diag.get("x_max", np.nan),
            "alpha": diag.get("alpha", np.nan),
        })
    return pd.DataFrame(rows)
