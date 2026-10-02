"""
Tests for src/policy.py — policy extraction and switching-point analysis.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grid import Grid
from src.policy import (
    classify_actions, detect_intervals, detect_switches,
    full_policy_diagnostics, intervals_to_dataframe, robustness_compare,
)
from src.utils import U1
from src.pde_solver import solve_hjb_pide


@pytest.fixture
def small_grid():
    return Grid(m_max=3.0, x_max=3.0, N_m=10, N_x=10, N_z=20,
                eps_buy=0.05, eps_sell=0.05)


@pytest.fixture
def dummy_nu(small_grid):
    """Solve for a tiny number of steps to get a valid nu0."""
    return solve_hjb_pide(small_grid, U1, alpha=100.0, N_t=3)


# ---------------------------------------------------------------------------
# classify_actions
# ---------------------------------------------------------------------------

def test_classify_buy():
    """pi-hat > pi + tol -> buy."""
    pi = np.array([0.2, 0.4])
    pi_hat = np.array([0.5, 0.7])
    actions = classify_actions(pi, pi_hat, tol=1e-3)
    assert all(a == "buy" for a in actions)


def test_classify_sell():
    """pi-hat < pi - tol -> sell."""
    pi = np.array([0.6, 0.8])
    pi_hat = np.array([0.3, 0.5])
    actions = classify_actions(pi, pi_hat, tol=1e-3)
    assert all(a == "sell" for a in actions)


def test_classify_no_trade():
    """abs(pi-hat - pi) <= tol -> no-trade."""
    pi = np.array([0.5, 0.7])
    pi_hat = pi + 5e-4
    actions = classify_actions(pi, pi_hat, tol=1e-3)
    assert all(a == "notrade" for a in actions)


def test_classify_mixed():
    """Mixed case."""
    pi = np.array([0.2, 0.5, 0.8])
    pi_hat = np.array([0.5, 0.5, 0.3])
    actions = classify_actions(pi, pi_hat, tol=1e-3)
    assert actions[0] == "buy"
    assert actions[1] == "notrade"
    assert actions[2] == "sell"


# ---------------------------------------------------------------------------
# detect_intervals
# ---------------------------------------------------------------------------

def test_detect_intervals_single():
    """Single action returns one interval."""
    pi = np.linspace(0, 1, 100)
    actions = ["buy"] * 100
    ivs = detect_intervals(pi, actions)
    assert len(ivs) == 1
    assert ivs[0]["action"] == "buy"
    assert abs(ivs[0]["pi_start"]) < 1e-10
    assert abs(ivs[0]["pi_end"] - 1.0) < 1e-10


def test_detect_intervals_three():
    """Three consecutive intervals."""
    n = 300
    pi = np.linspace(0, 1, n)
    actions = (["buy"] * 100 + ["notrade"] * 100 + ["sell"] * 100)
    ivs = detect_intervals(pi, actions)
    assert len(ivs) == 3
    assert ivs[0]["action"] == "buy"
    assert ivs[1]["action"] == "notrade"
    assert ivs[2]["action"] == "sell"


def test_detect_intervals_boundaries():
    """Intervals should have contiguous pi boundaries."""
    pi = np.linspace(0, 1, 200)
    actions = ["buy"] * 100 + ["sell"] * 100
    ivs = detect_intervals(pi, actions)
    assert abs(ivs[0]["pi_start"] - 0.0) < 1e-10
    assert abs(ivs[-1]["pi_end"] - 1.0) < 1e-10
    # Consecutive intervals share boundary
    for i in range(len(ivs) - 1):
        assert abs(ivs[i]["pi_end"] - ivs[i + 1]["pi_start"]) < 1e-10


# ---------------------------------------------------------------------------
# detect_switches
# ---------------------------------------------------------------------------

def test_detect_switches_no_switch():
    """No switches for uniform action region."""
    pi = np.linspace(0, 1, 100)
    pi_hat = pi.copy()  # no-trade everywhere
    kbar_top2 = np.zeros((100, 2))
    sw = detect_switches(pi, pi_hat, kbar_top2, obj_tol=0.01)
    assert len(sw) == 0


def test_detect_switches_identifies_near_tie():
    """Should detect a switch at pi with near-equal top-2 Kbar values."""
    pi = np.linspace(0, 1, 100)
    pi_hat = np.where(pi < 0.5, 0.0, 1.0)  # jump at 0.5
    kbar_top2 = np.ones((100, 2)) * 5.0    # tie everywhere
    kbar_top2[:, 1] = 4.99                  # slight gap
    sw = detect_switches(pi, pi_hat, kbar_top2, obj_tol=0.05)
    # Should find at least one switch near pi=0.5
    assert len(sw) > 0


# ---------------------------------------------------------------------------
# full_policy_diagnostics
# ---------------------------------------------------------------------------

def test_full_diagnostics_structure(small_grid, dummy_nu):
    """full_policy_diagnostics should return expected keys."""
    pi_grid = np.linspace(0, 1, 50)
    diag = full_policy_diagnostics(dummy_nu, small_grid, wealth=1.5, pi_grid=pi_grid)
    required_keys = [
        "wealth", "pi_grid", "pi_hat", "actions",
        "intervals", "switches",
        "n_buy_intervals", "n_sell_intervals", "n_notrade_intervals",
        "kbar_all",
    ]
    for k in required_keys:
        assert k in diag, f"Missing key: {k}"


def test_full_diagnostics_pi_hat_range(small_grid, dummy_nu):
    """pi-hat should be in [0, 1]."""
    pi_grid = np.linspace(0, 1, 50)
    diag = full_policy_diagnostics(dummy_nu, small_grid, wealth=1.5, pi_grid=pi_grid)
    assert np.all(diag["pi_hat"] >= -1e-10)
    assert np.all(diag["pi_hat"] <= 1 + 1e-10)


def test_full_diagnostics_action_counts(small_grid, dummy_nu):
    """Interval counts must equal actual interval detection."""
    pi_grid = np.linspace(0, 1, 50)
    diag = full_policy_diagnostics(dummy_nu, small_grid, wealth=1.5, pi_grid=pi_grid)
    buys = sum(1 for iv in diag["intervals"] if iv["action"] == "buy")
    sells = sum(1 for iv in diag["intervals"] if iv["action"] == "sell")
    notrades = sum(1 for iv in diag["intervals"] if iv["action"] == "notrade")
    assert buys == diag["n_buy_intervals"]
    assert sells == diag["n_sell_intervals"]
    assert notrades == diag["n_notrade_intervals"]


# ---------------------------------------------------------------------------
# intervals_to_dataframe
# ---------------------------------------------------------------------------

def test_intervals_to_dataframe():
    """intervals_to_dataframe should return a DataFrame with correct columns."""
    intervals = [
        {"action": "buy", "pi_start": 0.0, "pi_end": 0.3},
        {"action": "notrade", "pi_start": 0.3, "pi_end": 0.7},
        {"action": "sell", "pi_start": 0.7, "pi_end": 1.0},
    ]
    df = intervals_to_dataframe(intervals)
    assert isinstance(df, pd.DataFrame)
    assert "action" in df.columns
    assert "pi_start" in df.columns
    assert "pi_end" in df.columns
    assert len(df) == 3


# ---------------------------------------------------------------------------
# robustness_compare
# ---------------------------------------------------------------------------

def test_robustness_compare_output():
    """robustness_compare should return a DataFrame with configuration metrics."""
    diag1 = {
        "label": "cfg1", "n_buy_intervals": 1, "n_sell_intervals": 1,
        "n_notrade_intervals": 1, "switches": [],
        "intervals": [{"action": "buy", "pi_start": 0.0, "pi_end": 0.3},
                      {"action": "notrade", "pi_start": 0.3, "pi_end": 0.7},
                      {"action": "sell", "pi_start": 0.7, "pi_end": 1.0}],
        "N_m": 60, "N_x": 60, "N_t": 500, "N_z": 120,
        "m_max": 4.0, "x_max": 4.0, "alpha": 200.0,
    }
    diag2 = dict(diag1)
    diag2["label"] = "cfg2"
    df = robustness_compare([diag1, diag2], ["cfg1", "cfg2"])
    assert isinstance(df, pd.DataFrame)
    assert "config" in df.columns
    assert len(df) == 2
