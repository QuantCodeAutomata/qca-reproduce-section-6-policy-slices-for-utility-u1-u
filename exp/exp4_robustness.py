"""
Experiment 4: Numerical robustness study for switching-point stability.

Systematically varies:
  (a) time grid N_t
  (b) spatial grid (N_m, N_x)
  (c) domain size (m_max, x_max)
  (d) z-search density N_z
  (e) alpha value

For each configuration, re-runs the relevant utility scenario and records
the interval counts, switch locations, and objective gaps.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import U1, U2, U3, compute_alpha_lower_bound
from src.grid import make_grid
from src.pde_solver import solve_hjb_pide
from src.policy import full_policy_diagnostics, robustness_compare
from src.plots import plot_robustness

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_PI = 200   # pi-grid density for robustness runs (coarser for speed)


def run_scenario(
    utility_fn,
    utility_label: str,
    wealth: float,
    eps_buy: float,
    eps_sell: float,
    N_m: int,
    N_x: int,
    N_t: int,
    N_z: int,
    m_max: float,
    x_max: float,
    alpha: float | None = None,
) -> dict:
    """Run a single solver configuration and return policy diagnostics."""
    if alpha is None:
        alpha = compute_alpha_lower_bound(max(eps_buy, eps_sell), gamma_inf_estimate=15.0)

    grid = make_grid(
        utility_label, eps_buy, eps_sell, N_m=N_m, N_x=N_x, N_z=N_z,
        m_max=m_max, x_max=x_max
    )
    nu0 = solve_hjb_pide(grid, utility_fn, alpha, N_t=N_t, verbose=False)
    pi_grid = np.linspace(0.0, 1.0, N_PI)
    diag = full_policy_diagnostics(
        nu0, grid, wealth, pi_grid,
        tol_action=1e-3, obj_tol=0.02,
    )
    diag["alpha"] = alpha
    diag["N_m"] = N_m
    diag["N_x"] = N_x
    diag["N_t"] = N_t
    diag["N_z"] = N_z
    diag["m_max"] = m_max
    diag["x_max"] = x_max
    return diag


def robustness_suite_U1_w196() -> pd.DataFrame:
    """
    Robustness sweep for U1, w=1.96 (the alternating no-trade/sell case).
    """
    print("\n--- Robustness: U1 w=1.96 ---")
    configs = [
        # (label, N_m, N_x, N_t, N_z, m_max, x_max)
        ("baseline: 60x60,Nt=500,Nz=120",   60, 60, 500, 120, 4.0, 4.0),
        ("finer-t: Nt=800",                  60, 60, 800, 120, 4.0, 4.0),
        ("finer-xy: 80x80",                  80, 80, 500, 120, 4.0, 4.0),
        ("larger-domain: 5x5",               60, 60, 500, 120, 5.0, 5.0),
        ("finer-z: Nz=200",                  60, 60, 500, 200, 4.0, 4.0),
    ]

    results_list = []
    config_labels = []
    for (label, N_m, N_x, N_t, N_z, m_max, x_max) in configs:
        t0 = time.time()
        print(f"  Config: {label} ...", end=" ", flush=True)
        diag = run_scenario(
            U1, "U1", 1.96, 0.05, 0.05,
            N_m, N_x, N_t, N_z, m_max, x_max
        )
        diag["label"] = label
        results_list.append(diag)
        config_labels.append(label)
        elapsed = time.time() - t0
        print(f"buy={diag['n_buy_intervals']}, sell={diag['n_sell_intervals']},"
              f" notrade={diag['n_notrade_intervals']}, switches={len(diag['switches'])}"
              f" ({elapsed:.1f}s)")

    df = robustness_compare(results_list, config_labels)
    return df, results_list


def robustness_suite_U2_w0915() -> pd.DataFrame:
    """
    Robustness sweep for U2, w=0.915 (alternating buy/no-trade case).
    """
    print("\n--- Robustness: U2 w=0.915 ---")
    configs = [
        ("baseline: 60x60,Nt=500,Nz=120",   60, 60, 500, 120, 2.0, 2.0),
        ("finer-t: Nt=800",                  60, 60, 800, 120, 2.0, 2.0),
        ("finer-xy: 80x80",                  80, 80, 500, 120, 2.0, 2.0),
        ("larger-domain: 3x3",               60, 60, 500, 120, 3.0, 3.0),
        ("finer-z: Nz=200",                  60, 60, 500, 200, 2.0, 2.0),
    ]

    results_list = []
    config_labels = []
    for (label, N_m, N_x, N_t, N_z, m_max, x_max) in configs:
        t0 = time.time()
        print(f"  Config: {label} ...", end=" ", flush=True)
        diag = run_scenario(
            U2, "U2", 0.915, 0.02, 0.02,
            N_m, N_x, N_t, N_z, m_max, x_max
        )
        diag["label"] = label
        results_list.append(diag)
        config_labels.append(label)
        elapsed = time.time() - t0
        print(f"buy={diag['n_buy_intervals']}, sell={diag['n_sell_intervals']},"
              f" notrade={diag['n_notrade_intervals']}, switches={len(diag['switches'])}"
              f" ({elapsed:.1f}s)")

    df = robustness_compare(results_list, config_labels)
    return df, results_list


def robustness_suite_U3_w1655() -> pd.DataFrame:
    """
    Robustness sweep for U3, w=1.655 (order reversal / jump case).
    """
    print("\n--- Robustness: U3 w=1.655 ---")
    configs = [
        ("baseline: 60x60,Nt=500,Nz=200",   60, 60, 500, 200, 4.0, 4.0),
        ("finer-t: Nt=800",                  60, 60, 800, 200, 4.0, 4.0),
        ("finer-xy: 80x80",                  80, 80, 500, 200, 4.0, 4.0),
        ("larger-domain: 5x5",               60, 60, 500, 200, 5.0, 5.0),
        ("finer-z: Nz=300",                  60, 60, 500, 300, 4.0, 4.0),
    ]

    results_list = []
    config_labels = []
    for (label, N_m, N_x, N_t, N_z, m_max, x_max) in configs:
        t0 = time.time()
        print(f"  Config: {label} ...", end=" ", flush=True)
        diag = run_scenario(
            U3, "U3", 1.655, 0.05, 0.05,
            N_m, N_x, N_t, N_z, m_max, x_max
        )
        diag["label"] = label
        results_list.append(diag)
        config_labels.append(label)
        elapsed = time.time() - t0
        print(f"buy={diag['n_buy_intervals']}, sell={diag['n_sell_intervals']},"
              f" notrade={diag['n_notrade_intervals']}, switches={len(diag['switches'])}"
              f" ({elapsed:.1f}s)")

    df = robustness_compare(results_list, config_labels)
    return df, results_list


def run_exp4() -> None:
    """Run the full robustness study and save results."""
    print(f"\n{'='*60}")
    print("Exp 4 — Numerical Robustness Study")

    all_dfs = {}
    all_results = {}

    df_U1, res_U1 = robustness_suite_U1_w196()
    df_U2, res_U2 = robustness_suite_U2_w0915()
    df_U3, res_U3 = robustness_suite_U3_w1655()

    all_dfs["U1_w196"] = df_U1
    all_dfs["U2_w0915"] = df_U2
    all_dfs["U3_w1655"] = df_U3

    # Print summary tables
    for scenario, df in all_dfs.items():
        print(f"\nRobustness table — {scenario}:")
        print(df.to_string(index=False))

    # Save tables
    for scenario, df in all_dfs.items():
        csv_path = RESULTS_DIR / f"exp4_robustness_{scenario}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Plots
    for scenario, (df, res_list) in zip(
        ["U1_w196", "U2_w0915", "U3_w1655"],
        [(df_U1, res_U1), (df_U2, res_U2), (df_U3, res_U3)]
    ):
        plot_robustness(
            df,
            title=f"Robustness: {scenario}",
            save_path=str(RESULTS_DIR / f"exp4_robustness_{scenario}.png"),
        )

    # JSON meta
    meta = {
        "experiment": "exp4_robustness",
        "scenarios": {
            s: df.to_dict(orient="records")
            for s, df in all_dfs.items()
        }
    }
    with open(RESULTS_DIR / "exp4_robustness_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nExp 4 complete.")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    run_exp4()
