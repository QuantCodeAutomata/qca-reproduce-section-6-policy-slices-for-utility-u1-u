"""
Experiment 3: Policy slices for utility U3 under 5% transaction costs.

Reproduces Figure 4 of the paper (Section 6):
- w=1.655: expected nonstandard ordering with sell interval left of buy interval,
  plus an abrupt jump near pi~0.18 (possibly between pi-hat=0 and pi-hat=1).

Numerical choices:
  alpha  : theoretical lower bound with safety margin
  domain : m_max = x_max = 4.0
  grid   : N_m = N_x = 60, N_t = 500, N_z = 200 (finer z for jump detection)
  interp : bilinear
  z tie  : largest maximising z
  no-trade tol: 1e-3
"""

import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import U3, compute_alpha_lower_bound
from src.grid import make_grid
from src.pde_solver import solve_hjb_pide
from src.policy import full_policy_diagnostics, intervals_to_dataframe
from src.plots import plot_utility, plot_policy_slice, plot_kbar_profile

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPS_BUY = 0.05
EPS_SELL = 0.05
WEALTH_SLICES = [1.655]
N_PI = 500    # finer pi-grid for jump detection
UTILITY_LABEL = "U3"

N_M = 60
N_X = 60
N_T = 500
N_Z = 200    # finer z-grid for U3's abrupt jump
M_MAX = 4.0
X_MAX = 4.0


def run_exp3(
    N_m: int = N_M,
    N_x: int = N_X,
    N_t: int = N_T,
    N_z: int = N_Z,
    m_max: float = M_MAX,
    x_max: float = X_MAX,
    suffix: str = "baseline",
) -> dict:
    """Run Experiment 3: U3 policy slice at w=1.655."""
    print(f"\n{'='*60}")
    print(f"Exp 3 — Utility U3  [{suffix}]")
    print(f"  eps={EPS_BUY:.0%}, domain=[{m_max},{x_max}]"
          f", grid=({N_m},{N_x}), N_t={N_t}, N_z={N_z}")

    alpha = compute_alpha_lower_bound(max(EPS_BUY, EPS_SELL), gamma_inf_estimate=15.0)
    print(f"  alpha = {alpha:.4f}")

    grid = make_grid(
        "U3", EPS_BUY, EPS_SELL, N_m=N_m, N_x=N_x, N_z=N_z,
        m_max=m_max, x_max=x_max
    )
    print(f"  Grid: dm={grid.dm:.4f}, dx={grid.dx:.4f}")

    t0 = time.time()
    nu0 = solve_hjb_pide(grid, U3, alpha, N_t=N_t, verbose=True, report_interval=100)
    elapsed = time.time() - t0
    print(f"  Solve time: {elapsed:.1f}s")

    pi_grid = np.linspace(0.0, 1.0, N_PI)
    results = {}
    diag_list = []

    for w in WEALTH_SLICES:
        diag = full_policy_diagnostics(
            nu0, grid, w, pi_grid,
            tol_action=1e-3, obj_tol=0.02,
            label=f"U3_w{w}_{suffix}"
        )
        diag_list.append(diag)
        results[f"w{w}"] = {
            "wealth": w,
            "n_buy": diag["n_buy_intervals"],
            "n_sell": diag["n_sell_intervals"],
            "n_notrade": diag["n_notrade_intervals"],
            "intervals": [
                {"action": iv["action"], "pi_start": iv["pi_start"], "pi_end": iv["pi_end"]}
                for iv in diag["intervals"]
            ],
            "n_switches": len(diag["switches"]),
        }
        print(f"\n  w={w}:")
        print(f"    Intervals: {intervals_to_dataframe(diag['intervals']).to_string(index=False)}")
        print(f"    Buy={diag['n_buy_intervals']}, Sell={diag['n_sell_intervals']},"
              f" NoTrade={diag['n_notrade_intervals']}, Switches={len(diag['switches'])}")

        # Check ordering: detect if any sell interval has pi_start < any buy interval pi_start
        sell_starts = [iv["pi_start"] for iv in diag["intervals"] if iv["action"] == "sell"]
        buy_starts = [iv["pi_start"] for iv in diag["intervals"] if iv["action"] == "buy"]
        reversed_order = any(
            ss < bs for ss in sell_starts for bs in buy_starts
        )
        print(f"    Reversed order (sell left of buy): {reversed_order}")
        results[f"w{w}"]["reversed_order"] = reversed_order

        # Jumps near pi~0.18
        jump_near_018 = any(abs(sw["pi"] - 0.18) < 0.05 for sw in diag["switches"])
        print(f"    Jump near pi~0.18: {jump_near_018}")
        results[f"w{w}"]["jump_near_018"] = jump_near_018

    plot_utility(
        [U3], ["U3"],
        w_range=(0.5, 2.5),
        title="Utility U3 Shape",
        save_path=str(RESULTS_DIR / f"exp3_U3_utility_{suffix}.png"),
    )
    plot_policy_slice(
        diag_list,
        title=f"Policy Slices — U3, eps={EPS_BUY:.0%} [{suffix}]",
        save_path=str(RESULTS_DIR / f"exp3_U3_policy_{suffix}.png"),
    )

    # Kbar profiles at pi values near jump location (pi~0.18 and nearby)
    for diag in diag_list:
        probe_pis = [0.05, 0.15, 0.18, 0.22, 0.30, 0.50, 0.80]
        pi_idxs = [int(np.argmin(np.abs(pi_grid - p))) for p in probe_pis]
        plot_kbar_profile(
            pi_grid, diag["kbar_all"], grid.z_grid, pi_idxs[:4],
            title=f"Kbar near jump — U3 w={diag['wealth']} [{suffix}]",
            save_path=str(RESULTS_DIR / f"exp3_U3_kbar_jump_{suffix}.png"),
        )

    meta = {
        "experiment": "exp3_U3",
        "suffix": suffix,
        "alpha": float(alpha),
        "m_max": m_max, "x_max": x_max,
        "N_m": N_m, "N_x": N_x, "N_t": N_t, "N_z": N_z,
        "eps_buy": EPS_BUY, "eps_sell": EPS_SELL,
        "elapsed_s": round(elapsed, 2),
        "results": results,
    }
    meta_path = RESULTS_DIR / f"exp3_U3_meta_{suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved: {meta_path}")

    return {"diag_list": diag_list, "meta": meta, "nu0": nu0, "grid": grid, "pi_grid": pi_grid}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    run_exp3()
    print("\nExp 3 complete.")
