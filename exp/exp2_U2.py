"""
Experiment 2: Policy slices for utility U2 under 2% transaction costs.

Reproduces Figure 3 of the paper (Section 6):
- w=0.83:  expected buy-no-trade-sell + extra no-trade at high pi
- w=0.915: expected two buying intervals alternating with two no-trade intervals

Numerical choices:
  alpha  : theoretical lower bound with safety margin
  domain : m_max = x_max = 2.0 (covers low-wealth target slices)
  grid   : N_m = N_x = 60, N_t = 500, N_z = 120
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

from src.utils import U2, compute_alpha_lower_bound
from src.grid import make_grid
from src.pde_solver import solve_hjb_pide
from src.policy import full_policy_diagnostics, intervals_to_dataframe
from src.plots import plot_utility, plot_policy_slice, plot_kbar_profile

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPS_BUY = 0.02
EPS_SELL = 0.02
WEALTH_SLICES = [0.83, 0.915]
N_PI = 300
UTILITY_LABEL = "U2"

N_M = 60
N_X = 60
N_T = 500
N_Z = 120
M_MAX = 2.0
X_MAX = 2.0


def run_exp2(
    N_m: int = N_M,
    N_x: int = N_X,
    N_t: int = N_T,
    N_z: int = N_Z,
    m_max: float = M_MAX,
    x_max: float = X_MAX,
    suffix: str = "baseline",
) -> dict:
    """Run Experiment 2: U2 policy slices."""
    print(f"\n{'='*60}")
    print(f"Exp 2 — Utility U2  [{suffix}]")
    print(f"  eps={EPS_BUY:.0%}, domain=[{m_max},{x_max}]"
          f", grid=({N_m},{N_x}), N_t={N_t}, N_z={N_z}")

    alpha = compute_alpha_lower_bound(max(EPS_BUY, EPS_SELL), gamma_inf_estimate=15.0)
    print(f"  alpha = {alpha:.4f}")

    grid = make_grid(
        "U2", EPS_BUY, EPS_SELL, N_m=N_m, N_x=N_x, N_z=N_z,
        m_max=m_max, x_max=x_max
    )
    print(f"  Grid: dm={grid.dm:.4f}, dx={grid.dx:.4f}")

    t0 = time.time()
    nu0 = solve_hjb_pide(grid, U2, alpha, N_t=N_t, verbose=True, report_interval=100)
    elapsed = time.time() - t0
    print(f"  Solve time: {elapsed:.1f}s")

    pi_grid = np.linspace(0.0, 1.0, N_PI)
    results = {}
    diag_list = []

    for w in WEALTH_SLICES:
        diag = full_policy_diagnostics(
            nu0, grid, w, pi_grid,
            tol_action=1e-3, obj_tol=0.02,
            label=f"U2_w{w}_{suffix}"
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

    plot_utility(
        [U2], ["U2"],
        w_range=(0.4, 2.5),
        title="Utility U2 Shape",
        save_path=str(RESULTS_DIR / f"exp2_U2_utility_{suffix}.png"),
    )
    plot_policy_slice(
        diag_list,
        title=f"Policy Slices — U2, eps={EPS_BUY:.0%} [{suffix}]",
        save_path=str(RESULTS_DIR / f"exp2_U2_policy_{suffix}.png"),
    )

    for diag in diag_list:
        if diag["switches"]:
            sw_pis = [sw["pi"] for sw in diag["switches"][:3]]
            pi_idxs = [int(np.argmin(np.abs(pi_grid - p))) for p in sw_pis]
            plot_kbar_profile(
                pi_grid, diag["kbar_all"], grid.z_grid, pi_idxs,
                title=f"Kbar profile — U2 w={diag['wealth']} [{suffix}]",
                save_path=str(RESULTS_DIR / f"exp2_U2_kbar_w{diag['wealth']}_{suffix}.png"),
            )

    meta = {
        "experiment": "exp2_U2",
        "suffix": suffix,
        "alpha": float(alpha),
        "m_max": m_max, "x_max": x_max,
        "N_m": N_m, "N_x": N_x, "N_t": N_t, "N_z": N_z,
        "eps_buy": EPS_BUY, "eps_sell": EPS_SELL,
        "elapsed_s": round(elapsed, 2),
        "results": results,
    }
    meta_path = RESULTS_DIR / f"exp2_U2_meta_{suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved: {meta_path}")

    return {"diag_list": diag_list, "meta": meta, "nu0": nu0, "grid": grid, "pi_grid": pi_grid}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    run_exp2()
    print("\nExp 2 complete.")
