# AGENTS.md — Repository Knowledge Base

This file captures persistent knowledge about the codebase for future agent sessions.

---

## Project Overview

Numerical reproduction of Section 6 policy-slice figures from a paper on optimal
portfolio management with non-concave utilities and proportional transaction costs
under Cox/Poisson trading opportunities.

**No external market data is used.** The entire project is a custom IMEX PDE solver
for the reduced finite-horizon HJB/PIDE.

---

## Repository Structure

```
src/          Core implementation (models, PDE, control, policy)
exp/          Experiment scripts (exp1–exp4)
tests/        pytest tests (69 tests, all must pass)
data/         Empty (no empirical data)
results/      Output: figures (.png), CSVs, JSON metadata, RESULTS.md
```

---

## Key Design Decisions

### 1. Normalisation — No `exp(alpha*t)` Prefactor

The solver works with `v = V/Ūbar(w)` directly (NOT `nu = exp(alpha*t)*v`).
This choice prevents catastrophic cancellation from the O(exp(148×1)) = O(10^64)
terminal condition. The `alpha` parameter is computed from the theoretical lower
bound but does **not** appear in the IMEX matrix coefficients (`pde_solver.py`).

### 2. IMEX Splitting

- **Implicit**: local diffusion/drift in `m` (tridiagonal Thomas solve per x-column)
- **Explicit**: nonlocal supremum Kbar(v) from the previous time step
- Matrices are prebuilt once (`build_tridiagonal_matrices`) and reused every step

### 3. No-Trade Staircase Fix

Because `z_nt` (the identity control) rarely falls exactly on the discrete `z_grid`,
the argmax can oscillate between neighboring `z` values, producing a staircase in
`pi_hat`. The fix in `control.extract_policy_slice` checks whether `|z_hat - z_nt|
<= 0.52 * dz` and if so forces `pi_hat = pi`. The parameter `ntrade_dz_frac=0.52`
is documented and passed explicitly.

### 4. Tie-Breaking

The argmax over `z` uses the **largest** maximising `z` (paper-consistent with
Appendix A.1). Implemented by reversing the `kbar_vals` array and taking
`argmax` of the reversed slice.

### 5. Terminal Condition

`v(T, m, x) = U_i(x + (1 - eps_sell)*m) / Ūbar(x + m)` — the normalised
liquidation payoff. Implemented in `pde_solver.apply_terminal_condition`.

---

## Module Map

| File | Purpose |
|------|---------|
| `src/utils.py` | U1/U2/U3, Ūbar/dŪbar/d²Ūbar, μ̃, γ, `gtilde`, `compute_alpha_lower_bound` |
| `src/grid.py` | `Grid` dataclass: m/x/z grids, precomputed `mz_table`, `xz_table`, `Ubar_ratio` |
| `src/interpolation.py` | `bilinear_interp_batch` (single points), `bilinear_interp_3d` (full table) |
| `src/control.py` | `post_trade_states`, `sup_Kbar`, `compute_sup_Kbar`, `extract_policy_slice`, `classify_actions` |
| `src/pde_solver.py` | `build_tridiagonal_m`, `build_tridiagonal_matrices`, `backward_step`, `apply_terminal_condition`, `solve_hjb_pide` |
| `src/policy.py` | `classify_actions`, `detect_intervals`, `detect_switches`, `full_policy_diagnostics`, `intervals_to_dataframe`, `robustness_compare` |
| `src/plots.py` | All matplotlib figure generation (`plot_utility`, `plot_policy_slices`, `plot_kbar_profile`, `plot_robustness_comparison`) |

---

## Experiment Configurations

| Exp | Utility | eps | m_max / x_max | N_m | N_x | N_t | N_z | alpha |
|-----|---------|-----|---------------|-----|-----|-----|-----|-------|
| 1 | U1 | 5% | 4.0 | 60 | 60 | 500 | 120 | 148.11 |
| 2 | U2 | 2% | 2.0 | 60 | 60 | 500 | 120 | 141.92 |
| 3 | U3 | 5% | 4.0 | 60 | 60 | 500 | 200 | 148.11 |
| 4 | U1–U3 | both | varies | varies | varies | varies | varies | as above |

---

## Test Suite (69 tests)

```
tests/test_utils.py       — Mathematical correctness of utilities and coefficients
tests/test_grid.py        — Grid construction, post-trade state maps, Ubar_ratio
tests/test_interpolation.py — Bilinear interpolation accuracy
tests/test_control.py     — Kbar properties, action classification
tests/test_solver.py      — Solver stability, terminal condition, IMEX operator
tests/test_policy.py      — Policy recovery, interval detection, switch detection
```

Run: `python -m pytest tests/ -v`

All 69 tests must pass. Never modify test assertions to mask failures.

---

## Results

All experiment outputs go to `results/`. Do not commit raw API data.
Key results:

| Scenario | Key claim | Status |
|----------|-----------|--------|
| U1, w=1.96 | No buying; alternating sell/no-trade | ✅ |
| U2, w=0.83 | Conventional buy-no-trade-sell | ✅ |
| U3, w=1.655 | Reversed ordering; jump near π≈0.18 | ✅ |
| Robustness | Qualitative structure stable | ✅ |

---

## Common Pitfalls

1. **Do not add `exp(alpha*T)` to the terminal condition** — the solver uses `v = V/Ūbar` directly.
2. **Domain must be large enough**: for U1/U3 use `m_max = x_max = 4.0`; for U2 use 2.0.
3. **z-grid size matters for U3**: use `N_z = 200` (coarser grids miss the narrow sell-to-buy jump).
4. **`bilinear_interp_3d` takes the full 3D table** `(N_m+1, N_x+1, N_z)` as input — not per-point.
5. **Import paths**: always use `from src.xxx import yyy` (the `src` package).
6. **`scipy.sparse.linalg.spsolve`** is used for the per-column tridiagonal solve; the matrices are prebuilt as CSC for efficiency.
