# Reproducing Section 6 Policy Slices — HJB/PIDE with Non-Concave Utilities

Numerical reproduction of Figure 2, 3, and 4 from the paper using a custom
IMEX finite-difference solver for the reduced finite-horizon HJB/PIDE with
proportional transaction costs and Cox/Poisson trading opportunities.

## Research Objectives

| Experiment | Utility | Costs | Wealth slices | Key claim |
|------------|---------|-------|---------------|-----------|
| Exp 1 | U1 | 5% | 1.51, 1.96 | Conventional buy/no-trade/sell at w=1.51; no buying at w=1.96 |
| Exp 2 | U2 | 2% | 0.83, 0.915 | Conventional at w=0.83; buy-dominated disconnected intervals at w=0.915 |
| Exp 3 | U3 | 5% | 1.655 | Reversed ordering (sell left of buy); abrupt jump near π≈0.18 |
| Exp 4 | U1–U3 | — | — | Systematic robustness study across grid/domain/z-resolution configurations |

## Model

- **Utilities** (non-concave):
  - U1(w) = −1/w + 6·Φ((w−1.2)/0.1) + 3.5·Φ((w−1.8)/0.2)
  - U2(w) = −1/w² + 8·Φ((w−1.8)/0.15)
  - U3(w) = −1/w + 6·Φ((w−1.6)/0.02) + 12·Φ((w−2.1)/0.05)
- **Normalisation weight**: Ū(w) = 1 + w² + w⁻²
- **Parameters**: μ=0.06, σ=0.35, λ=20, T=1, r=0
- **Transaction costs**: symmetric (ε_buy = ε_sell)
- **State variables**: m (risky holdings), x (cash); w = m + x; π = m/w

## Numerical Method

- **Formulation**: reduced normalised value function v = V/Ū(w)
- **PDE**: backward parabolic with nonlocal supremum (impulse control)
- **Discretisation**: IMEX (implicit local diffusion in m via Thomas algorithm; explicit nonlocal Kbar)
- **Spatial grid**: uniform on [0, m_max] × [0, x_max]
- **Baseline settings**: 60×60 grid, 500 time steps, 120–200 z-search points

## Repository Structure

```
src/
├── utils.py          # Utilities U1/U2/U3, Ūbar, PDE coefficients, α lower bound
├── grid.py           # Grid construction, precomputed post-trade state tables
├── interpolation.py  # Vectorised bilinear interpolation
├── control.py        # Kbar evaluation, sup_Kbar with z-optimisation
├── pde_solver.py     # IMEX backward solver, terminal condition
├── policy.py         # Policy recovery (π → π̂), switching-point diagnostics
└── plots.py          # All matplotlib figure generation

exp/
├── exp1_U1.py        # Figure 2 reproduction (U1, 5% costs)
├── exp2_U2.py        # Figure 3 reproduction (U2, 2% costs)
├── exp3_U3.py        # Figure 4 reproduction (U3, 5% costs)
└── exp4_robustness.py # Systematic refinement study

tests/
├── test_utils.py     # Mathematical property tests for U1/U2/U3, Ūbar, coefficients
├── test_grid.py      # Grid construction and intervention-map tests
├── test_interpolation.py  # Bilinear interpolation correctness
├── test_control.py   # Kbar operator properties
├── test_solver.py    # IMEX solver stability and terminal condition
└── test_policy.py    # Policy recovery and interval detection

data/                 # Empty (no external market data required)
results/              # Generated figures, CSVs, JSON metadata, RESULTS.md
```

## Setup

```bash
pip install numpy scipy matplotlib pandas
```

All required packages are standard. No external API keys are needed.

## Running Experiments

```bash
# Individual experiments
python exp/exp1_U1.py
python exp/exp2_U2.py
python exp/exp3_U3.py
python exp/exp4_robustness.py

# Full test suite (69 tests)
python -m pytest tests/ -v
```

Results (figures + CSVs + metadata) are saved to `results/`.

## Key Results

See [`results/RESULTS.md`](results/RESULTS.md) for full quantitative results and tables.

Summary:

| Scenario | Key Claim | Status |
|----------|-----------|--------|
| U1, w=1.96 | No buying; alternating sell/no-trade | ✅ |
| U2, w=0.83 | Conventional buy-no-trade-sell | ✅ |
| U3, w=1.655 | Reversed ordering + jump near π≈0.18 | ✅ |
| Robustness | Qualitative structure stable under refinement | ✅ |

## Documented Numerical Choices

| Choice | Value | Rationale |
|--------|-------|-----------|
| α | 148.11 (5% costs), 141.92 (2% costs) | 1.2× theoretical lower bound |
| Domain (U1, U3) | [0,4] × [0,4] | Covers target w with margin |
| Domain (U2) | [0,2] × [0,2] | Covers low-wealth targets |
| IMEX split | Implicit local m-operator; explicit Kbar | Standard for this PDE class |
| Interpolation | Vectorised bilinear on regular grid | Monotone, memory-efficient |
| Tie-breaking | Largest maximising z | Paper-consistent (Appendix A.1) |
| No-trade tolerance | 1e-3 in \|π̂−π\| | Documented equality criterion |
| Upper boundaries | Neumann (zero-flux) | Conservative; domain large enough |
