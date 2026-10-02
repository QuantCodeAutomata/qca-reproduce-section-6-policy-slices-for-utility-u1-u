"""
Visualization functions for utility shapes, policy slices, and robustness tables.
All plots use matplotlib with seaborn styling.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import List, Dict

sns.set_theme(style="whitegrid", palette="deep")
COLORS = {"buy": "#2196F3", "sell": "#F44336", "notrade": "#4CAF50", "no-trade": "#4CAF50"}


def plot_utility(
    utility_fns: List,
    labels: List[str],
    w_range: tuple = (0.5, 2.5),
    title: str = "Utility Functions",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot one or more utility functions over a wealth range.

    Parameters
    ----------
    utility_fns : list of callable
    labels : list of str
    w_range : (w_min, w_max)
    title : str
    save_path : str or None
    """
    w = np.linspace(w_range[0], w_range[1], 800)
    fig, ax = plt.subplots(figsize=(8, 5))

    for fn, lbl in zip(utility_fns, labels):
        y = fn(w)
        ax.plot(w, y, linewidth=2, label=lbl)

    ax.set_xlabel("Wealth $w$", fontsize=13)
    ax.set_ylabel("Utility $U(w)$", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_policy_slice(
    diag_list: List[Dict],
    title: str = "Policy Slice",
    save_path: str | None = None,
    tol_action: float = 1e-3,
) -> plt.Figure:
    """
    Plot pi-hat vs pi for multiple wealth slices on the same figure.

    Parameters
    ----------
    diag_list : list of dicts from full_policy_diagnostics
    title : str
    save_path : str or None
    tol_action : float
        Tolerance for action shading.
    """
    fig, axes = plt.subplots(1, len(diag_list), figsize=(7 * len(diag_list), 6), squeeze=False)

    for ax, diag in zip(axes[0], diag_list):
        pi_hat = diag["pi_hat"]
        # Use stored pi_grid if available, else reconstruct
        pi_grid = diag.get("pi_grid", np.linspace(0, 1, len(pi_hat)))
        labels = diag.get("actions", diag.get("labels", []))
        wealth = diag["wealth"]

        # Shade action regions
        colors_arr = np.array([COLORS.get(lbl, "#9E9E9E") for lbl in labels])
        # Use fill_between for each connected region
        for iv in diag["intervals"]:
            mask = (pi_grid >= iv["pi_start"] - 1e-9) & (pi_grid <= iv["pi_end"] + 1e-9)
            ax.fill_between(
                pi_grid[mask],
                0, 1,
                alpha=0.15,
                color=COLORS.get(iv["action"], "#9E9E9E"),
                label=f"_{iv['action']}",   # underscore suppresses duplicate legend entries
            )

        # Identity line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="$\\hat{\\pi} = \\pi$ (no-trade)")

        # Policy curve
        ax.plot(pi_grid, pi_hat, "b-", linewidth=2.0, label="$\\hat{\\pi}(\\pi)$")

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Pre-trade $\\pi$", fontsize=12)
        ax.set_ylabel("Post-trade $\\hat{\\pi}$", fontsize=12)
        ax.set_title(f"$w = {wealth}$", fontsize=13)

        # Custom legend patches
        patches = [
            mpatches.Patch(color=COLORS["buy"], alpha=0.4, label="Buy"),
            mpatches.Patch(color=COLORS["sell"], alpha=0.4, label="Sell"),
            mpatches.Patch(color=COLORS["notrade"], alpha=0.4, label="No-trade"),
        ]
        ax.legend(handles=patches + [
            plt.Line2D([0], [0], color="k", linestyle="--", label="Identity"),
            plt.Line2D([0], [0], color="b", linewidth=2, label="$\\hat{\\pi}$"),
        ], fontsize=10, loc="upper left")

        # Annotate interval structure
        interval_str = "  ".join(
            f"{iv['action']}[{iv['pi_start']:.2f},{iv['pi_end']:.2f}]"
            for iv in diag["intervals"]
        )
        ax.set_title(f"$w={wealth}$ | {interval_str}", fontsize=10)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_robustness(
    df,
    title: str = "Robustness: Switch Locations",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot switch locations across configurations as a grouped bar / scatter.

    Parameters
    ----------
    df : pd.DataFrame
        Output of robustness_compare.
    title : str
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: interval counts
    ax = axes[0]
    configs = df["config"].tolist()
    x = np.arange(len(configs))
    w = 0.25
    ax.bar(x - w, df["n_buy"],    width=w, color=COLORS["buy"],      label="Buy intervals",      alpha=0.8)
    ax.bar(x,     df["n_notrade"], width=w, color=COLORS["no-trade"], label="No-trade intervals", alpha=0.8)
    ax.bar(x + w, df["n_sell"],   width=w, color=COLORS["sell"],     label="Sell intervals",     alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of intervals")
    ax.set_title("Interval count by configuration")
    ax.legend(fontsize=9)

    # Right: first switch location
    ax2 = axes[1]
    ax2.scatter(configs, df["first_switch_pi"], c="purple", s=80, zorder=3)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("First switching point $\\pi$")
    ax2.set_title("First switching-point location")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_kbar_profile(
    pi_grid: np.ndarray,
    kbar_all: np.ndarray,
    z_grid: np.ndarray,
    pi_indices: List[int],
    title: str = "Kbar profile at selected pi",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot Kbar(z) objective profile at selected pi values (for near-tie analysis).

    Parameters
    ----------
    pi_grid : (N_pi,)
    kbar_all : (N_pi, N_z)
    z_grid : (N_z,)
    pi_indices : list of int
        Indices into pi_grid to display.
    """
    n = len(pi_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, idx in zip(axes[0], pi_indices):
        ax.plot(z_grid, kbar_all[idx], "b-", linewidth=1.5)
        ax.set_xlabel("Control $z$", fontsize=11)
        ax.set_ylabel("$\\bar{K}(\\nu; z)$", fontsize=11)
        ax.set_title(f"$\\pi = {pi_grid[idx]:.3f}$", fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig
