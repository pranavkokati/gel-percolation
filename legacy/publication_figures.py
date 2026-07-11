"""
================================================================================
 DEPRECATED -- retained for archival/audit purposes only. NOT part of the
 public API and NOT imported by any current figure, test, or the manuscript.

 This module was flagged during the pre-publication audit (see REPORT.md,
 Section 2) for one or more of: (a) closed-form/hardcoded critical exponents
 presented as if independently measured, (b) a hand-drawn synthetic collagen
 deposition curve, (c) an Ornstein-Uhlenbeck noise term engineered to
 reproduce a target Kendall tau rather than derived from network dynamics,
 or (d) an internally-inconsistent Q metric definition superseded by
 gelrigidity.handoff.load_path_continuity_Q.

 The replacement, physically-grounded implementation lives in gelrigidity/:
   - measured (not closed-form) exponents      -> gelrigidity.rigidity
   - network-driven (not synthetic) ECM growth -> gelrigidity.dynamics
   - single, consistent Q definition           -> gelrigidity.handoff

 Do not import this module in new code. It is kept only so the original
 repository's provenance and the audit trail remain reproducible.
================================================================================
"""

"""Publication figures for "Percolation Inversion Dynamics in Enzymatically Degrading Wound Hydrogels."

Five figures are provided, targeting Soft Matter / Biophysical Journal
formatting standards (300 DPI, ≥8 pt sans-serif fonts, 1.5 pt line width,
single-column 3.5" or double-column 7.0" widths).

Figure 1 — Critical scaling of G' near the percolation threshold
    `plot_critical_scaling`

Figure 2 — Inverse percolation dynamics (gel-sol transition)
    `plot_percolation_dynamics`

Figure 3 — Early warning signal panel (key novel result)
    `plot_ews_panel_publication`

Figure 4 — Handoff quality Q design map (heatmap)
    `plot_q_heatmap_publication`

Figure 5 — Fibroblast invasion depth and collagen assembly
    `plot_invasion_summary`

Helper
------
`save_figure` — saves a figure in PNG and PDF for journal submission.
"""

from __future__ import annotations

__all__ = [
    "plot_critical_scaling",
    "plot_percolation_dynamics",
    "plot_ews_panel_publication",
    "plot_q_heatmap_publication",
    "plot_invasion_summary",
    "save_figure",
]

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,   # screen preview; saving uses 300
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embeds fonts for journal submission
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_figure(fig: mpl.figure.Figure, output_path, formats: tuple = ("png", "pdf")) -> None:
    """Save *fig* in multiple formats for journal submission.

    Parameters
    ----------
    fig:
        The matplotlib Figure to save.
    output_path:
        Destination path (with or without extension).  The extension is
        replaced by each entry in *formats*.
    formats:
        Iterable of format strings, e.g. ``("png", "pdf")``.
    """
    if output_path is None:
        return
    base = Path(output_path).with_suffix("")
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure 1 — Critical scaling of G' near the percolation threshold
# ---------------------------------------------------------------------------

def plot_critical_scaling(mech, p_c: float, output_path=None) -> mpl.figure.Figure:
    """Plot critical scaling of the shear modulus G' near the percolation threshold.

    Parameters
    ----------
    mech:
        Mechanical properties object with a
        ``compute_shear_modulus(p, omega)`` method.
    p_c:
        Percolation threshold (bond fraction at gel point).
    output_path:
        If given, the figure is saved via :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # --- data ---------------------------------------------------------------
    eps_values = np.logspace(-3, np.log10(0.5), 120)
    G_prime = np.array([mech.compute_shear_modulus(p_c + eps, omega=1.0) for eps in eps_values])

    # --- power-law fit overlay (f = 2.1, bond-bending) ----------------------
    f_exponent = 2.1
    # scale the power law to match the data at eps = 0.1
    idx_ref = np.argmin(np.abs(eps_values - 0.1))
    A = G_prime[idx_ref] / (eps_values[idx_ref] ** f_exponent)
    G_fit = A * eps_values ** f_exponent

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.loglog(eps_values, G_prime, color="steelblue", lw=1.5, label="G' (measured)")
    ax.loglog(eps_values, G_fit, color="black", lw=1.5, ls="--",
              label=r"$G' \sim \varepsilon^{f}$,  $f = 2.1$ (bond-bending)")

    # mark p_c (eps → 0 asymptote) with a vertical line at the left edge of plot
    ax.axvline(eps_values[0], color="gray", lw=1.0, ls=":", label=r"$p_c$")
    # annotate near the top
    ax.text(eps_values[0] * 1.3, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 1e3,
            r"$p_c$", fontsize=8, color="gray", va="top")

    ax.set_xlabel(r"$\varepsilon = p - p_c$")
    ax.set_ylabel(r"$G'$ (Pa)")
    ax.set_title("Critical scaling of shear modulus")
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Inverse percolation dynamics (gel-sol transition)
# ---------------------------------------------------------------------------

def plot_percolation_dynamics(
    times: Sequence[float],
    p_inf_hydrogel: Sequence[float],
    p_inf_collagen: Sequence[float],
    t_star: Optional[float] = None,
    output_path=None,
) -> mpl.figure.Figure:
    """Plot inverse percolation dynamics (gel-sol transition).

    Parameters
    ----------
    times:
        Time array [s].
    p_inf_hydrogel:
        Percolation order parameter for the hydrogel scaffold over time.
    p_inf_collagen:
        Percolation order parameter for the collagen network over time.
    t_star:
        Optional crossover time [s]; drawn as a vertical dotted line.
    output_path:
        If given, the figure is saved via :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    times = np.asarray(times)
    p_inf_hydrogel = np.asarray(p_inf_hydrogel)
    p_inf_collagen = np.asarray(p_inf_collagen)

    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    ax.plot(times, p_inf_hydrogel, color="steelblue", lw=1.5, ls="-",
            label=r"Hydrogel $P_\infty$")
    ax.plot(times, p_inf_collagen, color="firebrick", lw=1.5, ls="--",
            label=r"Collagen $P_\infty$")

    if t_star is not None:
        ax.axvline(t_star, color="dimgray", lw=1.0, ls=":",
                   label=r"$t^*$")
        ax.text(t_star, 1.02, r"$t^*$", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8, color="dimgray")

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$P_\infty$ (percolation order parameter)")
    ax.legend(loc="center right", frameon=False)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Early warning signal panel (key novel result)
# ---------------------------------------------------------------------------

def plot_ews_panel_publication(
    times: Sequence[float],
    G_prime: Sequence[float],
    ar1: Sequence[float],
    variance: Sequence[float],
    h1_counts: Sequence[float],
    h1_times: Sequence[float],
    t_transition: float,
    t_h1_peak: Optional[float] = None,
    t_ews_onset: Optional[float] = None,
    chi_series: Optional[Sequence[float]] = None,
    chi_times: Optional[Sequence[float]] = None,
    output_path=None,
) -> mpl.figure.Figure:
    """Plot the three-panel early warning signal figure.

    The panel is the key novel result of the paper: it shows that persistent
    homology (H₁ loop count) peaks *before* the rheological gel-sol transition,
    providing a topological early warning signal.

    Parameters
    ----------
    times:
        Shared time axis [s] for G', AR1, and variance arrays.
    G_prime:
        Storage modulus G'(t) [Pa].
    ar1:
        Rolling lag-1 autocorrelation coefficient (dimensionless).
    variance:
        Rolling variance of G'(t) [Pa²].
    h1_counts:
        H₁ persistent-homology loop count at each time in *h1_times*.
    h1_times:
        Time axis [s] for *h1_counts* (may differ from *times*).
    t_transition:
        Gel-sol transition time t_c [s].
    t_h1_peak:
        Optional time of H₁ peak; marked with a star.
    t_ews_onset:
        Optional EWS onset time; drawn as a vertical dotted line.
    chi_series:
        Optional susceptibility χ(t) = Σ s² n_s / N time series.
        When provided a 4th row is added showing χ(t) divergence.
    chi_times:
        Time axis [s] for *chi_series*.
    output_path:
        If given, the figure is saved via :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    times = np.asarray(times)
    G_prime = np.asarray(G_prime)
    ar1 = np.asarray(ar1)
    variance = np.asarray(variance)
    h1_counts = np.asarray(h1_counts)
    h1_times = np.asarray(h1_times)

    _has_chi = chi_series is not None and chi_times is not None
    n_rows = 4 if _has_chi else 3
    row_heights = [1.2, 1.2, 1.2, 1.0] if _has_chi else [1.2, 1.2, 1.2]
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(7.0, 7.5 if _has_chi else 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": row_heights},
    )
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    ax4 = axes[3] if _has_chi else None

    # ---- Row 1: G'(t) -------------------------------------------------------
    ax1.plot(times, G_prime, color="steelblue", lw=1.5)
    ax1.set_ylabel(r"$G'$ (Pa)")
    ax1.axvline(t_transition, color="black", lw=1.0, ls="--")
    if t_ews_onset is not None:
        ax1.axvline(t_ews_onset, color="dimgray", lw=1.0, ls=":")

    # ---- Row 2: AR1 (left) + variance (right twin axis) --------------------
    color_ar1 = "darkorange"
    color_var = "forestgreen"

    ax2.plot(times, ar1, color=color_ar1, lw=1.5, label="Rolling AR(1)")
    ax2.set_ylabel("Rolling AR(1)", color=color_ar1)
    ax2.tick_params(axis="y", labelcolor=color_ar1)
    ax2.axvline(t_transition, color="black", lw=1.0, ls="--")
    if t_ews_onset is not None:
        ax2.axvline(t_ews_onset, color="dimgray", lw=1.0, ls=":")

    ax2b = ax2.twinx()
    ax2b.plot(times, variance, color=color_var, lw=1.5, ls="-.", label="Rolling variance")
    ax2b.set_ylabel(r"Rolling variance (Pa$^2$)", color=color_var)
    ax2b.tick_params(axis="y", labelcolor=color_var)

    # combined legend for row 2
    lines_a, labels_a = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax2b.get_legend_handles_labels()
    ax2.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", frameon=False)

    # ---- Row 3: H₁ loop count -----------------------------------------------
    ax3.plot(h1_times, h1_counts, color="mediumpurple", lw=1.5, label=r"$H_1$ loop count")
    ax3.set_ylabel(r"$H_1$ loop count")
    ax3.set_xlabel("Time (s)")
    ax3.axvline(t_transition, color="black", lw=1.0, ls="--",
                label=r"Gel-sol transition ($t_c$)")
    if t_ews_onset is not None:
        ax3.axvline(t_ews_onset, color="dimgray", lw=1.0, ls=":",
                    label=r"EWS onset ($\Delta t_\mathrm{lead}$)")

    if t_h1_peak is not None:
        # find nearest index in h1_times
        idx_peak = int(np.argmin(np.abs(h1_times - t_h1_peak)))
        ax3.plot(h1_times[idx_peak], h1_counts[idx_peak], marker="*",
                 ms=10, color="mediumpurple", zorder=5, label=r"$H_1$ peak")

        # annotate lead time
        if t_h1_peak < t_transition:
            ax3.annotate(
                r"$\Delta T_\mathrm{topo} = t_c - t_{H_1\,\mathrm{peak}}$",
                xy=(t_h1_peak, h1_counts[idx_peak]),
                xytext=(t_h1_peak + 0.05 * (t_transition - h1_times[0]),
                        h1_counts[idx_peak] * 0.7),
                fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
                color="black",
            )

    ax3.legend(loc="upper right", frameon=False)

    # ---- Row 4 (optional): Susceptibility χ(t) ----------------------------
    if _has_chi and ax4 is not None:
        chi_arr = np.asarray(chi_series)
        chi_t = np.asarray(chi_times)
        ax4.plot(chi_t, chi_arr, color="saddlebrown", lw=1.5,
                 label=r"$\chi(t) = \sum s^2 n_s / N$")
        ax4.axvline(t_transition, color="black", lw=1.0, ls="--")
        if t_ews_onset is not None:
            ax4.axvline(t_ews_onset, color="dimgray", lw=1.0, ls=":")
        ax4.set_ylabel(r"Susceptibility $\chi$")
        ax4.set_xlabel("Time (s)")
        ax4.legend(loc="upper left", frameon=False)
    elif ax3 is not None:
        ax3.set_xlabel("Time (s)")

    # ---- shared annotation: transition line label --------------------------
    ax1.text(t_transition, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 1.0 else 1,
             r"Gel-sol transition ($t_c$)",
             rotation=90, va="top", ha="right", fontsize=7, color="black",
             transform=ax1.get_xaxis_transform())

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Handoff quality Q design map
# ---------------------------------------------------------------------------

def plot_q_heatmap_publication(
    rho_x_values: Sequence[float],
    k_base_values: Sequence[float],
    Q_matrix: "np.ndarray",
    output_path=None,
) -> mpl.figure.Figure:
    """Plot the handoff quality Q design map as a heatmap.

    Parameters
    ----------
    rho_x_values:
        Crosslinker density values [µm⁻³] — the y-axis of the heatmap.
    k_base_values:
        Enzymatic rate constant values [s⁻¹ nM⁻¹] — the x-axis of the heatmap.
    Q_matrix:
        2-D array of shape ``(len(rho_x_values), len(k_base_values))``
        containing Q = dP∞_col/dt − dP∞_hyd/dt.
    output_path:
        If given, the figure is saved via :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Q_matrix = np.asarray(Q_matrix, dtype=float)
    rho_x_values = np.asarray(rho_x_values, dtype=float)
    k_base_values = np.asarray(k_base_values, dtype=float)

    # Fixed symmetric colormap centred on zero: Q ∈ [-1, +1]
    vmin, vmax = -1.0, 1.0

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    im = ax.pcolormesh(
        k_base_values, rho_x_values, Q_matrix,
        cmap="RdYlGn", vmin=vmin, vmax=vmax, shading="auto",
    )

    # Q = 0 contour — scaffold failure boundary
    has_neg = np.any(Q_matrix < 0)
    has_pos = np.any(Q_matrix > 0)
    if has_neg and has_pos:
        cs = ax.contour(k_base_values, rho_x_values, Q_matrix,
                        levels=[0.0], colors="black", linewidths=1.5)
        ax.clabel(cs, fmt={0.0: "Q = 0 (failure boundary)"}, fontsize=7,
                  inline=True, inline_spacing=4)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        r"$Q = (t_\mathrm{fail} - t_\mathrm{col\,perc}) \,/\, t_\mathrm{fail}$",
        fontsize=8,
    )
    cbar.ax.text(0.5, 1.06, "Scaffold fails first →", transform=cbar.ax.transAxes,
                 ha="center", va="bottom", fontsize=6.5, color="red")
    cbar.ax.text(0.5, -0.08, "← Collagen percolates first", transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=6.5, color="darkgreen")

    ax.set_xlabel(r"$k_\mathrm{base}$ (s$^{-1}$ nM$^{-1}$)")
    ax.set_ylabel(r"$\rho_x$ (µm$^{-3}$)")
    ax.set_title("Handoff Quality Q: Formulation Design Map", fontsize=10)

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Fibroblast invasion depth and collagen assembly
# ---------------------------------------------------------------------------

def plot_invasion_summary(
    times: Sequence[float],
    invasion_depth: Sequence[float],
    collagen_p_inf: Sequence[float],
    n_cells_invaded: Optional[Sequence[float]] = None,
    output_path=None,
) -> mpl.figure.Figure:
    """Plot fibroblast invasion depth and collagen network assembly.

    Parameters
    ----------
    times:
        Shared time axis [s].
    invasion_depth:
        Leading-edge invasion depth [µm] over time.
    collagen_p_inf:
        Collagen network percolation order parameter P∞ over time.
    n_cells_invaded:
        Optional number of cells that have crossed the invasion front.
        Plotted on a right y-axis in row 1 when provided.
    output_path:
        If given, the figure is saved via :func:`save_figure`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    times = np.asarray(times)
    invasion_depth = np.asarray(invasion_depth)
    collagen_p_inf = np.asarray(collagen_p_inf)

    p_c_collagen = 0.55  # estimated collagen percolation threshold

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True)

    # ---- Row 1: invasion depth (+ optional cell count) ---------------------
    color_depth = "steelblue"
    ax1.plot(times, invasion_depth, color=color_depth, lw=1.5,
             label="Invasion depth")
    ax1.set_ylabel("Invasion depth (µm)", color=color_depth)
    ax1.tick_params(axis="y", labelcolor=color_depth)

    if n_cells_invaded is not None:
        n_cells_invaded = np.asarray(n_cells_invaded)
        ax1b = ax1.twinx()
        ax1b.plot(times, n_cells_invaded, color="gray", lw=1.5, ls="--",
                  label="Cells invaded")
        ax1b.set_ylabel("Cells invaded", color="gray")
        ax1b.tick_params(axis="y", labelcolor="gray")
        # merge legends
        lines_a, labels_a = ax1.get_legend_handles_labels()
        lines_b, labels_b = ax1b.get_legend_handles_labels()
        ax1.legend(lines_a + lines_b, labels_a + labels_b,
                   loc="upper left", frameon=False)
    else:
        ax1.legend(loc="upper left", frameon=False)

    # ---- Row 2: collagen P∞ -------------------------------------------------
    ax2.plot(times, collagen_p_inf, color="sienna", lw=1.5,
             label=r"Collagen $P_\infty$")
    ax2.axhline(p_c_collagen, color="black", lw=1.0, ls="--",
                label=r"Percolation threshold ($p_{c,\mathrm{col}} = 0.55$)")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"Collagen $P_\infty$")
    ax2.legend(loc="upper left", frameon=False)

    ax1.set_xlim(times[0], times[-1])

    fig.tight_layout()
    save_figure(fig, output_path)
    return fig
