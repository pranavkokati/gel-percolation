#!/usr/bin/env python
"""Generate all five publication figures and save them to figures/.

Uses direct network degradation with uniform [MMP] = 1 nM (the k_base calibration
target), bypassing the cell-MMP diffusion loop which would require ~50,000 steps
to reach the transition.  This correctly demonstrates the percolation physics at
the design-target enzyme concentration.

Usage
-----
    python scripts/generate_figures.py
    python scripts/generate_figures.py --n-steps 600 --seed 42
    python scripts/generate_figures.py --skip-sweep   # skip Q heatmap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

from src.network_model import HydrogelNetwork, HydrogelParams
from src.mechanical_properties import PercolationMechanics, MechanicsParams
from src.early_warning import EarlyWarningSignalDetector, TopologicalDataAnalyzer
from src.percolation_analysis import DualPercolationTracker
from src.publication_figures import (
    plot_critical_scaling,
    plot_percolation_dynamics,
    save_figure,
)


# ---------------------------------------------------------------------------
# Direct-degradation run (uniform [MMP] = 1 nM)
# ---------------------------------------------------------------------------

def run_degradation(
    box_size: float,
    n_steps: int,
    seed: int,
    k_base: float = 0.001,
    h1_lifetime_threshold: float = 0.1,
) -> dict:
    """Degrade bonds with uniform [MMP]=1 nM; track all observables."""
    print(f"  Building network (box={box_size} µm, seed={seed}) ...", flush=True)
    hp = HydrogelParams(box_size=box_size, rho_x=1.0, r_c=1.0, k_base=k_base)
    net = HydrogelNetwork(hp, seed=seed)

    print("  Measuring empirical p_c ...", flush=True)
    measured_p_c = net.measure_percolation_threshold(n_p_points=30, n_trials=3, rng_seed=seed)
    print(f"  p_c = {measured_p_c:.4f}")

    mp0 = MechanicsParams()
    mp = mp0.__class__(
        p_c=measured_p_c, p_crossover=mp0.p_crossover, T=mp0.T,
        rho_chain_ref=mp0.rho_chain_ref, E_ref=mp0.E_ref,
        omega_ref=mp0.omega_ref, kB=mp0.kB, exponents=mp0.exponents,
    )
    mech = PercolationMechanics(mp)

    # Uniform [MMP] = 1 nM over a 10×10×10 grid (constant throughout run)
    grid_res = 10
    mmp_uniform = np.ones((grid_res, grid_res, grid_res), dtype=float)

    tda_snap_every = max(1, n_steps // 20)
    tda_snapshots, tda_step_times = [], []
    chi_step = max(1, n_steps // 100)   # chi at every ~1% of run
    chi_series, chi_step_times = [], []

    # Record interval for P∞ and G'
    rec_every = max(1, n_steps // 200)
    rec_times, rec_p_hyd = [], []

    print(f"  Degrading {n_steps} steps at [MMP]=1 nM ...", flush=True)
    for step_i in range(n_steps):
        net.degrade_step(mmp_uniform, dt=1.0)

        if step_i % rec_every == 0:
            p_inf = net.get_percolation_order_parameter()
            rec_times.append(float(step_i))
            rec_p_hyd.append(p_inf)

        if step_i % chi_step == 0:
            chi_series.append(net.compute_susceptibility())
            chi_step_times.append(float(step_i))

        if step_i % tda_snap_every == 0:
            tda_snapshots.append((net.get_node_positions(), net.get_active_edges()))
            tda_step_times.append(float(step_i))

        if step_i % max(n_steps // 5, 1) == 0:
            print(f"    step {step_i}/{n_steps}  P∞={rec_p_hyd[-1]:.3f}", flush=True)

    times = np.array(rec_times)
    p_hyd = np.array(rec_p_hyd)
    G_prime = np.array([mech.compute_shear_modulus(float(p), omega=1.0) for p in p_hyd])
    chi_arr = np.array(chi_series)
    chi_t = np.array(chi_step_times)

    # EWS
    detector = EarlyWarningSignalDetector(window_size=max(10, len(times) // 20), detrend=True)
    ews = detector.compute_ews_indicators(G_prime, times)
    transition_t = ews.get("transition_time", float(times[-1]))
    transition_idx = ews.get("transition_idx", len(times) - 1)
    print(f"  Transition at t ≈ {transition_t:.0f} s", flush=True)

    # TDA
    print("  Running TDA ...", flush=True)
    tda = TopologicalDataAnalyzer(max_edge_length=hp.r_c * 2.0, max_dimension=1)
    tda.lifetime_threshold = h1_lifetime_threshold   # used by compute_h1_statistics
    tda_times = np.array(tda_step_times)
    tda_ts = tda.compute_topology_timeseries(tda_snapshots)
    h1_counts = tda_ts["n_long_lived_h1"]
    h1_peak_t = tda.detect_h1_peak_time(h1_counts, tda_times)
    print(f"  H1 peak at {h1_peak_t:.0f} s  max={h1_counts.max():.0f}", flush=True)

    # Synthetic collagen P∞: sigmoidal growth (cells deposit ECM as gel degrades)
    # Peaks at t* ≈ 0.4 * transition_t, plateaus at ~0.7
    t_norm = times / max(transition_t, 1.0)
    p_col = 0.7 / (1.0 + np.exp(-8.0 * (t_norm - 0.4)))

    t_star = float(times[np.argmin(np.abs(p_hyd - p_col))]) if len(p_hyd) > 1 else 0.0

    # Kendall τ for χ(t) before transition (primary EWS metric)
    tau_chi = float("nan")
    if len(chi_t) >= 4:
        pre_mask = chi_t < transition_t
        if pre_mask.sum() >= 4:
            tau_chi_val, _ = stats.kendalltau(chi_t[pre_mask], chi_arr[pre_mask])
            tau_chi = float(tau_chi_val)

    return dict(
        mech=mech, mp=mp, hp=hp, measured_p_c=measured_p_c,
        times=times, p_hyd=p_hyd, p_col=p_col, G_prime=G_prime,
        ews=ews, transition_t=transition_t, transition_idx=transition_idx,
        tda_times=tda_times, h1_counts=h1_counts, h1_peak_t=h1_peak_t,
        chi_series=chi_arr, chi_times=chi_t,
        t_star=t_star, tau_chi=tau_chi,
    )


# ---------------------------------------------------------------------------
# Figure 1 — Critical scaling
# ---------------------------------------------------------------------------

def make_fig1(d: dict) -> plt.Figure:
    mech = d["mech"]
    p_c = d["measured_p_c"]
    eps_values = np.logspace(-3, np.log10(0.45), 200)
    G = np.array([mech.compute_shear_modulus(p_c + eps, omega=1.0) for eps in eps_values])

    f_exp = mech.params.exponents.f_elastic
    idx_ref = int(np.argmin(np.abs(eps_values - 0.08)))
    A = G[idx_ref] / (eps_values[idx_ref] ** f_exp)
    G_fit = A * eps_values ** f_exp

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.loglog(eps_values, G, color="steelblue", lw=1.8, label=r"$G'(p)$ (model)")
    ax.loglog(eps_values, G_fit, color="black", lw=1.4, ls="--",
              label=rf"$G' \sim \varepsilon^{{{f_exp}}}$  ($f={f_exp}$, bond-bending)")
    ax.axvline(eps_values[0], color="gray", lw=1.0, ls=":",
               label=rf"$p_c = {p_c:.3f}$")
    ax.set_xlabel(r"$\varepsilon = p - p_c$")
    ax.set_ylabel(r"$G'$ (Pa)")
    ax.set_title("Critical scaling of shear modulus")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "fig1_critical_scaling")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Transition signatures (4-panel)
# ---------------------------------------------------------------------------

def make_fig3(d: dict) -> plt.Figure:
    times = d["times"]
    G_prime = d["G_prime"]
    p_hyd = d["p_hyd"]
    p_col = d["p_col"]
    chi_arr = d["chi_series"]
    chi_t = d["chi_times"]
    t_c = d["transition_t"]
    measured_p_c = d["measured_p_c"]
    tau_chi = d["tau_chi"]

    fig = plt.figure(figsize=(7.0, 9.0))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.15,
                           height_ratios=[1.3, 1.4, 1.2])
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    ax1, ax2, ax3 = axes
    vkw = dict(color="black", lw=1.2, ls="--", zorder=4)
    xlim = (float(times[0]), float(times[-1]))

    # Row 1: G'(t) with gel-sol transition
    ax1.plot(times, G_prime / 1e3, color="steelblue", lw=1.8)
    ax1.axvline(t_c, **vkw)
    ax1.set_ylabel(r"$G'$ (kPa)")
    ax1.set_title(
        rf"Gel–Sol Transition Signatures  ($p_c = {measured_p_c:.3f}$, $k_\mathrm{{base}}=0.001$)",
        fontsize=10, pad=5,
    )
    ax1.set_xlim(*xlim)
    ax1.set_xticklabels([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(t_c * 1.01, ax1.get_ylim()[1] * 0.9, r"$t_c$",
             fontsize=8, color="black")

    # Row 2: χ(t) — primary EWS result (cluster-size susceptibility)
    chi_pos = chi_arr[chi_arr > 0]
    chi_min_pos = chi_pos.min() if len(chi_pos) > 0 else 1e-3
    chi_plot = np.where(chi_arr > 0, chi_arr, chi_min_pos * 0.1)
    ax2.semilogy(chi_t, chi_plot, color="saddlebrown", lw=1.8)
    ax2.axvline(t_c, **vkw)
    ax2.set_ylabel(r"$\chi(t) = \Sigma s^2 n_s / N$", color="saddlebrown")
    ax2.tick_params(axis="y", labelcolor="saddlebrown")
    ax2.set_xlim(*xlim)
    ax2.set_xticklabels([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    if chi_arr.max() > 0 and chi_min_pos > 0:
        jump = chi_arr.max() / chi_min_pos
        annotations = [rf"$\chi$ rises {jump:.0f}× at $t_c$"]
        if np.isfinite(tau_chi):
            annotations.append(rf"Kendall $\tau$ = {tau_chi:+.3f} (pre-$t_c$)")
        text_str = "\n".join(annotations)
        ax2.text(0.55, 0.15, text_str,
                 transform=ax2.transAxes, fontsize=8,
                 color="saddlebrown", fontweight="bold",
                 va="bottom", ha="left")

    # Row 3: Dual P∞ — hydrogel degradation and collagen assembly
    ax3.plot(times, p_hyd, color="steelblue", lw=1.8, ls="-", label=r"Hydrogel $P_\infty$")
    ax3.plot(times, p_col, color="firebrick", lw=1.8, ls="--", label=r"Collagen $P_\infty$ (model)")
    ax3.axhline(measured_p_c, color="gray", lw=0.8, ls=":",
                label=rf"$p_c = {measured_p_c:.3f}$")
    ax3.axvline(t_c, color="black", lw=1.2, ls="--", label=r"$t_c$")
    ax3.set_xlabel("Time  (steps × 1 s)")
    ax3.set_ylabel(r"$P_\infty$")
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="center left", frameon=False, fontsize=7)
    ax3.set_xlim(*xlim)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "fig3_transition_signatures")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Invasion summary (model-based proxy)
# ---------------------------------------------------------------------------

def make_fig5(d: dict) -> plt.Figure:
    """Simplified invasion figure: invasion depth proxy from p_hyd decline."""
    times = d["times"]
    p_hyd = d["p_hyd"]
    p_col = d["p_col"]
    measured_p_c = d["measured_p_c"]

    # Invasion depth proxy: cells invade once stiffness drops (p < p_c)
    # Invasion depth ~ integral of (p_c - p) for p < p_c
    invasion_depth = np.cumsum(np.where(p_hyd < measured_p_c,
                                        (measured_p_c - p_hyd) * 5.0, 0.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True)

    ax1.plot(times, invasion_depth, color="steelblue", lw=1.8, label="Invasion depth (proxy)")
    ax1.set_ylabel("Invasion depth (µm)")
    ax1.legend(loc="upper left", frameon=False, fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(times, p_hyd, color="steelblue", lw=1.8, ls="-", label=r"Hydrogel $P_\infty$")
    ax2.plot(times, p_col, color="firebrick", lw=1.8, ls="--",
             label=r"Collagen $P_\infty$ (model)")
    ax2.axhline(measured_p_c, color="gray", lw=0.8, ls=":",
                label=rf"$p_c = {measured_p_c:.3f}$")
    ax2.set_xlabel("Time  (steps × 1 s)")
    ax2.set_ylabel(r"$P_\infty$")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="center right", frameon=False, fontsize=7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Fibroblast Invasion and Scaffold Replacement", fontsize=10)
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "fig5_invasion_summary")
    return fig


# ---------------------------------------------------------------------------
# Q heatmap mini sweep (3×3)
# ---------------------------------------------------------------------------

def make_q_heatmap_mini(seed: int = 0) -> tuple:
    """3×3 Q heatmap: timing-based handoff quality over (ρ_x, k_base) space.

    Q = (t_fail_hyd − t_col_perc) / t_fail_hyd
       > 0  →  collagen percolates BEFORE scaffold fails (green, heals)
       < 0  →  scaffold fails BEFORE collagen percolates (red, re-opens)

    Uses direct [MMP]=1 nM degradation for the hydrogel and a physics-based
    absolute-time sigmoid for collagen (growth rate ∝ ρ_x, fixed 50-step lag).
    N_STEPS is fixed at 700 for all conditions — if the hydrogel hasn't failed
    by then (fast-recovering dense networks), Q = +1.0.
    """
    rho_vals = [0.5, 1.0, 2.0]
    k_vals   = [0.0005, 0.001, 0.003]
    Q_mat    = np.full((len(rho_vals), len(k_vals)), np.nan)

    mmp_uniform = np.ones((10, 10, 10), dtype=float)
    N_STEPS   = 700     # fixed window; non-failure in window → Q = +1.0
    REC_EVERY = 5
    # Collagen model constants
    P_COL_MAX = 0.75    # maximum achievable collagen P∞
    P_COL_PC  = 0.50    # collagen percolation threshold (mid-plateau)
    LAG_STEPS = 50.0    # minimum delay before fibroblasts start depositing ECM

    for i, rho in enumerate(rho_vals):
        for j, k in enumerate(k_vals):
            try:
                hp = HydrogelParams(box_size=20.0, rho_x=rho, r_c=1.0, k_base=k)
                net = HydrogelNetwork(hp, seed=seed)
                p_c_hyd = net.measure_percolation_threshold(
                    n_p_points=15, n_trials=2, rng_seed=seed
                )

                times_list, p_hyd_list = [], []
                for step_i in range(N_STEPS):
                    net.degrade_step(mmp_uniform, dt=1.0)
                    if step_i % REC_EVERY == 0:
                        times_list.append(float(step_i))
                        p_hyd_list.append(net.get_percolation_order_parameter())

                times_arr = np.array(times_list, dtype=float)
                p_hyd_arr = np.array(p_hyd_list, dtype=float)

                # Hydrogel failure time: first recorded step where P∞ < p_c
                hyd_failed = p_hyd_arr < p_c_hyd
                t_hyd_fail = float(times_arr[np.argmax(hyd_failed)]) if hyd_failed.any() \
                             else float('inf')

                # Physics-based synthetic collagen:
                #   Higher ρ_x → stiffer scaffold → stronger durotaxis gradient →
                #   earlier and faster fibroblast invasion → earlier collagen ECM.
                # t_col_mid (half-saturation): 150 steps for ρ=2, 250 for ρ=1, 450 for ρ=0.5
                # steepness: sharper invasion front for denser scaffold
                t_col_mid = LAG_STEPS + 200.0 / rho
                steepness = 0.012 * rho
                p_col_arr = P_COL_MAX / (1.0 + np.exp(-steepness * (times_arr - t_col_mid)))

                # Collagen percolation time: first step where p_col > P_COL_PC
                col_percolated = p_col_arr > P_COL_PC
                t_col_perc = float(times_arr[np.argmax(col_percolated)]) if col_percolated.any() \
                             else float('inf')

                # Timing-based handoff quality
                if np.isinf(t_hyd_fail):
                    Q = 1.0    # scaffold intact throughout → perfect handoff
                elif np.isinf(t_col_perc):
                    Q = -1.0   # collagen never percolated → catastrophic failure
                else:
                    Q = (t_hyd_fail - t_col_perc) / max(t_hyd_fail, 1.0)
                    Q = float(np.clip(Q, -1.0, 1.0))

                Q_mat[i, j] = Q
                print(
                    f"    ρ={rho}, k={k}: p_c={p_c_hyd:.3f}  "
                    f"t_fail={t_hyd_fail:.0f}  t_col_perc={t_col_perc:.0f}  Q={Q:+.3f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"    ρ={rho}, k={k}: FAILED — {exc}", flush=True)

    return rho_vals, k_vals, Q_mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip the Q-heatmap parameter sweep")
    parser.add_argument("--h1-threshold", type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 60)
    print("  gel-percolation: Generating Publication Figures")
    print(f"  box={args.box_size} µm  n_steps={args.n_steps}  seed={args.seed}")
    print(f"  Uniform [MMP]=1 nM degradation (k_base calibration target)")
    print(f"  Output → {FIGURES_DIR}/")
    print("=" * 60)

    print("\n[1] Running direct network degradation...")
    d = run_degradation(
        args.box_size, args.n_steps, args.seed,
        h1_lifetime_threshold=args.h1_threshold,
    )

    print("\n[Fig 1] Critical scaling of G'...")
    fig1 = make_fig1(d)
    plt.close(fig1)
    print("  → fig1_critical_scaling.png/.pdf")

    print("\n[Fig 2] Percolation dynamics (dual P∞)...")
    fig2 = plot_percolation_dynamics(
        d["times"], d["p_hyd"], d["p_col"], t_star=d["t_star"],
        output_path=FIGURES_DIR / "fig2_percolation_dynamics",
    )
    plt.close(fig2)
    print("  → fig2_percolation_dynamics.png/.pdf")

    print("\n[Fig 3] Transition signatures panel...")
    fig3 = make_fig3(d)
    plt.close(fig3)
    print("  → fig3_transition_signatures.png/.pdf")

    if args.skip_sweep:
        print("\n[Fig 4] Q heatmap SKIPPED (--skip-sweep)")
    else:
        from src.publication_figures import plot_q_heatmap_publication
        print("\n[Fig 4] Q heatmap (3×3 sweep with uniform MMP degradation)...")
        rho_v, k_v, Q_mat = make_q_heatmap_mini(seed=args.seed)
        fig4 = plot_q_heatmap_publication(
            rho_v, k_v, Q_mat,
            output_path=FIGURES_DIR / "fig4_q_heatmap",
        )
        plt.close(fig4)
        print("  → fig4_q_heatmap.png/.pdf")

    print("\n[Fig 5] Invasion depth & collagen assembly...")
    fig5 = make_fig5(d)
    plt.close(fig5)
    print("  → fig5_invasion_summary.png/.pdf")

    print("\n" + "=" * 60)
    print("  Done.  Figures written to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
