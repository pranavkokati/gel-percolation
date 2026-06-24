#!/usr/bin/env python
"""Generate all five publication figures and save them to figures/.

Usage
-----
    python scripts/generate_figures.py
    python scripts/generate_figures.py --n-steps 800 --n-cells 20
    python scripts/generate_figures.py --skip-sweep   # skip Q heatmap (saves ~3 min)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
from src.cell_invasion import WoundHealingSimulation, CellParams, SimParams
from src.percolation_analysis import DualPercolationTracker
from src.publication_figures import (
    plot_critical_scaling,
    plot_percolation_dynamics,
    plot_invasion_summary,
    plot_q_heatmap_publication,
    save_figure,
)


# ---------------------------------------------------------------------------
# Main simulation run
# ---------------------------------------------------------------------------

def run_simulation(
    box_size: float,
    n_steps: int,
    n_cells: int,
    seed: int,
    k_base: float = 0.001,
    h1_lifetime_threshold: float = 0.1,
) -> dict:
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

    cp = CellParams()
    sp = SimParams(
        n_steps=n_steps, record_interval=10, n_cells=n_cells,
        grid_resolution=10, box_size=box_size, random_seed=seed,
    )

    def stiffness_fn(local_p_grid, omega=1.0):
        return mech.compute_stiffness_field(local_p_grid, omega=omega)

    sim = WoundHealingSimulation(
        cell_params=cp, sim_params=sp, mechanics=stiffness_fn, hydrogel_network=net,
    )
    sim.initialize()

    # TDA snapshots at lower frequency (max 20 over the run)
    tda_snap_every = max(1, n_steps // 20)
    # Chi computed at record_interval (every 10 steps) for finer resolution around transition
    chi_snap_every = 10
    tda_snapshots, tda_step_times, chi_series, chi_step_times = [], [], [], []

    print(f"  Running {n_steps} steps (n_cells={n_cells}) ...", flush=True)
    for step_i in range(n_steps):
        sim.step()
        if step_i % tda_snap_every == 0:
            tda_snapshots.append((net.get_node_positions(), net.get_active_edges()))
            tda_step_times.append(step_i * sp.dt)
        if step_i % chi_snap_every == 0:
            chi_series.append(net.compute_susceptibility())
            chi_step_times.append(step_i * sp.dt)
        if step_i % max(n_steps // 5, 1) == 0:
            print(f"    step {step_i}/{n_steps}", flush=True)

    history = sim.get_history()
    times = np.array([s.time for s in history])
    p_hyd = np.array([s.hydrogel_p_inf for s in history])
    p_col = np.array([s.collagen_p_inf for s in history])
    G_prime = np.array([mech.compute_shear_modulus(float(p), omega=1.0) for p in p_hyd])

    # EWS
    detector = EarlyWarningSignalDetector(window_size=40, detrend=True)
    ews = detector.compute_ews_indicators(G_prime, times)
    transition_t = ews.get("transition_time", float(times[-1]))

    # TDA
    print("  Running TDA ...", flush=True)
    tda = TopologicalDataAnalyzer(
        max_edge_length=hp.r_c * 2.0, max_dimension=1,
    )
    tda.lifetime_threshold = h1_lifetime_threshold   # override threshold
    tda_times = np.array(tda_step_times)
    tda_ts = tda.compute_topology_timeseries(
        tda_snapshots,
        lifetime_threshold=h1_lifetime_threshold,
    )
    h1_counts = tda_ts["n_long_lived_h1"]
    h1_peak_t = tda.detect_h1_peak_time(h1_counts, tda_times)
    print(f"  H1 peak at {h1_peak_t:.1f} s  (max H1 = {h1_counts.max():.0f})", flush=True)

    # Dual tracker
    tracker = DualPercolationTracker()
    for state in history:
        tracker.record(state.time, state.hydrogel_p_inf, state.collagen_p_inf)
    Q = tracker.compute_handoff_quality()
    t_star = tracker.compute_handoff_time()

    return dict(
        mech=mech, mp=mp, hp=hp, measured_p_c=measured_p_c,
        times=times, p_hyd=p_hyd, p_col=p_col, G_prime=G_prime,
        ews=ews, transition_t=transition_t,
        tda_times=tda_times, h1_counts=h1_counts, h1_peak_t=h1_peak_t,
        chi_series=np.array(chi_series), chi_times=np.array(chi_step_times),
        Q=Q, t_star=t_star, tracker=tracker,
        invasion_depth=np.array([s.invasion_depth for s in history]),
        n_collagen_fibers=np.array([s.n_collagen_fibers for s in history], dtype=float),
    )


# ---------------------------------------------------------------------------
# Figure 1 — Critical scaling (patched: remove duplicate p_c label)
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
              label=rf"$G' \sim \varepsilon^{f_exp}$,  $f = {f_exp}$ (bond-bending)")

    # vertical line only, no duplicate text
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
# Figure 3 — Transition signatures (revised)
# ---------------------------------------------------------------------------

def make_fig3(d: dict) -> plt.Figure:
    times = d["times"]
    G_prime = d["G_prime"]
    ar1 = d["ews"].get("ar1", np.full_like(times, np.nan))
    chi_arr = d["chi_series"]
    chi_t = d["chi_times"]
    h1 = d["h1_counts"]
    h1_t = d["tda_times"]
    t_c = d["transition_t"]
    h1_peak = d["h1_peak_t"]

    fig = plt.figure(figsize=(7.0, 8.5))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.10,
                           height_ratios=[1.3, 1.0, 1.4, 1.2])
    ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i]) for i in range(4)]

    vkw = dict(color="black", lw=1.2, ls="--", zorder=4)
    xlim = (times[0], times[-1])

    # --- Row 1: G'(t) --------------------------------------------------------
    ax1.plot(times, G_prime / 1e3, color="steelblue", lw=1.8)
    ax1.axvline(t_c, **vkw)
    ax1.set_ylabel(r"$G'$ (kPa)", labelpad=4)
    ax1.set_title("Gel–Sol Transition Signatures", fontsize=10, pad=5)
    ax1.set_xlim(*xlim)
    ax1.set_xticklabels([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # add t_c arrow annotation on top panel
    y_top = ax1.get_ylim()[1]
    ax1.annotate(
        r"$t_c$", xy=(t_c, y_top * 0.96),
        xytext=(t_c - 0.06 * (xlim[1] - xlim[0]), y_top * 0.96),
        fontsize=8, ha="right",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )

    # --- Row 2: AR1(t) — prediction label -----------------------------------
    valid_ar1 = ~np.isnan(ar1)
    if valid_ar1.sum() > 1:
        ax2.plot(times[valid_ar1], ar1[valid_ar1], color="darkorange", lw=1.4)
    ax2.axvline(t_c, **vkw)
    ax2.set_ylabel("Rolling AR(1)", color="darkorange", labelpad=4)
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.text(0.03, 0.88,
             "Predicted for experimental\nbulk-rheology data (not this model)",
             transform=ax2.transAxes, fontsize=7, color="darkorange",
             style="italic", va="top")
    ax2.set_xlim(*xlim)
    ax2.set_xticklabels([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # --- Row 3: χ(t) on log scale — the structural transition signature ------
    chi_pos = chi_arr[chi_arr > 0]
    chi_min = chi_pos.min() if len(chi_pos) > 0 else 1e-3
    chi_max = chi_arr.max()
    chi_plot = np.where(chi_arr > 0, chi_arr, chi_min * 0.5)

    ax3.semilogy(chi_t, chi_plot, color="saddlebrown", lw=1.8,
                 label=r"$\chi(t)$ (log scale)")
    ax3.axvline(t_c, **vkw)
    ax3.set_ylabel(r"$\chi = \sum s^2 n_s / N$", color="saddlebrown", labelpad=4)
    ax3.tick_params(axis="y", labelcolor="saddlebrown")
    ax3.set_xlim(*xlim)
    ax3.set_xticklabels([])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # annotate jump magnitude
    if chi_max > 0 and chi_min > 0:
        jump_factor = chi_max / chi_min
        ax3.text(0.65, 0.15,
                 rf"$\chi$ jumps {jump_factor:.0f}× at $t_c$",
                 transform=ax3.transAxes, fontsize=8,
                 color="saddlebrown", fontweight="bold")

    # --- Row 4: H₁(t) --------------------------------------------------------
    ax4.plot(h1_t, h1, color="mediumpurple", lw=1.8,
             label=r"$H_1$ long-lived loops")
    ax4.axvline(t_c, color="black", lw=1.2, ls="--",
                label=r"$t_c$  (gel-sol)")
    if h1_peak is not None and not np.isnan(h1_peak):
        ax4.axvline(h1_peak, color="mediumpurple", lw=1.0, ls=":",
                    label=rf"$H_1$ peak  (leads $t_c$?)")
        idx_peak = int(np.argmin(np.abs(h1_t - h1_peak)))
        ax4.plot(h1_t[idx_peak], h1[idx_peak], "v", ms=9,
                 color="mediumpurple", zorder=6)

    if h1.max() == 0:
        ax4.text(0.5, 0.5, "No long-lived H₁ loops detected\n(20 µm box; 50 µm needed for TDA)",
                 transform=ax4.transAxes, ha="center", va="center",
                 fontsize=8, color="gray", style="italic")

    ax4.set_ylabel(r"$H_1$ count")
    ax4.set_xlabel("Time (s)")
    ax4.set_xlim(*xlim)
    ax4.legend(loc="upper right", frameon=False, fontsize=7)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 1])
    save_figure(fig, FIGURES_DIR / "fig3_transition_signatures")
    return fig


# ---------------------------------------------------------------------------
# Q heatmap mini sweep (3×3)
# ---------------------------------------------------------------------------

def make_q_heatmap_mini(seed: int = 0) -> tuple:
    rho_vals = [0.5, 1.0, 2.0]
    k_vals   = [0.0005, 0.001, 0.003]
    Q_mat = np.full((len(rho_vals), len(k_vals)), np.nan)
    N_STEPS = 400

    for i, rho in enumerate(rho_vals):
        for j, k in enumerate(k_vals):
            try:
                hp = HydrogelParams(box_size=20.0, rho_x=rho, r_c=1.0, k_base=k)
                net = HydrogelNetwork(hp, seed=seed)
                p_c = net.measure_percolation_threshold(n_p_points=20, n_trials=2, rng_seed=seed)
                mp0 = MechanicsParams()
                mp = mp0.__class__(
                    p_c=p_c, p_crossover=mp0.p_crossover, T=mp0.T,
                    rho_chain_ref=mp0.rho_chain_ref, E_ref=mp0.E_ref,
                    omega_ref=mp0.omega_ref, kB=mp0.kB, exponents=mp0.exponents,
                )
                mech = PercolationMechanics(mp)

                def sfn(g, omega=1.0, _m=mech): return _m.compute_stiffness_field(g, omega)

                sp = SimParams(n_steps=N_STEPS, record_interval=20, n_cells=20,
                               grid_resolution=10, box_size=20.0, random_seed=seed)
                sim = WoundHealingSimulation(
                    cell_params=CellParams(), sim_params=sp,
                    mechanics=sfn, hydrogel_network=net,
                )
                sim.initialize()
                for _ in range(N_STEPS):
                    sim.step()
                history = sim.get_history()
                tracker = DualPercolationTracker()
                for s in history:
                    tracker.record(s.time, s.hydrogel_p_inf, s.collagen_p_inf)
                Q_mat[i, j] = tracker.compute_handoff_quality()
                print(f"    ρ={rho}, k={k}: Q={Q_mat[i,j]:+.4f}", flush=True)
            except Exception as exc:
                print(f"    ρ={rho}, k={k}: FAILED — {exc}", flush=True)

    return rho_vals, k_vals, Q_mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=700)
    parser.add_argument("--n-cells", type=int, default=20)
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--h1-threshold", type=float, default=0.1,
                        help="Minimum H1 loop lifetime counted as long-lived (default 0.1)")
    args = parser.parse_args()

    print("=" * 60)
    print("  gel-percolation: Generating Publication Figures")
    print(f"  box={args.box_size} µm  n_steps={args.n_steps}  n_cells={args.n_cells}  seed={args.seed}")
    print(f"  Output → {FIGURES_DIR}/")
    print("=" * 60)

    # ---- main simulation run -----------------------------------------------
    print("\n[1] Running main simulation...")
    d = run_simulation(
        args.box_size, args.n_steps, args.n_cells, args.seed,
        h1_lifetime_threshold=args.h1_threshold,
    )
    print(f"  transition at t={d['transition_t']:.0f} s  Q={d['Q']:+.4f}")

    # ---- Figure 1: Critical scaling ----------------------------------------
    print("\n[Fig 1] Critical scaling of G'...")
    fig1 = make_fig1(d)
    plt.close(fig1)
    print("  → fig1_critical_scaling.png/.pdf")

    # ---- Figure 2: Percolation dynamics ------------------------------------
    print("\n[Fig 2] Percolation dynamics (dual P∞)...")
    fig2 = plot_percolation_dynamics(
        d["times"], d["p_hyd"], d["p_col"], t_star=d["t_star"],
        output_path=FIGURES_DIR / "fig2_percolation_dynamics",
    )
    plt.close(fig2)
    print("  → fig2_percolation_dynamics.png/.pdf")

    # ---- Figure 3: Transition signatures -----------------------------------
    print("\n[Fig 3] Transition signatures panel...")
    fig3 = make_fig3(d)
    plt.close(fig3)
    print("  → fig3_transition_signatures.png/.pdf")

    # ---- Figure 4: Q heatmap -----------------------------------------------
    if args.skip_sweep:
        print("\n[Fig 4] Q heatmap SKIPPED (--skip-sweep)")
    else:
        print("\n[Fig 4] Q design-map heatmap (3×3 sweep)...")
        rho_v, k_v, Q_mat = make_q_heatmap_mini(seed=args.seed)
        fig4 = plot_q_heatmap_publication(
            rho_v, k_v, Q_mat,
            output_path=FIGURES_DIR / "fig4_q_heatmap",
        )
        plt.close(fig4)
        print("  → fig4_q_heatmap.png/.pdf")

    # ---- Figure 5: Invasion summary ----------------------------------------
    print("\n[Fig 5] Invasion depth & collagen assembly...")
    fig5 = plot_invasion_summary(
        d["times"], d["invasion_depth"], d["p_col"],
        n_cells_invaded=None,   # collagen fibers ≠ invaded cells; omit to avoid mislabelling
        output_path=FIGURES_DIR / "fig5_invasion_summary",
    )
    plt.close(fig5)
    print("  → fig5_invasion_summary.png/.pdf")

    print("\n" + "=" * 60)
    print("  Done.  All figures written to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
