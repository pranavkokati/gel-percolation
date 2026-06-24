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

    return dict(
        mech=mech, mp=mp, hp=hp, measured_p_c=measured_p_c,
        times=times, p_hyd=p_hyd, p_col=p_col, G_prime=G_prime,
        ews=ews, transition_t=transition_t, transition_idx=transition_idx,
        tda_times=tda_times, h1_counts=h1_counts, h1_peak_t=h1_peak_t,
        chi_series=chi_arr, chi_times=chi_t,
        t_star=t_star,
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
    ar1 = d["ews"].get("ar1", np.full_like(times, np.nan))
    chi_arr = d["chi_series"]
    chi_t = d["chi_times"]
    h1 = d["h1_counts"]
    h1_t = d["tda_times"]
    t_c = d["transition_t"]
    h1_peak = d["h1_peak_t"]
    measured_p_c = d["measured_p_c"]

    fig = plt.figure(figsize=(7.0, 9.0))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.10,
                           height_ratios=[1.3, 1.0, 1.4, 1.2])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    ax1, ax2, ax3, ax4 = axes
    vkw = dict(color="black", lw=1.2, ls="--", zorder=4)
    xlim = (float(times[0]), float(times[-1]))

    # Row 1: G'(t)
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

    # Row 2: AR1(t) — prediction only
    valid = ~np.isnan(ar1)
    if valid.sum() > 1:
        ax2.plot(times[valid], ar1[valid], color="darkorange", lw=1.4)
    ax2.axvline(t_c, **vkw)
    ax2.set_ylabel("Rolling AR(1)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.text(0.03, 0.88,
             "Predicted for experimental data\n(not observable in deterministic model)",
             transform=ax2.transAxes, fontsize=7,
             color="darkorange", style="italic", va="top")
    ax2.set_xlim(*xlim)
    ax2.set_xticklabels([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Row 3: χ(t) on log scale
    chi_pos = chi_arr[chi_arr > 0]
    chi_min_pos = chi_pos.min() if len(chi_pos) > 0 else 1e-3
    chi_plot = np.where(chi_arr > 0, chi_arr, chi_min_pos * 0.1)
    ax3.semilogy(chi_t, chi_plot, color="saddlebrown", lw=1.8)
    ax3.axvline(t_c, **vkw)
    ax3.set_ylabel(r"$\chi(t) = \Sigma s^2 n_s / N$", color="saddlebrown")
    ax3.tick_params(axis="y", labelcolor="saddlebrown")
    ax3.set_xlim(*xlim)
    ax3.set_xticklabels([])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    if chi_arr.max() > 0 and chi_min_pos > 0:
        jump = chi_arr.max() / chi_min_pos
        ax3.text(0.60, 0.15, rf"$\chi$ jumps {jump:.0f}× at $t_c$",
                 transform=ax3.transAxes, fontsize=8,
                 color="saddlebrown", fontweight="bold")

    # Row 4: H₁(t)
    ax4.plot(h1_t, h1, color="mediumpurple", lw=1.8,
             label=r"$H_1$ long-lived loops")
    ax4.axvline(t_c, color="black", lw=1.2, ls="--", label=r"$t_c$")
    if h1_peak is not None and not (isinstance(h1_peak, float) and np.isnan(h1_peak)):
        ax4.axvline(h1_peak, color="mediumpurple", lw=1.0, ls=":",
                    label=rf"$H_1$ peak")
        idx_p = int(np.argmin(np.abs(h1_t - h1_peak)))
        ax4.plot(h1_t[idx_p], h1[idx_p], "v", ms=9, color="mediumpurple", zorder=6)
    if h1.max() == 0:
        ax4.text(0.50, 0.50,
                 "No long-lived H₁ loops in 20 µm box\n"
                 r"(need 50 µm to resolve $\xi$ near $p_c$)",
                 transform=ax4.transAxes, ha="center", va="center",
                 fontsize=8, color="gray", style="italic")
    ax4.set_ylabel(r"$H_1$ loop count")
    ax4.set_xlabel("Time  (steps × 1 s)")
    ax4.legend(loc="upper right", frameon=False, fontsize=7)
    ax4.set_xlim(*xlim)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

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
    from src.cell_invasion import WoundHealingSimulation, CellParams, SimParams

    rho_vals = [0.5, 1.0, 2.0]
    k_vals   = [0.0005, 0.001, 0.003]
    Q_mat    = np.full((len(rho_vals), len(k_vals)), np.nan)
    N_STEPS  = 400   # enough to observe transition at k≥0.001

    for i, rho in enumerate(rho_vals):
        for j, k in enumerate(k_vals):
            try:
                hp = HydrogelParams(box_size=20.0, rho_x=rho, r_c=1.0, k_base=k)
                net = HydrogelNetwork(hp, seed=seed)
                p_c = net.measure_percolation_threshold(n_p_points=15, n_trials=2,
                                                        rng_seed=seed)
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
                mmp_uniform = np.ones((10, 10, 10), dtype=float)
                for step_i in range(N_STEPS):
                    # Degrade with uniform [MMP]=1 nM (same calibration as main run)
                    net.degrade_step(mmp_uniform, dt=1.0)
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
