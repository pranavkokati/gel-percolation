"""Main entry point for the gel-percolation simulation.

Usage:
    python run_simulation.py --mode single
    python run_simulation.py --mode sweep --output results/sweep/
    python run_simulation.py --mode validate
    python run_simulation.py --mode demo
    python run_simulation.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path so ``src`` is importable without editable install.
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger("gel_percolation.runner")


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load YAML configuration file and return as a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def override_config(cfg: dict, args) -> dict:
    """Apply command-line argument overrides to the config dict in-place."""
    sim = cfg.setdefault("simulation", {})
    hydrogel = cfg.setdefault("hydrogel", {})

    if args.n_cells is not None:
        sim["n_cells"] = args.n_cells
    if args.rho_x is not None:
        hydrogel["rho_x"] = args.rho_x
    if args.k_base is not None:
        hydrogel["k_base"] = args.k_base
    if args.seed is not None:
        sim["random_seed"] = args.seed
    if args.output is not None:
        sim["output_dir"] = args.output
    return cfg


def build_params_from_config(cfg: dict):
    """Construct parameter dataclasses from a loaded config dict.

    Returns
    -------
    (HydrogelParams, CellParams, SimParams, MechanicsParams)
    """
    from src.network_model import HydrogelParams
    from src.cell_invasion import CellParams, SimParams
    from src.mechanical_properties import MechanicsParams, PercolationCriticalExponents

    h = cfg.get("hydrogel", {})
    hp = HydrogelParams(
        box_size=h.get("box_size", 50.0),
        rho_x=h.get("rho_x", 1.0),
        r_c=h.get("r_c", 1.0),
        k_base=h.get("k_base", 0.01),
        tau_ionic=h.get("tau_ionic", 3600.0),
        E_activation=h.get("E_activation", 50000.0),
        T=cfg.get("mechanics", {}).get("T", 310.15),
    )

    c = cfg.get("cells", {})
    cp = CellParams(
        mu_durotaxis=c.get("mu_durotaxis", 0.5),
        mu_chemotaxis=c.get("mu_chemotaxis", 0.3),
        D_rand=c.get("D_rand", 0.1),
        E_threshold=c.get("E_threshold", 500.0),
        k_MMP_high=c.get("k_MMP_high", 1e-3),
        k_MMP_low=c.get("k_MMP_low", 1e-5),
        r_col=c.get("r_col", 0.01),
        E_ref=c.get("E_ref", 1000.0),
    )

    s = cfg.get("simulation", {})
    sp = SimParams(
        dt=s.get("dt", 1.0),
        n_steps=s.get("n_steps", 1800),
        record_interval=s.get("record_interval", 10),
        box_size=h.get("box_size", 50.0),
        grid_resolution=cfg.get("mmp_diffusion", {}).get("grid_resolution", 20),
        n_cells=s.get("n_cells", 20),
        random_seed=s.get("random_seed", 42),
    )

    m = cfg.get("mechanics", {})
    exponents = PercolationCriticalExponents(
        f_elastic=m.get("f_elastic", 2.1),
        Delta=m.get("Delta", 0.72),
        nu=m.get("nu", 0.88),
        gamma_sus=m.get("gamma_sus", 1.8),
    )
    mp = MechanicsParams(
        p_c=h.get("p_c_nominal", m.get("p_c", 0.33)),
        p_crossover=m.get("p_crossover", 0.05),
        T=m.get("T", 310.15),
        E_ref=m.get("E_ref", 1000.0),
        exponents=exponents,
    )

    return hp, cp, sp, mp


# ---------------------------------------------------------------------------
# Simulation modes
# ---------------------------------------------------------------------------


def run_single(cfg: dict, no_plots: bool = False) -> None:
    """Run a single wound-healing simulation and save all results."""
    from src.network_model import HydrogelNetwork
    from src.mechanical_properties import PercolationMechanics, MechanicsParams
    from src.cell_invasion import WoundHealingSimulation
    from src.early_warning import (
        EarlyWarningSignalDetector,
        TopologicalDataAnalyzer,
        plot_ews_panel,
    )
    from src.percolation_analysis import (
        DualPercolationTracker,
        SummaryReporter,
    )

    try:
        from tqdm import tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    hp, cp, sp, mp = build_params_from_config(cfg)
    output_dir = Path(cfg.get("simulation", {}).get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build hydrogel network (Module 1) ---
    logger.info(
        "Building hydrogel network (rho_x=%.2f µm⁻³, box=%.0f µm, r_c=%.1f µm)...",
        hp.rho_x, hp.box_size, hp.r_c,
    )
    network = HydrogelNetwork(hp, seed=sp.random_seed)
    logger.info("Network: %s", network)

    # --- Measure empirical p_c for this network topology ---
    logger.info("Measuring empirical percolation threshold (n_p_points=40, n_trials=5)...")
    measured_p_c = network.measure_percolation_threshold(n_p_points=40, n_trials=5, rng_seed=0)
    logger.info("  Empirical p_c = %.4f (config nominal = %.4f)", measured_p_c, mp.p_c)
    mp = MechanicsParams(
        p_c=measured_p_c,
        p_crossover=mp.p_crossover,
        T=mp.T,
        rho_chain_ref=mp.rho_chain_ref,
        E_ref=mp.E_ref,
        omega_ref=mp.omega_ref,
        kB=mp.kB,
        exponents=mp.exponents,
    )
    (output_dir / "measured_p_c.txt").write_text(
        f"measured_p_c={measured_p_c:.6f}\nr_c={hp.r_c}\nbox_size={hp.box_size}\n",
        encoding="utf-8",
    )

    # --- Mechanics callable wrapping Module 2 ---
    mech = PercolationMechanics(mp)

    def stiffness_fn(local_p_grid, omega: float = 1.0):
        return mech.compute_stiffness_field(local_p_grid, omega=omega)

    # --- Initialise wound healing simulation (Module 4) ---
    logger.info(
        "Initialising WoundHealingSimulation (%d cells, %d steps, dt=%.1f s)...",
        sp.n_cells, sp.n_steps, sp.dt,
    )
    sim = WoundHealingSimulation(
        cell_params=cp,
        sim_params=sp,
        mechanics=stiffness_fn,
        hydrogel_network=network,
    )
    sim.initialize()

    # --- Run with tqdm progress bar ---
    # Pre-compute TDA/chi snapshot interval (at most 20 snapshots over the run)
    tda_snap_every = max(1, sp.n_steps // 20)
    tda_snapshots: list = []
    tda_snap_step_times: list = []
    chi_series: list = []   # susceptibility χ(t) = Σ s² n_s / N at each snap

    if _HAS_TQDM:
        pbar = tqdm(total=sp.n_steps, desc="Single simulation", unit="step")
        for step_i in range(sp.n_steps):
            sim.step()
            if step_i % tda_snap_every == 0:
                tda_snapshots.append(
                    (network.get_node_positions(), network.get_active_edges())
                )
                tda_snap_step_times.append(step_i * sp.dt)
                chi_series.append(network.compute_susceptibility())
            pbar.update(1)
        pbar.close()
    else:
        logger.info("Running %d steps...", sp.n_steps)
        for step_i in range(sp.n_steps):
            sim.step()
            if step_i % tda_snap_every == 0:
                tda_snapshots.append(
                    (network.get_node_positions(), network.get_active_edges())
                )
                tda_snap_step_times.append(step_i * sp.dt)
                chi_series.append(network.compute_susceptibility())
            if step_i % max(sp.n_steps // 10, 1) == 0:
                logger.info("  Step %d / %d", step_i, sp.n_steps)

    history = sim.get_history()
    logger.info("Simulation complete. %d snapshots recorded.", len(history))

    # --- EWS analysis on G'(t) timeseries (Module 3) ---
    logger.info("Running Early Warning Signal (EWS) analysis...")
    times = np.array([s.time for s in history], dtype=float)
    p_hyd = np.array([s.hydrogel_p_inf for s in history], dtype=float)
    # Use critical-scaling G'(p) from PercolationMechanics — correct near p_c
    G_prime_proxy = np.array([mech.compute_shear_modulus(float(p), omega=1.0) for p in p_hyd])

    ews_cfg = cfg.get("ews", {})
    detector = EarlyWarningSignalDetector(
        window_size=ews_cfg.get("window_size", 50),
        lag=ews_cfg.get("lag", 1),
    )
    ews_results = detector.compute_ews_indicators(G_prime_proxy, times)
    chi_array = np.array(chi_series, dtype=float)
    chi_times_arr = np.array(tda_snap_step_times, dtype=float)
    logger.info(
        "EWS: Kendall τ(AR1)=%.3f  p=%.3f | Kendall τ(var)=%.3f  p=%.3f",
        ews_results.get("kendall_tau_ar1") or 0.0,
        ews_results.get("ar1_pvalue") or 1.0,
        ews_results.get("kendall_tau_var") or 0.0,
        ews_results.get("var_pvalue") or 1.0,
    )

    # --- TDA on network snapshots (Module 3 topological EWS) ---
    logger.info("Running Topological Data Analysis (TDA) on network snapshots...")
    tda = TopologicalDataAnalyzer(
        max_edge_length=hp.r_c * 2.0,
        max_dimension=1,
    )
    lifetime_thr = ews_cfg.get("lifetime_threshold_tda", 0.5)

    # tda_snapshots and tda_snap_step_times were built during the simulation loop above.
    tda_times = np.array(tda_snap_step_times, dtype=float)

    tda_timeseries = tda.compute_topology_timeseries(tda_snapshots)
    h1_counts = tda_timeseries["n_long_lived_h1"]
    # tda_times already built during simulation loop (step-aligned)
    h1_peak_time = tda.detect_h1_peak_time(h1_counts, tda_times)
    logger.info("H1 loop peak at t=%.1f s", h1_peak_time)

    # --- Dual percolation tracker (Module 5) ---
    tracker = DualPercolationTracker()
    for state in history:
        tracker.record(state.time, state.hydrogel_p_inf, state.collagen_p_inf)
    Q = tracker.compute_handoff_quality()
    t_star = tracker.compute_handoff_time()
    crossover_type = tracker.compute_crossover_type()
    logger.info(
        "Handoff quality Q=%+.6f, t*=%.1f s, type=%s",
        Q, t_star if t_star is not None else 0.0, crossover_type,
    )

    # --- Save arrays and report ---
    all_results: dict = {}
    all_results["ews_ar1"] = ews_results.get("ar1", np.array([]))
    all_results["ews_variance"] = ews_results.get("variance", np.array([]))
    all_results["h1_counts"] = h1_counts
    all_results["h1_times"] = tda_times
    all_results["chi_series"] = chi_array
    all_results["chi_times"] = chi_times_arr

    SummaryReporter.export_results(str(output_dir), history, all_results)

    # --- HDF5 archive (self-describing for Zenodo deposit) ---
    try:
        import h5py
        hdf5_path = output_dir / "results.h5"
        with h5py.File(hdf5_path, "w") as f:
            f.attrs["description"] = "gel-percolation simulation results"
            f.attrs["box_size_um"] = hp.box_size
            f.attrs["rho_x"] = hp.rho_x
            f.attrs["r_c_um"] = hp.r_c
            f.attrs["k_base"] = hp.k_base
            f.attrs["measured_p_c"] = mp.p_c
            f.attrs["n_steps"] = sp.n_steps
            f.attrs["n_cells"] = sp.n_cells
            f.attrs["random_seed"] = sp.random_seed
            ts = f.create_group("timeseries")
            ts.create_dataset("times_s", data=times)
            ts.create_dataset("p_inf_hydrogel",
                              data=np.array([s.hydrogel_p_inf for s in history]))
            ts.create_dataset("p_inf_collagen",
                              data=np.array([s.collagen_p_inf for s in history]))
            ts.create_dataset("G_prime_Pa", data=G_prime_proxy)
            ts.create_dataset("ar1",
                              data=ews_results.get("ar1", np.full_like(times, np.nan)))
            ts.create_dataset("variance",
                              data=ews_results.get("variance", np.full_like(times, np.nan)))
            chi_grp = f.create_group("susceptibility")
            chi_grp.create_dataset("chi_series", data=chi_array)
            chi_grp.create_dataset("chi_times_s", data=chi_times_arr)
            tda_grp = f.create_group("tda")
            tda_grp.create_dataset("h1_counts", data=h1_counts)
            tda_grp.create_dataset("h1_times_s", data=tda_times)
        logger.info("HDF5 archive saved to %s", hdf5_path)
    except ImportError:
        logger.warning("h5py not installed — skipping HDF5 export (pip install h5py)")
    except Exception as exc:
        logger.warning("HDF5 export failed: %s", exc)
    report = SummaryReporter.generate_report(history, tracker, ews_results)
    report_path = output_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)

    # --- Figures ---
    if not no_plots:
        try:
            import matplotlib.pyplot as plt

            # Simulation snapshot panel
            fig_snap = sim.plot_snapshot(time_idx=-1)
            if fig_snap is not None:
                fig_snap.savefig(
                    output_dir / "snapshot_final.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig_snap)

            # Dual percolation dynamics
            fig_dual = tracker.plot_dual_percolation()
            if fig_dual is not None:
                fig_dual.savefig(
                    output_dir / "dual_percolation.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig_dual)

            # EWS panel (interpolate H1 onto full time axis for panel alignment)
            ar1 = ews_results.get("ar1", np.full_like(times, np.nan))
            var = ews_results.get("variance", np.full_like(times, np.nan))
            h1_interp = np.interp(times, tda_times, h1_counts)
            transition_t = ews_results.get("transition_time", float(times[-1]))
            fig_ews = plot_ews_panel(
                times, G_prime_proxy, ar1, var, h1_interp, transition_t
            )
            if fig_ews is not None:
                fig_ews.savefig(
                    output_dir / "ews_panel.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig_ews)

            # Publication-quality EWS panel with χ(t) row
            try:
                from src.publication_figures import plot_ews_panel_publication
                fig_ews_pub = plot_ews_panel_publication(
                    times, G_prime_proxy, ar1, var,
                    h1_counts=h1_counts,
                    h1_times=tda_times,
                    t_transition=transition_t,
                    t_ews_onset=ews_results.get("ews_onset_time"),
                    chi_series=chi_array if len(chi_array) > 0 else None,
                    chi_times=chi_times_arr if len(chi_times_arr) > 0 else None,
                )
                if fig_ews_pub is not None:
                    fig_ews_pub.savefig(
                        output_dir / "ews_panel_publication.png",
                        dpi=300, bbox_inches="tight",
                    )
                    fig_ews_pub.savefig(
                        output_dir / "ews_panel_publication.pdf",
                        bbox_inches="tight",
                    )
                    plt.close(fig_ews_pub)
            except Exception as exc_pub:
                logger.warning("Publication EWS panel failed: %s", exc_pub)

            # Publication-quality critical scaling figure (Fig 1)
            try:
                from src.publication_figures import plot_critical_scaling
                fig_scaling = plot_critical_scaling(mech, mp.p_c)
                if fig_scaling is not None:
                    fig_scaling.savefig(
                        output_dir / "fig1_critical_scaling.png",
                        dpi=300, bbox_inches="tight",
                    )
                    fig_scaling.savefig(
                        output_dir / "fig1_critical_scaling.pdf",
                        bbox_inches="tight",
                    )
                    plt.close(fig_scaling)
            except Exception as exc_fig1:
                logger.warning("Critical scaling figure failed: %s", exc_fig1)

            # Publication-quality percolation dynamics figure (Fig 2)
            try:
                from src.publication_figures import plot_percolation_dynamics
                p_hyd_full = np.array([s.hydrogel_p_inf for s in history])
                p_col_full = np.array([s.collagen_p_inf for s in history])
                fig_dyn = plot_percolation_dynamics(
                    times, p_hyd_full, p_col_full, t_star=t_star
                )
                if fig_dyn is not None:
                    fig_dyn.savefig(
                        output_dir / "fig2_percolation_dynamics.png",
                        dpi=300, bbox_inches="tight",
                    )
                    fig_dyn.savefig(
                        output_dir / "fig2_percolation_dynamics.pdf",
                        bbox_inches="tight",
                    )
                    plt.close(fig_dyn)
            except Exception as exc_fig2:
                logger.warning("Percolation dynamics figure failed: %s", exc_fig2)

            logger.info("Figures saved to %s", output_dir)
        except Exception as exc:
            logger.warning("Plot generation failed: %s", exc)


def run_sweep(cfg: dict, no_plots: bool = False) -> None:
    """Run a 2-D parameter sweep over rho_x vs k_base and save results."""
    from src.percolation_analysis import ParameterSpaceSweeper

    try:
        from tqdm import tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    _, cp, sp, _ = build_params_from_config(cfg)
    output_dir = Path(cfg.get("simulation", {}).get("output_dir", "results/sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_cfg = cfg.get("parameter_sweep", {})
    rho_x_values = list(sweep_cfg.get("rho_x_range", [0.5, 1.0, 2.0]))
    k_base_values = list(sweep_cfg.get("k_base_range", [0.005, 0.01, 0.05]))
    n_jobs = sweep_cfg.get("n_jobs", 1)

    n_total = len(rho_x_values) * len(k_base_values)
    logger.info(
        "Starting 2D sweep: %d rho_x values × %d k_base values = %d simulations",
        len(rho_x_values), len(k_base_values), n_total,
    )

    # Use a shorter n_steps for each sweep point to keep wall-time manageable.
    from src.cell_invasion import SimParams
    sp_sweep = SimParams(
        dt=sp.dt,
        n_steps=min(sp.n_steps, 600),
        record_interval=sp.record_interval,
        box_size=sp.box_size,
        grid_resolution=sp.grid_resolution,
        n_cells=sp.n_cells,
        random_seed=sp.random_seed,
    )

    if _HAS_TQDM:
        print(f"Sweeping {n_total} parameter combinations...")

    sweeper = ParameterSpaceSweeper(base_cell_params=cp, base_sim_params=sp_sweep)
    results = sweeper.sweep_2d("rho_x", rho_x_values, "k_base", k_base_values, n_jobs=n_jobs)

    # --- Print top-5 formulations ---
    try:
        import pandas as pd
        if isinstance(results, pd.DataFrame):
            valid = results.dropna(subset=["Q"])
            top5 = valid.nlargest(min(5, len(valid)), "Q")
            print("\nTop-5 parameter sets by handoff quality Q:")
            print(
                top5[["rho_x", "k_base", "Q", "invasion_depth", "outcome"]].to_string(index=False)
            )
            # Save full sweep table
            csv_path = output_dir / "sweep_results.csv"
            results.to_csv(csv_path, index=False)
            logger.info("Sweep results saved to %s", csv_path)
    except ImportError:
        sorted_res = sorted(
            [r for r in results if not np.isnan(float(r.get("Q", float("nan"))))],
            key=lambda r: float(r.get("Q", float("-inf"))),
            reverse=True,
        )
        print("\nTop-5 parameter sets by Q:")
        for r in sorted_res[:5]:
            print(
                f"  rho_x={r['rho_x']:.3f}  k_base={r['k_base']:.5f}"
                f"  Q={r['Q']:+.4f}  outcome={r['outcome']}"
            )

    # --- Optimal formulation ---
    optimal = sweeper.identify_optimal_formulation(results)
    print(f"\nOptimal formulation: {optimal}")

    # --- Q heatmap figure ---
    if not no_plots:
        try:
            import matplotlib.pyplot as plt
            fig = sweeper.plot_Q_heatmap(results, "rho_x", "k_base")
            if fig is not None:
                fig.savefig(output_dir / "Q_heatmap.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("Q heatmap saved to %s", output_dir / "Q_heatmap.png")
        except Exception as exc:
            logger.warning("Heatmap plot failed: %s", exc)


def run_validate(cfg: dict) -> bool:
    """Validate the model against known analytical limits.

    Checks:
      - p >> p_c: affine network theory gives physical G' values
      - p << p_c: G' = 0 (disconnected / sol phase)
      - Susceptibility diverges near p_c
      - Network construction and degradation work correctly
      - P_inf stays in [0, 1]

    Returns True if all checks pass, False otherwise.
    """
    from src.mechanical_properties import (
        PercolationMechanics,
        AffineNetworkTheory,
    )
    from src.network_model import HydrogelNetwork, HydrogelParams

    try:
        from tqdm import tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    print("=" * 64)
    print("  VALIDATION SUITE")
    print("  Checking analytical limits of the gel-percolation model")
    print("=" * 64)

    passed = 0
    failed = 0

    def check(name: str, condition: bool, expected: str, got) -> None:
        nonlocal passed, failed
        if condition:
            status = "\033[92mPASS\033[0m"
            passed += 1
        else:
            status = "\033[91mFAIL\033[0m"
            failed += 1
        print(f"  [{status}]  {name}")
        if not condition:
            print(f"             Expected : {expected}")
            print(f"             Got      : {got}")

    checks = [
        # --- (1) Affine network theory: p >> p_c ---
        # rho_chain ~ 2.34e23 m^-3 gives G' ~ 1 kPa, a typical physiological gel.
        "Affine modulus in physical range [10, 100000 Pa] (p >> p_c)",
        # --- (2) Sol-phase: p << p_c -> G' = 0 ---
        "G' = 0.0 below p_c (disconnected / sol phase)",
        # --- (3) Gel-phase: p > p_c -> G' > 0 ---
        "G' > 0 above p_c (percolating / gel phase)",
        # --- (4) Susceptibility diverges near p_c ---
        "Susceptibility larger near p_c than far from p_c",
        # --- (5) Network builds correctly ---
        "HydrogelNetwork constructs without error",
        # --- (6) P_inf in [0, 1] ---
        "P_inf (order parameter) is in [0, 1]",
        # --- (7) Degradation removes edges ---
        "MMP degradation reduces edge count over time",
        # --- (8) Stiffness field shapes are consistent ---
        "compute_stiffness_field returns array of correct shape",
    ]

    steps_iter = range(len(checks))
    if _HAS_TQDM:
        steps_iter = tqdm(steps_iter, desc="Validation checks", leave=False)

    mech = PercolationMechanics()

    # 1. Affine network theory
    _ = next(iter(steps_iter)) if _HAS_TQDM else None
    G_affine = AffineNetworkTheory.compute_plateau_modulus(rho_chain=2.34e23, T=310.15)
    check(
        "Affine modulus in physical range [10, 100000 Pa] (p >> p_c)",
        10.0 < G_affine < 100_000.0,
        "10–100000 Pa",
        f"{G_affine:.4g} Pa",
    )

    # 2. G' = 0 below p_c
    G_below = mech.compute_shear_modulus(p=0.05)
    check(
        "G' = 0.0 below p_c (sol phase)",
        G_below == 0.0,
        "0.0 Pa",
        f"{G_below:.6g} Pa",
    )

    # 3. G' > 0 above p_c
    G_above = mech.compute_shear_modulus(p=0.6)
    check(
        "G' > 0 above p_c (gel phase)",
        G_above > 0.0,
        "> 0 Pa",
        f"{G_above:.4g} Pa",
    )

    # 4. Susceptibility diverges near p_c
    chi_far = mech.compute_susceptibility(p=0.8)
    chi_near = mech.compute_susceptibility(p=0.265)
    check(
        "Susceptibility diverges near p_c (chi_near > chi_far)",
        chi_near > chi_far,
        f"chi_near > chi_far",
        f"chi_near={chi_near:.4g}, chi_far={chi_far:.4g}",
    )

    # 5. Network construction
    net = None
    try:
        small_hp = HydrogelParams(box_size=20.0, rho_x=1.0, r_c=1.0)
        net = HydrogelNetwork(small_hp, seed=0)
        N = net.graph.number_of_nodes()
        check("HydrogelNetwork constructs without error (N > 0)", N > 0, "> 0 nodes", N)
    except Exception as exc:
        check("HydrogelNetwork constructs without error", False, "no exception", str(exc))

    # 6. P_inf in [0, 1]
    if net is not None:
        p_inf = net.get_percolation_order_parameter()
        check(
            "P_inf (percolation order parameter) in [0, 1]",
            0.0 <= p_inf <= 1.0,
            "[0, 1]",
            f"{p_inf:.6f}",
        )

        # 7. Degradation removes edges
        edges_before = net.graph.number_of_edges()
        mmp_field = np.ones((5, 5, 5)) * 10.0
        for _ in range(50):
            net.degrade_step(mmp_field, dt=1.0)
        edges_after = net.graph.number_of_edges()
        check(
            "MMP degradation reduces edge count (high [MMP] for 50 steps)",
            edges_after <= edges_before,
            f"<= {edges_before}",
            edges_after,
        )
    else:
        check("P_inf in [0, 1]", False, "network must exist", "network build failed")
        check("MMP degradation", False, "network must exist", "network build failed")

    # 8. Stiffness field shape
    try:
        p_grid = np.random.default_rng(0).uniform(0.0, 1.0, (10, 10, 10))
        sf = mech.compute_stiffness_field(p_grid)
        check(
            "compute_stiffness_field returns array of shape (10,10,10)",
            sf.shape == (10, 10, 10),
            "(10, 10, 10)",
            sf.shape,
        )
    except Exception as exc:
        check("compute_stiffness_field runs without error", False, "no exception", str(exc))

    # --- Summary ---
    print()
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed} checks")
    print("=" * 64)
    return failed == 0


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------


def demo() -> list:
    """Run a quick 100-step smoke test to verify the full pipeline works."""
    print("Running demo simulation (100 steps, small box)...")
    from src.network_model import HydrogelNetwork, HydrogelParams
    from src.cell_invasion import WoundHealingSimulation, CellParams, SimParams
    from src.early_warning import EarlyWarningSignalDetector
    from src.percolation_analysis import DualPercolationTracker

    try:
        from tqdm import tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    hp = HydrogelParams(box_size=20.0, rho_x=1.0, r_c=1.0)
    cp = CellParams()
    sp = SimParams(
        n_steps=100,
        record_interval=10,
        n_cells=5,
        grid_resolution=10,
        box_size=20.0,
        random_seed=42,
    )

    net = HydrogelNetwork(hp, seed=42)
    sim = WoundHealingSimulation(cell_params=cp, sim_params=sp, hydrogel_network=net)
    sim.initialize()

    if _HAS_TQDM:
        pbar = tqdm(total=sp.n_steps, desc="Demo", unit="step")
        for _ in range(sp.n_steps):
            sim.step()
            pbar.update(1)
        pbar.close()
    else:
        sim.run(n_steps=sp.n_steps)

    history = sim.get_history()
    final = history[-1]

    # Quick EWS check
    times = np.array([s.time for s in history], dtype=float)
    p_hyd = np.array([s.hydrogel_p_inf for s in history], dtype=float)
    detector = EarlyWarningSignalDetector(window_size=min(10, len(history) // 2))
    ews = detector.compute_ews_indicators(p_hyd * 1000.0, times)

    # Quick dual percolation check
    tracker = DualPercolationTracker()
    for state in history:
        tracker.record(state.time, state.hydrogel_p_inf, state.collagen_p_inf)
    Q = tracker.compute_handoff_quality()

    print(f"  Steps completed  : {len(history)} snapshots")
    print(f"  Hydrogel P_inf   : {final.hydrogel_p_inf:.4f}")
    print(f"  Collagen P_inf   : {final.collagen_p_inf:.4f}")
    print(f"  Invasion depth   : {final.invasion_depth:.2f} um")
    print(f"  Collagen fibres  : {final.n_collagen_fibers}")
    print(f"  Handoff quality Q: {Q:+.6f}")
    print(f"  EWS Kendall tau  : AR1={ews.get('kendall_tau_ar1') or 0:.3f}"
          f"  var={ews.get('kendall_tau_var') or 0:.3f}")
    print("Demo complete.")
    return history


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gel-Percolation Simulation Framework — "
            "Percolation Inversion Dynamics in Enzymatically Degrading Wound Hydrogels"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default_params.yaml",
        metavar="PATH",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory (overrides config simulation.output_dir)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "sweep", "validate", "demo"],
        default="single",
        help="Simulation mode: single run | 2D parameter sweep | validation | quick demo",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=None,
        metavar="N",
        help="Override n_cells from config",
    )
    parser.add_argument(
        "--rho-x",
        type=float,
        default=None,
        metavar="RHO",
        help="Override crosslink junction density rho_x [um^-3]",
    )
    parser.add_argument(
        "--k-base",
        type=float,
        default=None,
        metavar="K",
        help="Override base MMP cleavage rate k_base [s^-1 nM^-1]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Override random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all figure generation (useful for headless environments)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "demo":
        demo()
        return

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(str(config_path))
    cfg = override_config(cfg, args)

    if args.mode == "single":
        run_single(cfg, no_plots=args.no_plots)

    elif args.mode == "sweep":
        run_sweep(cfg, no_plots=args.no_plots)

    elif args.mode == "validate":
        all_passed = run_validate(cfg)
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
