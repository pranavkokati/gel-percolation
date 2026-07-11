#!/usr/bin/env python
"""N-seed EWS ensemble: report AR1 and susceptibility Kendall τ across seeds.

Usage
-----
    python scripts/run_ensemble.py                    # 10 seeds, 20 µm box, 600 steps
    python scripts/run_ensemble.py --n-seeds 10 --n-steps 600 --box-size 50
    python scripts/run_ensemble.py --help

The 50 µm box run (Action 3) is required before the H₁ comparison figure can be
generated — it gives χ(t) a gradual pre-transition rise rather than a sharp finite-
size jump.  Budget ~3 h for 10 seeds at 50 µm / 600 steps on a laptop CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_seed(seed: int, n_steps: int, box_size: float, k_base: float = 0.001) -> dict:
    """Run one seed and return AR1 τ, susceptibility τ, and transition step."""
    from scipy import stats
    from src.network_model import HydrogelNetwork, HydrogelParams
    from src.mechanical_properties import PercolationMechanics, MechanicsParams
    from src.early_warning import EarlyWarningSignalDetector
    from src.cell_invasion import WoundHealingSimulation, CellParams, SimParams

    hp = HydrogelParams(box_size=box_size, rho_x=1.0, r_c=1.0, k_base=k_base)
    mp_base = MechanicsParams()

    net = HydrogelNetwork(hp, seed=seed)
    measured_p_c = net.measure_percolation_threshold(n_p_points=30, n_trials=3, rng_seed=seed)

    mp = mp_base.__class__(
        p_c=measured_p_c,
        p_crossover=mp_base.p_crossover,
        T=mp_base.T,
        rho_chain_ref=mp_base.rho_chain_ref,
        E_ref=mp_base.E_ref,
        omega_ref=mp_base.omega_ref,
        kB=mp_base.kB,
        exponents=mp_base.exponents,
    )
    mech = PercolationMechanics(mp)

    cp = CellParams()
    sp = SimParams(
        n_steps=n_steps,
        record_interval=10,
        n_cells=5,
        grid_resolution=10,
        box_size=box_size,
        random_seed=seed,
    )

    sim = WoundHealingSimulation(cell_params=cp, sim_params=sp, hydrogel_network=net)
    sim.initialize()

    snap_every = max(1, n_steps // 40)
    chi_series: list = []
    chi_step_times: list = []

    for step_i in range(n_steps):
        sim.step()
        if step_i % snap_every == 0:
            chi_series.append(net.compute_susceptibility())
            chi_step_times.append(step_i * sp.dt)

    history = sim.get_history()
    times = np.array([s.time for s in history])
    p_hyd = np.array([s.hydrogel_p_inf for s in history])
    G_prime = np.array([mech.compute_shear_modulus(float(p), omega=1.0) for p in p_hyd])

    detector = EarlyWarningSignalDetector(window_size=40, detrend=True)
    ews = detector.compute_ews_indicators(G_prime, times)

    tau_ar1 = float(ews.get("kendall_tau_ar1") or 0.0)
    transition_idx = int(ews.get("transition_idx") or len(times))

    chi_arr = np.array(chi_series, dtype=float)
    chi_t = np.array(chi_step_times, dtype=float)
    if transition_idx < len(times):
        t_transition = times[transition_idx]
    else:
        t_transition = times[-1]
    pre_chi = chi_t < t_transition
    tau_chi = float("nan")
    if pre_chi.sum() >= 4:
        tau_chi_val, _ = stats.kendalltau(chi_t[pre_chi], chi_arr[pre_chi])
        tau_chi = float(tau_chi_val)

    return {
        "seed": seed,
        "measured_p_c": measured_p_c,
        "transition_idx": transition_idx,
        "tau_ar1": tau_ar1,
        "tau_chi": tau_chi,
        "ar1_significant": bool(ews.get("ar1_significant", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EWS ensemble over N independent seeds.")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--box-size", type=float, default=20.0,
                        help="Box side length in µm. Use 50 for Action 3 (gradual χ rise).")
    parser.add_argument("--k-base", type=float, default=0.001)
    args = parser.parse_args()

    expected_step = int(-np.log(0.547) / args.k_base) if args.k_base > 0 else 0
    print(f"Running {args.n_seeds} seeds × {args.n_steps} steps in {args.box_size} µm box")
    print(f"k_base = {args.k_base}  (expected transition ~step {expected_step})")
    print()

    results = []
    for seed in range(args.n_seeds):
        print(f"  Seed {seed + 1}/{args.n_seeds} ...", flush=True)
        r = run_seed(seed, args.n_steps, args.box_size, args.k_base)
        results.append(r)
        print(
            f"    p_c={r['measured_p_c']:.3f}  "
            f"transition_step={r['transition_idx']}  "
            f"τ_AR1={r['tau_ar1']:+.3f}  "
            f"τ_χ={r['tau_chi']:+.3f}  "
            f"AR1_sig={r['ar1_significant']}"
        )

    tau_ar1_vals = np.array([r["tau_ar1"] for r in results])
    tau_chi_vals = np.array([r["tau_chi"] for r in results if np.isfinite(r["tau_chi"])])
    n_sig = sum(r["ar1_significant"] for r in results)
    t_steps = [r["transition_idx"] for r in results]

    print()
    print("=" * 60)
    print(f"Ensemble N = {args.n_seeds} seeds")
    print(f"AR1 Kendall τ  : {np.mean(tau_ar1_vals):+.3f} ± {np.std(tau_ar1_vals):.3f}")
    if len(tau_chi_vals) > 0:
        print(f"χ(t) Kendall τ : {np.mean(tau_chi_vals):+.3f} ± {np.std(tau_chi_vals):.3f}")
    print(f"AR1 significant: {n_sig}/{args.n_seeds} seeds")
    print(f"Transition step: {np.mean(t_steps):.0f} ± {np.std(t_steps):.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
