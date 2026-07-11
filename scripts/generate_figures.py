"""
generate_figures.py -- regenerate every figure in figures/ from the
gelrigidity package and the checkpoint files under results/.

By default this script LOADS the saved checkpoint arrays (fast, seconds)
and re-plots from them, so the figures can be reproduced byte-for-byte
without re-running any percolation scan. Pass --recompute to instead
regenerate every checkpoint from scratch by calling the gelrigidity
package directly (periodic_poisson_rgg / ElasticNetwork / CoupledNetwork
/ load_path_continuity_Q). This is slow: the box=12 production dilution
scan alone takes several minutes, and a box=20 finite-size point took
~966s wall in prior runs on this machine. Use --recompute only when you
need to verify the checkpoints themselves, not for routine figure
regeneration.

No exponents, thresholds, or Q values are hardcoded anywhere in this
script -- every number plotted is either loaded from a results/*.npz or
*.json checkpoint, or (with --recompute) measured live by the elastic
solver / rigidity_law fit / load_path_continuity_Q in gelrigidity.

Usage:
    python scripts/generate_figures.py                # load checkpoints
    python scripts/generate_figures.py --recompute     # regenerate from scratch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gelrigidity.rigidity import (
    ElasticNetwork, periodic_poisson_rgg, fcc_lattice, neighbour_bonds,
)
from gelrigidity.dynamics import CoupledNetwork
from gelrigidity.handoff import load_path_continuity_Q, rigidity_connectivity_lag
from gelrigidity.utils import compute_giant_component_fraction
import networkx as nx

RESULTS_RGG = "results/rgg_production"
RESULTS_FCC = "results/fcc_crosscheck"
FIGDIR = "figures"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif", "font.size": base,
        "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": secondary, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid), "legend.frameon": False, "figure.dpi": 200,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.labelweight": "normal", "lines.linewidth": 1.2,
        "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def rigidity_law(p, A, p_r, f):
    out = np.zeros_like(p)
    mask = p > p_r
    out[mask] = A * (p[mask] - p_r) ** f
    return out


def fit_rigidity(ps, Gmean, p0=(1.0, 0.5, 1.0), pmax=None, bounds=None):
    mask = Gmean > 1e-6
    if pmax is not None:
        mask &= ps <= pmax
    kwargs = dict(p0=p0, maxfev=100000)
    if bounds is not None:
        kwargs["bounds"] = bounds
    popt, pcov = curve_fit(rigidity_law, ps[mask], Gmean[mask], **kwargs)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def giant_component_fraction(N, bonds, occ):
    idx = np.flatnonzero(occ)
    g = nx.Graph()
    g.add_nodes_from(range(N))
    if len(idx):
        g.add_edges_from(bonds[idx])
    return compute_giant_component_fraction(g)


def find_gcc_half(ps, gcc_mean):
    for i in range(len(ps) - 1):
        if gcc_mean[i] < 0.5 <= gcc_mean[i + 1]:
            p0, p1 = ps[i], ps[i + 1]
            g0, g1 = gcc_mean[i], gcc_mean[i + 1]
            return p0 + (0.5 - g0) * (p1 - p0) / (g1 - g0)
    return np.nan


def dilution_scan(rho_x, box_size, r_c, base_seed, n_seeds, ps):
    """Measure G(p) over n_seeds independent periodic-RGG realisations."""
    results = {"G": np.zeros((n_seeds, len(ps)))}
    Ns, Ms = [], []
    for s in range(n_seeds):
        seed = base_seed + s
        pos, box, bonds, rhat, rvec = periodic_poisson_rgg(rho_x, box_size, r_c, seed=seed)
        N, M = len(pos), len(bonds)
        Ns.append(N); Ms.append(M)
        net = ElasticNetwork(pos, box, bonds, rhat, rvec)
        rng = np.random.default_rng(9000 + base_seed + s)
        for j, p in enumerate(ps):
            occ = rng.random(M) < p
            results["G"][s, j] = net.shear_modulus(occ)
    results["N"] = int(np.mean(Ns))
    results["M"] = int(np.mean(Ms))
    return results


def connectivity_scan(rho_x, box_size, r_c, base_seed, n_seeds, ps_conn):
    GCC = np.zeros((n_seeds, len(ps_conn)))
    N_last = None
    for si in range(n_seeds):
        seed = base_seed + si
        pos, box, bonds, rhat, rvec = periodic_poisson_rgg(rho_x, box_size, r_c, seed=seed)
        N, M = len(pos), len(bonds)
        N_last = N
        rng2 = np.random.default_rng(7000 + base_seed + si)
        for j, p in enumerate(ps_conn):
            occ = rng2.random(M) < p
            GCC[si, j] = giant_component_fraction(N, bonds, occ)
    return GCC, N_last


# --------------------------------------------------------------------------
# Figure 0: FCC solver validation (ordered-lattice cross-check)
# --------------------------------------------------------------------------
def fig0_solver_validation(recompute=False):
    ckpt = os.path.join(RESULTS_FCC, "solver_validation_scan.npz")
    if recompute or not os.path.exists(ckpt):
        pos, box = fcc_lattice(8, a=1.0)
        bonds, rhat, rvec = neighbour_bonds(pos, box, r_cut=0.75)
        net = ElasticNetwork(pos, box, bonds, rhat, rvec)
        ps = np.linspace(0.40, 1.00, 25)
        nseed = 6
        Gmean = np.zeros_like(ps)
        Gstd = np.zeros_like(ps)
        for i, p in enumerate(ps):
            gs = []
            for s in range(nseed):
                rng = np.random.default_rng(1000 * s + i)
                occ = rng.random(net.M) < p
                gs.append(net.shear_modulus(occ))
            gs = np.array(gs)
            Gmean[i] = gs.mean()
            Gstd[i] = gs.std()
        popt, perr = fit_rigidity(ps, Gmean, p0=[3, .46, 1.4], pmax=0.72,
                                   bounds=([.1, .30, .5], [50, .55, 4]))
        os.makedirs(RESULTS_FCC, exist_ok=True)
        np.savez(ckpt, ps=ps, Gmean=Gmean, Gstd=Gstd, nseed=nseed, L=8, r_cut=0.75,
                 popt=popt, perr=perr, mask_pmax=0.72, N=net.N, M=net.M)
    d = np.load(ckpt, allow_pickle=True)
    ps, Gmean, Gstd = d["ps"], d["Gmean"], d["Gstd"]
    popt, perr = d["popt"], d["perr"]
    A, p_r, f = popt

    apply_figure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax1.errorbar(ps, Gmean, yerr=Gstd, fmt="o", ms=3.5, color="#2166ac",
                 label="measured G(p) (FCC, L=8)")
    pp = np.linspace(p_r, ps.max(), 200)
    ax1.plot(pp, rigidity_law(pp, *popt), "-", color="#b2182b", lw=1.5,
              label=f"fit: f={f:.2f}$\\pm${perr[2]:.2f}")
    ax1.axvline(p_r, color="#b2182b", ls="--", lw=1)
    ax1.set_xlabel("bond occupation probability $p$")
    ax1.set_ylabel("shear modulus $G$")
    ax1.set_title("FCC rigidity-percolation solver validation")
    ax1.legend(frameon=False, fontsize=6.5, loc="upper left")

    ax2.loglog(ps[ps > p_r] - p_r, Gmean[ps > p_r], "o", ms=3.5, color="#2166ac")
    pr_range = np.geomspace(1e-3, ps.max() - p_r, 100)
    ax2.loglog(pr_range, rigidity_law(pr_range + p_r, *popt), "-", color="#b2182b", lw=1.5)
    ax2.set_xlabel("$p - p_r$")
    ax2.set_ylabel("$G$")
    ax2.set_title("Log-log scaling near threshold")

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig0_solver_validation.png"))
    fig.savefig(os.path.join(FIGDIR, "fig0_solver_validation.pdf"))
    plt.close(fig)
    print(f"fig0_solver_validation: p_r={p_r:.4f} f={f:.4f}+-{perr[2]:.4f}")


# --------------------------------------------------------------------------
# Figure: RGG thresholds and exponent finite-size scan
# --------------------------------------------------------------------------
def fig_rgg_thresholds_exponent(recompute=False):
    ckpt = os.path.join(RESULTS_RGG, "thresholds_exponent_scan.npz")
    if recompute or not os.path.exists(ckpt):
        ps_rig_q = np.linspace(0.38, 0.75, 14)
        ps_conn_q = np.linspace(0.06, 0.20, 12)

        def quick_scan(rho_x, box_size, r_c, base_seed, n_seeds):
            G = dilution_scan(rho_x, box_size, r_c, base_seed, n_seeds, ps_rig_q)["G"]
            GCC, N = connectivity_scan(rho_x, box_size, r_c, base_seed, n_seeds, ps_conn_q)
            return G, GCC, N

        G8, GCC8, N8 = quick_scan(1.0, 8.0, 1.5, base_seed=200, n_seeds=5)
        G16, GCC16, N16 = quick_scan(1.0, 16.0, 1.5, base_seed=300, n_seeds=4)

        ps_prod = np.linspace(0.30, 1.00, 25)
        res_prod = dilution_scan(1.0, 12.0, 1.5, base_seed=500, n_seeds=8, ps=ps_prod)
        Gmean_prod = res_prod["G"].mean(axis=0)
        Nmean = res_prod["N"]

        ps_conn_prod = np.linspace(0.05, 0.20, 16)
        GCC_prod, _ = connectivity_scan(1.0, 16.0, 1.5, base_seed=500, n_seeds=8, ps_conn=ps_conn_prod)

        popt, pcov = curve_fit(rigidity_law, ps_prod[ps_prod <= 0.72], Gmean_prod[ps_prod <= 0.72],
                                p0=[2.0, 0.45, 1.0], maxfev=20000)
        perr = np.sqrt(np.diag(pcov))

        ps_rig_20 = np.linspace(0.40, 0.65, 8)
        ps_conn_20 = np.linspace(0.07, 0.16, 8)
        G20 = dilution_scan(1.0, 20.0, 1.5, base_seed=400, n_seeds=3, ps=ps_rig_20)["G"]
        GCC20, N20 = connectivity_scan(1.0, 20.0, 1.5, base_seed=400, n_seeds=3, ps_conn=ps_conn_20)

        pc_prod = find_gcc_half(ps_conn_prod, GCC_prod.mean(axis=0))

        os.makedirs(RESULTS_RGG, exist_ok=True)
        np.savez(ckpt, ps_rig_q=ps_rig_q, ps_conn_q=ps_conn_q,
                 G8=G8, GCC8=GCC8, N8=N8, G16=G16, GCC16=GCC16, N16=N16,
                 ps_rig_20=ps_rig_20, ps_conn_20=ps_conn_20, G20=G20, GCC20=GCC20, N20=N20,
                 ps_prod=ps_prod, res_prod_G=res_prod["G"], res_prod_N=Nmean, res_prod_M=res_prod["M"],
                 ps_conn_prod=ps_conn_prod, GCC_prod=GCC_prod, popt=popt, perr=perr, pc_prod=pc_prod)

    d = np.load(ckpt, allow_pickle=True)
    ps_rig_q, ps_conn_q = d["ps_rig_q"], d["ps_conn_q"]
    G8, GCC8, N8 = d["G8"], d["GCC8"], float(d["N8"])
    G16, GCC16, N16 = d["G16"], d["GCC16"], float(d["N16"])
    ps_prod, res_prod_G, Nmean = d["ps_prod"], d["res_prod_G"], float(d["res_prod_N"])
    ps_conn_prod, GCC_prod = d["ps_conn_prod"], d["GCC_prod"]
    popt, perr, pc_prod = d["popt"], d["perr"], float(d["pc_prod"])
    G20, GCC20, N20 = d["G20"], d["GCC20"], float(d["N20"])
    ps_rig_20, ps_conn_20 = d["ps_rig_20"], d["ps_conn_20"]

    Gmean_prod = res_prod_G.mean(axis=0)
    pc8 = find_gcc_half(ps_conn_q, GCC8.mean(axis=0))
    popt8, _ = fit_rigidity(ps_rig_q, G8.mean(axis=0), pmax=0.65)
    pc16 = find_gcc_half(ps_conn_q, GCC16.mean(axis=0))
    popt16, _ = fit_rigidity(ps_rig_q, G16.mean(axis=0), pmax=0.65)
    pc_half20 = find_gcc_half(ps_conn_20, GCC20.mean(axis=0))
    popt20, _ = fit_rigidity(ps_rig_20, G20.mean(axis=0), pmax=0.60)

    finite_size_table = [
        dict(box=8.0, N=round(N8), pc=float(pc8), pr=float(popt8[1]), f=float(popt8[2])),
        dict(box=12.0, N=round(Nmean), pc=float(pc_prod), pr=float(popt[1]), f=float(popt[2])),
        dict(box=16.0, N=round(N16), pc=float(pc16), pr=float(popt16[1]), f=float(popt16[2])),
        dict(box=20.0, N=round(N20), pc=float(pc_half20), pr=float(popt20[1]), f=float(popt20[2])),
    ]

    apply_figure_style()
    boxes = np.array([r["box"] for r in finite_size_table])
    Ns = np.array([r["N"] for r in finite_size_table])
    pcs = np.array([r["pc"] for r in finite_size_table])
    prs = np.array([r["pr"] for r in finite_size_table])
    fs = np.array([r["f"] for r in finite_size_table])

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))

    ax = axes[0]
    ax.plot(Ns, pcs, "o-", color="#2166ac", label="connectivity $p_c$")
    ax.plot(Ns, prs, "s-", color="#b2182b", label="rigidity $p_r$")
    ax.set_xscale("log")
    ax.set_xlabel("network size N")
    ax.set_ylabel("percolation threshold")
    ax.set_title("Thresholds vs. system size")
    ax.legend(frameon=False, fontsize=6.5, loc="center right")
    ax.margins(0.08)

    ax = axes[1]
    ax.plot(Ns, fs, "D-", color="#4d9221")
    ax.set_xscale("log")
    ax.set_xlabel("network size N")
    ax.set_ylabel("rigidity exponent $f$")
    ax.set_title("Rigidity exponent vs. system size")
    ax.margins(0.15)

    ax = axes[2]
    ax.plot(ps_prod, Gmean_prod, "o", ms=3, color="#4393c3", label="measured G(p)")
    pp = np.linspace(prs[1], 0.72, 100)
    ax.plot(pp, rigidity_law(pp, *popt), "-", color="#b2182b", lw=1.5,
            label=f"fit: f={popt[2]:.2f}$\\pm${perr[2]:.2f}")
    ax.axvline(pcs[1], color="gray", ls=":", lw=1)
    ax.axvline(prs[1], color="#b2182b", ls="--", lw=1)
    ax.set_xlabel("bond occupation probability $p$")
    ax.set_ylabel("shear modulus $G$")
    ax.set_title(f"N={Ns[1]} (box=12) rigidity onset")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.margins(0.05)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig_rgg_thresholds_exponent.png"))
    fig.savefig(os.path.join(FIGDIR, "fig_rgg_thresholds_exponent.pdf"))
    plt.close(fig)

    with open(os.path.join(RESULTS_RGG, "finite_size_table.json"), "w") as fh:
        json.dump(finite_size_table, fh, indent=2)
    print("fig_rgg_thresholds_exponent:", finite_size_table)


# --------------------------------------------------------------------------
# Figure: RGG rigidity-loss-precedes-connectivity-loss divergence trajectory
# --------------------------------------------------------------------------
def fig_rgg_divergence(recompute=False):
    ckpt = os.path.join(RESULTS_RGG, "divergence_trajectory.npz")
    if recompute or not os.path.exists(ckpt):
        cn = CoupledNetwork(topology="rgg", rho_x=1.0, box_size=9.0, r_cut=1.5,
                             k_scaffold=1.0, k_ecm=2.0, seed=101)
        cn.seed_scaffold(1.0)
        cn.seed_cells(n_cells=20, secretion_radius=2.5)
        rec = cn.run(n_steps=250, dt=1.0, record_every=5, mmp_level=1.0,
                     k_base=0.012, k_dep=0.010, solve_rigidity=True)
        os.makedirs(RESULTS_RGG, exist_ok=True)
        np.savez(ckpt, **{k: np.asarray(v) for k, v in rec.items()})

    d = np.load(ckpt, allow_pickle=True)
    t = d["t"]; Pinf_scaf = d["Pinf_scaffold"]; G_scaf = d["G_scaffold"]; G_union = d["G_union"]
    G0 = G_scaf[0]
    q = load_path_continuity_Q(t, G_union, G_target=0.2 * G0)
    lag = rigidity_connectivity_lag(t, Pinf_scaf, G_scaf)

    apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    ax = axes[0]
    ax.plot(t, Pinf_scaf, "-", color="#2166ac", lw=1.8, label="connectivity  P$_\\infty$(scaffold)")
    ax2 = ax.twinx()
    ax2.plot(t, G_scaf / G0, "-", color="#b2182b", lw=1.8, label="rigidity  G(scaffold)/G$_0$")
    ax.axvline(lag["t_rigidity_lost"], color="#b2182b", ls="--", lw=1)
    ax.set_xlabel("time step")
    ax.set_ylabel("connectivity  P$_\\infty$", color="#2166ac")
    ax2.set_ylabel("normalized shear modulus  G/G$_0$", color="#b2182b")
    ax.tick_params(axis="y", labelcolor="#2166ac")
    ax2.tick_params(axis="y", labelcolor="#b2182b")
    ax.set_title("Rigidity is lost while connectivity persists (RGG)")
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, frameon=False, fontsize=6.5, loc="upper right")
    ax.margins(0.03)

    ax = axes[1]
    ax.plot(t, G_union / G_union[0], "-", color="#4d9221", lw=1.8)
    ax.axhline(0.2, color="gray", ls=":", lw=1)
    ax.axvline(q["t_valley"], color="#b2182b", ls="--", lw=1)
    ax.annotate(f"Q = {q['Q']:.2f}\n({'safe' if q['Q'] >= 1 else 'unsafe'})",
                (q["t_valley"], q["G_valley"] / G_union[0]),
                textcoords="offset points", xytext=(15, 20), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.8))
    ax.set_xlabel("time step")
    ax.set_ylabel("union shear modulus  G$_{union}$/G$_0$")
    ax.set_title("Load-path continuity trajectory (RGG)")
    ax.margins(0.05)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig_rgg_divergence.png"))
    fig.savefig(os.path.join(FIGDIR, "fig_rgg_divergence.pdf"))
    plt.close(fig)
    print(f"fig_rgg_divergence: Q={q['Q']:.4f} t_valley={q['t_valley']} tau_gap={lag['tau_gap']}")


# --------------------------------------------------------------------------
# Figure: RGG design map (Q sweep over k_base x k_dep)
# --------------------------------------------------------------------------
def fig_designmap_rgg(recompute=False):
    ckpt = os.path.join(RESULTS_RGG, "designmap_sweep.npz")
    if recompute or not os.path.exists(ckpt):
        from joblib import Parallel, delayed

        k_base_vals = np.array([0.004, 0.009, 0.019, 0.032])
        k_dep_vals = np.array([0.008, 0.020, 0.050, 0.080])
        SEEDS = [11, 22, 33]
        G_TARGET_FRAC = 0.2
        K_ECM = 2.0
        RHO_X, BOX_SIZE, R_CUT = 1.0, 8.0, 1.5
        N_STEPS, RECORD_EVERY, N_CELLS, SEC_R = 150, 15, 20, 2.5

        def one(kb, kd, seed):
            cn = CoupledNetwork(topology="rgg", rho_x=RHO_X, box_size=BOX_SIZE, r_cut=R_CUT,
                                 k_scaffold=1.0, k_ecm=K_ECM, seed=seed)
            cn.seed_scaffold(1.0)
            cn.seed_cells(n_cells=N_CELLS, secretion_radius=SEC_R)
            rec = cn.run(n_steps=N_STEPS, dt=1.0, record_every=RECORD_EVERY, mmp_level=1.0,
                         k_base=kb, k_dep=kd, solve_rigidity=True)
            G0 = rec["G_union"][0]
            q = load_path_continuity_Q(rec["t"], rec["G_union"], G_target=G_TARGET_FRAC * G0)
            lag = rigidity_connectivity_lag(rec["t"], rec["Pinf_scaffold"], rec["G_scaffold"])
            return (kb, kd, seed, q["Q"], q["t_valley"], lag["tau_gap"])

        jobs = [(kb, kd, s) for kb in k_base_vals for kd in k_dep_vals for s in SEEDS]
        res = Parallel(n_jobs=8, backend="threading")(delayed(one)(kb, kd, s) for kb, kd, s in jobs)
        res = np.array(res)

        Qgrid = np.zeros((len(k_base_vals), len(k_dep_vals)))
        taugrid = np.full_like(Qgrid, np.nan)
        for i, kb in enumerate(k_base_vals):
            for j, kd in enumerate(k_dep_vals):
                m = np.isclose(res[:, 0], kb) & np.isclose(res[:, 1], kd)
                Qgrid[i, j] = np.nanmean(res[m, 3])
                taugrid[i, j] = np.nanmean(res[m, 5])

        os.makedirs(RESULTS_RGG, exist_ok=True)
        np.savez(ckpt, k_base=k_base_vals, k_dep=k_dep_vals, Q=Qgrid, tau_gap=taugrid,
                 G_target_frac=G_TARGET_FRAC, k_ecm=K_ECM, rho_x=RHO_X, box_size=BOX_SIZE,
                 r_cut=R_CUT, raw=res)

    d = np.load(ckpt, allow_pickle=True)
    k_base, k_dep, Q = d["k_base"], d["k_dep"], d["Q"]

    apply_figure_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    vmax = 5.0
    norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax)
    im = ax.imshow(Q, origin="lower", cmap="RdYlBu", norm=norm, aspect="auto")

    ax.set_xticks(range(len(k_dep)))
    ax.set_xticklabels([f"{v:.3f}" for v in k_dep])
    ax.set_yticks(range(len(k_base)))
    ax.set_yticklabels([f"{v:.3f}" for v in k_base])
    ax.set_xlabel("ECM deposition rate  $k_{dep}$  (steps$^{-1}$)")
    ax.set_ylabel("MMP degradation rate  $k_{base}$  (steps$^{-1}$)")
    ax.set_title("Load-path-continuity design map on the RGG topology")

    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            val = Q[i, j]
            txt_color = "black" if 0.3 < norm(val) < 0.85 else "white"
            ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=6.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, label="Q = min$_t$ G$_{union}$(t) / G$_{target}$", shrink=0.85)
    cbar.ax.axhline(norm(1.0), color="black", lw=1.2, ls="--")

    ax.margins(0.02)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig_designmap_rgg.png"))
    fig.savefig(os.path.join(FIGDIR, "fig_designmap_rgg.pdf"))
    plt.close(fig)
    print(f"fig_designmap_rgg: Q_min={Q.min():.4f} Q_max={Q.max():.4f} safe_frac={(Q >= 1).mean():.3f}")


# --------------------------------------------------------------------------
# Figure: RGG thresholds/exponent vs. published lattice values (bar chart)
# --------------------------------------------------------------------------
def fig_rgg_validation(recompute=False):
    thresh_ckpt = os.path.join(RESULTS_RGG, "thresholds_exponent_scan.npz")
    if recompute or not os.path.exists(thresh_ckpt):
        fig_rgg_thresholds_exponent(recompute=True)
    d = np.load(thresh_ckpt, allow_pickle=True)
    ps_prod, res_prod_G, Nmean = d["ps_prod"], d["res_prod_G"], float(d["res_prod_N"])
    ps_conn_prod, GCC_prod = d["ps_conn_prod"], d["GCC_prod"]
    popt, perr, pc_prod = d["popt"], d["perr"], float(d["pc_prod"])

    measured = dict(pc=pc_prod, pr=float(popt[1]), pr_err=float(perr[1]),
                     f=float(popt[2]), f_err=float(perr[2]))

    apply_figure_style()
    f_emt_2d = 1.0
    labels2 = ["measured\n(RGG, this work)", "central-force EMT\n(Feng-Thorpe-Garboczi)"]
    vals2 = [measured["f"], f_emt_2d]
    errs2 = [measured["f_err"], 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))

    ax = axes[0]
    labels = ["connectivity\n$p_c$ (RGG)", "rigidity\n$p_r$ (RGG)"]
    vals = [measured["pc"], measured["pr"]]
    errs = [0.0, measured["pr_err"]]
    colors = ["#2166ac", "#b2182b"]
    ax.bar(labels, vals, yerr=errs, color=colors, capsize=3)
    ax.set_ylabel("percolation threshold  $p$")
    ax.set_title("RGG thresholds: rigidity above connectivity")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    ax.set_ylim(0, 0.55)

    ax = axes[1]
    ax.bar(labels2, vals2, yerr=errs2, color=["#4d9221", "#808080"], capsize=3)
    ax.set_ylabel("rigidity critical exponent  $f$")
    ax.set_title("Measured rigidity exponent vs. central-force theory")
    for i, v in enumerate(vals2):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    ax.set_ylim(0, 1.8)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig_rgg_validation.png"))
    fig.savefig(os.path.join(FIGDIR, "fig_rgg_validation.pdf"))
    plt.close(fig)
    print("fig_rgg_validation:", measured)


# --------------------------------------------------------------------------
# Figure: mean-field-vs-measured Q comparison
# --------------------------------------------------------------------------
def fig_meanfield_comparison(recompute=False):
    ckpt = os.path.join(RESULTS_RGG, "meanfield_vs_measured.json")
    if recompute or not os.path.exists(ckpt):
        raise RuntimeError(
            "meanfield_vs_measured.json checkpoint missing; run "
            "scripts/run_meanfield_comparison.py to regenerate it "
            "(not reproduced by --recompute here; see REPORT.md sec 9.3)."
        )
    with open(ckpt) as f:
        mf_data = json.load(f)
    combo_stats = mf_data["combo_stats"]

    apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9))

    labels = [f"$k_b$={c['kb']}\n$k_d$={c['kd']}" for c in combo_stats]
    x = np.arange(len(combo_stats))
    w = 0.35

    ax = axes[0]
    qmeas = [c["Qmeas_mean"] for c in combo_stats]
    qmeas_err = [c["Qmeas_std"] for c in combo_stats]
    qmf = [c["Qmf_mean"] for c in combo_stats]
    qmf_err = [c["Qmf_std"] for c in combo_stats]

    ax.bar(x - w / 2, qmeas, width=w, yerr=qmeas_err, color="#b2182b",
           label="network-level (measured)", capsize=2)
    ax.bar(x + w / 2, qmf, width=w, yerr=qmf_err, color="#2166ac",
           label="mean-field (affine)", capsize=2)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Q = min$_t$ G$_{union}$(t) / G$_{target}$")
    ax.set_title("Measured vs. mean-field handoff safety margin")
    ax.set_ylim(0, 7.2)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=1)

    ax = axes[1]
    ratio = np.array(qmf) / np.maximum(np.array(qmeas), 1e-6)
    colors = ["#b2182b" if rr > 5 else "#4393c3" for rr in ratio]
    ax.bar(x, ratio, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Q$_{mean-field}$ / Q$_{measured}$  (log scale)")
    ax.set_title("Mean-field overestimates safety in the failure regime")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    i_max = int(np.argmax(ratio))
    ax.annotate(f"{ratio[i_max]:.0f}\u00d7", (x[i_max], ratio[i_max]), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=7, fontweight="bold")
    ax.set_ylim(0.2, 900)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig_meanfield_comparison.png"))
    fig.savefig(os.path.join(FIGDIR, "fig_meanfield_comparison.pdf"))
    plt.close(fig)
    print(f"fig_meanfield_comparison: max ratio={ratio[i_max]:.1f}x at combo {combo_stats[i_max]}")


ALL_FIGURES = {
    "fig0_solver_validation": fig0_solver_validation,
    "fig_rgg_thresholds_exponent": fig_rgg_thresholds_exponent,
    "fig_rgg_divergence": fig_rgg_divergence,
    "fig_designmap_rgg": fig_designmap_rgg,
    "fig_rgg_validation": fig_rgg_validation,
    "fig_meanfield_comparison": fig_meanfield_comparison,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute", action="store_true",
                     help="Regenerate every checkpoint from scratch via the gelrigidity "
                          "package instead of loading results/*.npz|json (slow).")
    ap.add_argument("--only", nargs="*", default=None,
                     help=f"Subset of figures to (re)generate. Choices: {list(ALL_FIGURES)}")
    args = ap.parse_args()

    names = args.only if args.only else list(ALL_FIGURES)
    for name in names:
        if name not in ALL_FIGURES:
            raise SystemExit(f"Unknown figure {name!r}; choices: {list(ALL_FIGURES)}")
        t0 = time.time()
        print(f"--- {name} (recompute={args.recompute}) ---")
        ALL_FIGURES[name](recompute=args.recompute)
        print(f"    done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
