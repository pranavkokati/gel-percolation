"""Rebuild the figures in figures/ from the saved result tables.

Run: PYTHONPATH=. python scripts/make_figures.py
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 8, "legend.frameon": False, "figure.dpi": 150})
BLUE, ORANGE, GREY = "#1f5f99", "#c8641e", "#777777"


def fig_gap():
    d = pd.read_csv("results/thermal/rigidity_gap_vs_eps.csv")
    d = d[(d.tau == 0.01) & (d.box == 12)].sort_values("eps")
    pos, zero = d[d.eps > 0], d[d.eps == 0]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.errorbar(pos.eps, pos.gap, yerr=[pos.gap - pos.lo, pos.hi - pos.gap], fmt="o-", color=BLUE, ms=3.5, lw=1, capsize=2)
    x0 = pos.eps.min() / 3
    ax.errorbar([x0], zero.gap, yerr=[zero.gap - zero.lo, zero.hi - zero.gap], fmt="s", mfc="white", color=BLUE, ms=4, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("transverse stiffness ratio $\\epsilon$ (open square: $\\epsilon$ = 0)")
    ax.set_ylabel("rigidity gap, $p_{r}-p_{c}$")
    ax.set_title("The gap closes as $\\epsilon$ grows (box 12, 1 % criterion)", loc="left", fontsize=7)
    fig.tight_layout(); fig.savefig("figures/fig_gap_vs_eps.png", dpi=300); plt.close(fig)


def fig_agreement():
    cm = pd.read_csv("results/rgg_production/mf_vs_network_cell_map.csv")
    eps = sorted(cm.eps.unique())
    fig, axes = plt.subplots(1, len(eps), figsize=(7.2, 2.5), sharey=True)
    for ax, e in zip(axes, eps):
        c = cm[cm.eps == e].pivot(index="kd", columns="kb", values="misclass_frac").sort_index(ascending=False)
        im = ax.imshow(c.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(c.shape[1])); ax.set_xticklabels([f"{v:g}" for v in c.columns], fontsize=6)
        ax.set_yticks(range(c.shape[0])); ax.set_yticklabels([f"{v:g}" for v in c.index], fontsize=6)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                v = c.values[i, j]
                ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=6, color="white" if v > 0.5 else "black")
        ax.set_title(f"$\\epsilon$ = {e:g}", loc="left", fontsize=8)
        ax.set_xlabel("degradation rate $k_{base}$")
    axes[0].set_ylabel("deposition rate $k_{dep}$")
    fig.suptitle("Percent of runs where mean-field and network disagree on safe/unsafe (12 seeds per cell)", x=0.01, ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("figures/fig_agreement_map.png", dpi=300); plt.close(fig)


def fig_fcc():
    d = np.load("results/fcc_crosscheck/solver_validation_scan.npz")
    ps, G, s, (A, pr, f) = d["ps"], d["Gmean"], d["Gstd"], d["popt"]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.errorbar(ps, G, yerr=s, fmt="o", ms=3, color=BLUE, label="solver, FCC L = 8")
    pp = np.linspace(pr, ps.max(), 100)
    ax.plot(pp, A * (pp - pr) ** f, color=ORANGE, label=f"fit: $p_r$ = {pr:.3f}, $f$ = {f:.2f}")
    ax.axvline(0.495, color=GREY, ls=":", lw=1); ax.text(0.497, 0.02, "published $p_r$ 0.495", fontsize=6, color=GREY, transform=ax.get_xaxis_transform())
    ax.set_xlabel("bond occupation $p$"); ax.set_ylabel("shear modulus $G$")
    ax.set_title("Solver reproduces the FCC rigidity threshold", loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="lower right"); fig.tight_layout(); fig.savefig("figures/fig_fcc_validation.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig_gap(); fig_agreement(); fig_fcc()
