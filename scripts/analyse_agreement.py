"""Agreement between the network model and the affine mean-field law.

Input : results/rgg_production/full_comparison_box8.npz  (from run_full_comparison.py)
Output: results/rgg_production/full_comparison_per_seed.csv
        results/rgg_production/agreement_metrics.csv    (95 % percentile bootstrap)
        results/rgg_production/mf_vs_network_cell_map.csv
        results/sensitivity/qtarget_sensitivity.csv

Q = min_t G_union(t) / (f * G_union(0)), with f the target
fraction (0.2 by default). Because the minimum includes t = 0, Q cannot exceed
1/f: it equals 1/f exactly when the modulus never falls below its starting
value. With f = 0.2 that ceiling is 5. It is a structural ceiling, not a clip.

Floors: Qm_f and Qf_f are Q floored at 1e-6; logQm and logQf are log10 of Q
floored at 1e-3. The safe/unsafe call (Q >= 1) does not depend on either floor.

The mean-field modulus is the affine law evaluated on the same occupancy
trajectory: G_scaffold_mf (G0 = 1) + G_ecm_mf (G0 = k_ecm = 2), with p_c from
the npz (0.1126).

Run: PYTHONPATH=. python scripts/analyse_agreement.py
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from gelrigidity.mean_field import affine_meanfield_modulus

SRC = "results/rgg_production/"
FLOOR_LOG = 1e-3   # floor used inside log10 ratios; 10 % of eps = 0 runs sit below it
NBOOT = 2000
TARGETS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def trajectories(d):
    pc, ke = float(d["pc"]), float(d["k_ecm"])
    Gn = d["G_union"]
    Gm = affine_meanfield_modulus(d["p_scaffold"], pc) + affine_meanfield_modulus(d["p_ecm"], pc, G0=ke)
    return Gn, Gm, d["meta"]


def q_of(G, frac):
    return G.min(axis=1) / (frac * G[:, 0])      # raw; floors are applied where used


def metric(df, name):
    if name == "misclass":
        return (df.safe_m != df.safe_f).mean()
    if name == "MF_false_safe":
        return ((~df.safe_m) & df.safe_f).mean()
    if name == "MF_false_unsafe":
        return (df.safe_m & ~df.safe_f).mean()
    if name == "spearman":
        return spearmanr(np.maximum(df.Qm, 1e-6), np.maximum(df.Qf, 1e-6))[0]
    if name == "median_dlog10Q":
        return abs(np.median(df.logQm - df.logQf))   # magnitude of the median signed log-ratio
    if name == "frac_|dlog|>1":
        return (np.abs(df.logQm - df.logQf) > 1).mean()


def boot(df, name, seed=1):
    rng = np.random.default_rng(seed); n = len(df)
    b = np.array([metric(df.iloc[rng.integers(0, n, n)], name) for _ in range(NBOOT)])
    return np.percentile(b, [2.5, 97.5])


def per_seed(Gn, Gm, meta, frac=0.2):
    Qm, Qf = q_of(Gn, frac), q_of(Gm, frac)
    df = pd.DataFrame(dict(kb=meta[:, 0], kd=meta[:, 1], seed=meta[:, 2], eps=meta[:, 3], Qm=Qm, Qf=Qf))
    df["safe_m"] = df.Qm >= 1; df["safe_f"] = df.Qf >= 1
    df["logQm"] = np.log10(np.maximum(df.Qm, FLOOR_LOG)); df["logQf"] = np.log10(np.maximum(df.Qf, FLOOR_LOG))
    return df


def main():
    d = np.load(SRC + "full_comparison_box8.npz", allow_pickle=True)
    Gn, Gm, meta = trajectories(d)
    ps = per_seed(Gn, Gm, meta)
    ps["Qm_f"] = np.maximum(ps.Qm, 1e-6); ps["Qf_f"] = np.maximum(ps.Qf, 1e-6)
    ps.to_csv(SRC + "full_comparison_per_seed.csv", index=False)

    rows = []
    for e, g in ps.groupby("eps"):
        g = g.reset_index(drop=True)
        for m in ("misclass", "MF_false_safe", "MF_false_unsafe", "spearman", "median_dlog10Q", "frac_|dlog|>1"):
            lo, hi = boot(g, m)
            rows.append(dict(eps=e, metric=m, est=metric(g, m), lo=lo, hi=hi))
    am = pd.DataFrame(rows); am["interval"] = "95% percentile bootstrap"
    am.to_csv(SRC + "agreement_metrics.csv", index=False)

    cells = []
    for (e, kb, kd), g in ps.groupby(["eps", "kb", "kd"]):
        cells.append(dict(eps=e, kb=kb, kd=kd, n=len(g),
                          misclass_frac=metric(g, "misclass"),
                          MF_falsely_unsafe_frac=metric(g, "MF_false_unsafe"),
                          MF_falsely_safe_frac=metric(g, "MF_false_safe"),
                          median_log10Q_network_minus_MF=float(np.median(g.logQm - g.logQf)),
                          Q_meanfield_med=g.Qf.median(), Q_network_med=g.Qm.median(), kd_over_kb=kd / kb))
    pd.DataFrame(cells).to_csv(SRC + "mf_vs_network_cell_map.csv", index=False)

    sens = []
    for e in sorted(ps.eps.unique()):
        sel = meta[:, 3] == e
        for f in TARGETS:
            g = per_seed(Gn[sel], Gm[sel], meta[sel], f).reset_index(drop=True)
            lo, hi = boot(g, "misclass")
            sens.append(dict(eps=e, target=f, misclass=metric(g, "misclass"),
                             mf_false_safe=metric(g, "MF_false_safe"), mf_false_unsafe=metric(g, "MF_false_unsafe"),
                             lo=lo, hi=hi, net_unsafe=(~g.safe_m).mean(), mf_unsafe=(~g.safe_f).mean()))
    import os; os.makedirs("results/sensitivity", exist_ok=True)
    pd.DataFrame(sens).to_csv("results/sensitivity/qtarget_sensitivity.csv", index=False)
    print(am.round(3).to_string())


if __name__ == "__main__":
    main()
