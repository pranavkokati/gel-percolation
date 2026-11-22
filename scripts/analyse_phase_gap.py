"""Phase-gap analysis for the transverse-stiffness (eps) extension.
Inputs : results/thermal/G_eps_box{8,10,12}.npz  (G[seed, p, eps], ps, eps, N)
Outputs: results/thermal/rigidity_gap_vs_eps.csv, pr_independent_estimates.csv, collapse_joint_fit.csv
Gap definition: p_on(eps) = first p at which G(p,eps) exceeds tau*G(p=1,eps) (log-linear interpolation);
gap = p_on(eps) - p_on(eps=1)  [eps=1 is the phantom/random-resistor limit, i.e. connectivity-controlled].
Run: PYTHONPATH=. python scripts/analyse_phase_gap.py
"""
import numpy as np, pandas as pd
from scipy.optimize import least_squares, minimize

BOXES = (8, 10, 12)
TAUS = (0.001, 0.01, 0.03)
WINDOWS = [(0.42, 0.6), (0.45, 0.75), (0.5, 0.9), (0.42, 0.9)]
OUT = "results/thermal/"

def load(b):
    d = np.load(OUT + f"G_eps_box{b}.npz")
    return d["G"], d["ps"], d["eps"], int(d["N"])

def p_on_interp(Gp, ps, thr):
    idx = np.flatnonzero(Gp > thr)
    if len(idx) == 0: return np.nan
    i = idx[0]
    if i == 0: return ps[0]
    g0, g1 = max(Gp[i - 1], 1e-30), Gp[i]
    f = (np.log(thr) - np.log(g0)) / (np.log(g1) - np.log(g0))
    return ps[i - 1] + f * (ps[i] - ps[i - 1])

def gap_curves(G, ps, eps, tau):
    ns = G.shape[0]; out = np.zeros((ns, len(eps)))
    for s in range(ns):
        for j in range(len(eps)):
            out[s, j] = p_on_interp(G[s, :, j], ps, tau * G[s, -1, j])
    return out - out[:, [-1]]

def boot_mean(x, nb=2000, seed=1):
    rng = np.random.default_rng(seed); ns = x.shape[0]
    bs = np.array([x[rng.integers(0, ns, ns)].mean(0) for _ in range(nb)])
    return x.mean(0), *np.percentile(bs, [2.5, 97.5], axis=0)

def fit_pr(ps, g, lo, hi):
    m = (ps >= lo) & (ps <= hi) & (g > 0); x = ps[m]; y = np.log(g[m])
    def r(th): pr, f, A = th; return np.log(np.clip(x - pr, 1e-9, None)) * f + A - y
    best = None
    for pr0 in (0.25, 0.32, 0.36, 0.40):
        s = least_squares(r, [pr0, 1.5, 0.0], bounds=([0.1, 0.3, -20], [min(x) - 1e-3, 6, 20]))
        if best is None or s.cost < best.cost: best = s
    return best.x[0], best.x[1]

def main():
    rows = []; pr_rows = []; data = {}
    for b in BOXES:
        G, ps, eps, N = load(b)
        for tau in TAUS:
            m, lo, hi = boot_mean(gap_curves(G, ps, eps, tau))
            for e, a, l, h in zip(eps, m, lo, hi):
                rows.append(dict(box=b, N=N, tau=tau, eps=e, gap=a, lo=l, hi=h))
        data[b] = (ps, eps, G.mean(0))
        gm = G[:, :, 0]                                   # eps = 0 (athermal), seed-resolved
        rng = np.random.default_rng(3)
        for lo_w, hi_w in WINDOWS:
            est = fit_pr(ps, gm.mean(0), lo_w, hi_w)
            bs = np.array([fit_pr(ps, gm[rng.integers(0, len(gm), len(gm))].mean(0), lo_w, hi_w) for _ in range(200)])
            q = np.percentile(bs, [2.5, 97.5], axis=0)
            pr_rows.append(dict(box=b, N=N, win=f"{lo_w}-{hi_w}", pr=est[0], pr_lo=q[0, 0], pr_hi=q[1, 0],
                                f=est[1], f_lo=q[0, 1], f_hi=q[1, 1]))
    pd.DataFrame(rows).to_csv(OUT + "rigidity_gap_vs_eps.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(OUT + "pr_independent_estimates.csv", index=False)
    sel = [e for e in data[12][1] if 1e-4 <= e <= 1e-2]
    def cost(th, boxes):
        pr, y, iphi = th; X = []; Y = []
        for b in boxes:
            ps, eps, Gm = data[b]
            for e in sel:
                j = list(eps).index(e); m = (Gm[:, j] > 0) & (ps < 0.9) & (ps > 0.3)
                X += list((ps[m] - pr) / e ** iphi); Y += list(np.log10(Gm[m, j] / e ** y))
        X = np.array(X); Y = np.array(Y); o = np.argsort(X); X, Y = X[o], Y[o]
        bins = np.quantile(X, np.linspace(0, 1, 41)); idx = np.clip(np.digitize(X, bins) - 1, 0, 39)
        return sum(Y[idx == k].var() * np.sum(idx == k) for k in range(40) if np.sum(idx == k) > 1) / len(Y)
    out = []
    for boxes in [(12,), (8,), (10,), (8, 10, 12)]:
        r = minimize(cost, [0.41, 0.65, 0.34], args=(boxes,), method="Nelder-Mead")
        out.append(dict(boxes=str(boxes), pr=r.x[0], y=r.x[1], inv_phi=r.x[2], cost=r.fun))
    pd.DataFrame(out).to_csv(OUT + "collapse_joint_fit.csv", index=False)
    print(pd.DataFrame(out).round(4).to_string())

if __name__ == "__main__":
    main()
