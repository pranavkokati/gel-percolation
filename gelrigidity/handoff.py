"""
handoff.py — Load-path-continuity metric Q for the scaffold->ECM handoff.

The original repository carried TWO mutually inconsistent definitions of the
"handoff quality" Q: a derivative difference dP_col/dt - dP_hyd/dt (units 1/time,
in percolation_analysis.py) and a timing difference (t_fail - t_perc)/t_fail
(dimensionless, in the figure script). Neither was mechanical: both used
CONNECTIVITY of a hand-drawn collagen sigmoid, not the load-bearing state of a
real network. Here Q is redefined once, on rigidity.

Definition
----------
Let G_union(t) be the measured shear modulus of the combined
scaffold+ECM network (from rigidity.ElasticNetwork), and let G_target be the
minimum physiologically required load-bearing modulus of the application (an
EXTERNAL requirement, not a model parameter). The load-path-continuity metric is

    Q = min_t  G_union(t) / G_target                     (over the remodeling window)

  * Q >= 1  : a spanning stress-bearing path is maintained at every instant;
              the construct never falls below the required load-bearing modulus.
              The handoff is SAFE.
  * Q < 1   : there exists a floppy window where the composite cannot carry the
              required load; scaffold rigidity is lost before ECM rigidity is
              established. The handoff FAILS, and (1 - Q) measures how deep the
              mechanical valley is relative to requirement.

Why this is well-posed and formulation-independent
--------------------------------------------------
  * It is a minimum of a measured mechanical observable, so it has a single value
    for any trajectory (no t* ambiguity, no derivative sign conventions).
  * G_target is set by the tissue/application, not by scaffold chemistry, so Q
    ranks *any* formulation on the same external axis (a true design target).
  * It is monotone in the right direction: stiffer/faster ECM raises the valley,
    raising Q; faster degradation deepens the valley, lowering Q.

We also report the auxiliary connectivity-vs-rigidity lag, tau_gap = t(rigidity
lost) - t(connectivity lost), which is >=0 by the rigidity-gap theorem and
quantifies how misleading a connectivity-only design criterion would be.
"""
from __future__ import annotations

import numpy as np


def load_path_continuity_Q(t, G_union, G_target):
    """Worst-case load-bearing margin over the remodeling window.

    Parameters
    ----------
    t        : (T,) time array
    G_union  : (T,) measured union shear modulus
    G_target : float, externally-required minimum load-bearing modulus (>0)

    Returns
    -------
    dict with Q (float), t_valley (time of minimum), G_valley (min modulus),
    safe (bool).
    """
    G_union = np.asarray(G_union, float)
    i = int(np.argmin(G_union))
    G_valley = float(G_union[i])
    Q = G_valley / float(G_target)
    return {"Q": Q, "t_valley": float(t[i]), "G_valley": G_valley,
            "safe": Q >= 1.0}


def rigidity_connectivity_lag(t, Pinf, G, pinf_thresh=0.5, g_frac=0.02):
    """tau_gap = t(rigidity lost) - t(connectivity lost) for one channel.

    A positive lag means the network remains geometrically connected after it has
    stopped bearing load — the quantity a connectivity-only design would miss.
    Returns np.nan for an endpoint that is never crossed in the window.
    """
    t = np.asarray(t, float); Pinf = np.asarray(Pinf, float); G = np.asarray(G, float)
    G0 = G[0] if G[0] > 0 else 1.0
    def first_below(y, thr):
        idx = np.flatnonzero(y < thr)
        return t[idx[0]] if idx.size else np.nan
    t_conn = first_below(Pinf, pinf_thresh)
    t_rig = first_below(G / G0, g_frac)
    return {"t_connectivity_lost": t_conn, "t_rigidity_lost": t_rig,
            "tau_gap": t_conn - t_rig if np.isfinite(t_conn) and np.isfinite(t_rig) else np.nan}


def summarize_trajectory(rec, G_target):
    """Convenience: Q + lag for a trajectory dict from dynamics.CoupledNetwork.run."""
    out = load_path_continuity_Q(rec["t"], rec["G_union"], G_target)
    out.update({"lag_scaffold":
                rigidity_connectivity_lag(rec["t"], rec["Pinf_scaffold"], rec["G_scaffold"])})
    return out
