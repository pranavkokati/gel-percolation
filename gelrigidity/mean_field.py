"""
mean_field.py -- classical (Akalp/Vernerey-style) mean-field reverse-gelation
modulus law, for DIRECT, QUANTITATIVE comparison against the measured,
critically-scaling elastic solver in ``rigidity.py``.

Context
-------
Every existing coupled scaffold-degradation / tissue-growth model we could
find in the literature -- including the one paper that poses exactly this
problem, Akalp, Bryant & Vernerey, Soft Matter 12, 7505 (2016)
(DOI 10.1039/C6SM00583G), and its precursor Dhote & Vernerey, Biomech.
Model. Mechanobiol. 13, 167 (2014) (DOI 10.1007/s10237-013-0493-0) --
represents the mechanical collapse of a degrading gel with a MEAN-FIELD,
step-function "reverse gelation" law:

    classical rubber elasticity :  G(rho) ~ rho          for rho > rho_c
    reverse gelation            :  G(rho) = 0            for rho <= rho_c

i.e. the shear modulus is assumed LINEAR in the surviving crosslink density
(or bond occupancy p, its network analogue) all the way down to a single
critical density rho_c, at which point it is assumed to vanish discontinuously.
Akalp et al. state this explicitly: "when the gel reaches reverse gelation
(rho = rho_c), it loses all of its elasticity" (Eq. 3.10), a hard
discontinuity with NO critical scaling regime.

Crucially, rho_c in that treatment is a CONNECTIVITY-scale quantity (the
classical gel point / Flory-Stockmayer threshold) -- there is no distinct
"rigidity threshold" in a mean-field treatment, because mean-field theory
does not resolve the difference between a network being connected and a
network being able to bear stress. Real disordered elastic networks do NOT
work this way: connectivity percolates at p_c, but the network only becomes
RIGID (spanning stress-bearing backbone) at a higher threshold p_r > p_c
(the "rigidity gap"), with the modulus approaching zero continuously as
G ~ (p - p_r)^f (f measured, not assumed).

This module implements the mean-field law using the SAME occupancy
trajectory p(t) produced by the measured simulation (CoupledNetwork), so
the two predictions can be compared bond-for-bond on identical dynamics --
isolating the effect of the *constitutive assumption* from any difference
in the degradation/deposition kinetics themselves.
"""
from __future__ import annotations

import numpy as np


def affine_meanfield_modulus(p, p_c, G0=1.0):
    """Classical affine/mean-field rubber-elasticity modulus with a hard
    reverse-gelation cutoff at p_c (the CONNECTIVITY threshold), following
    Akalp, Bryant & Vernerey (2016) Eq. (3.9)-(3.10):

        G(p) = G0 * (p - p_c) / (1 - p_c)   for p > p_c
        G(p) = 0                             for p <= p_c

    This is linear in occupancy above the connectivity threshold and exactly
    zero below it -- there is no separate rigidity threshold and no critical
    exponent; the "gap" between connectivity loss and mechanical failure that
    the measured solver resolves is, by construction, absent here.
    """
    p = np.asarray(p, dtype=float)
    frac = np.clip((p - p_c) / (1.0 - p_c), 0.0, None)
    return G0 * frac


def meanfield_union_trajectory(traj, p_c_scaffold, p_c_ecm, G0_scaffold=1.0, G0_ecm=1.0):
    """Given a measured CoupledNetwork.run() trajectory dict (which carries
    the actual bond-occupancy time series p_scaffold(t), p_ecm(t), p_union(t)
    from the real degradation/deposition process), compute what the
    mean-field/step-function constitutive law WOULD have predicted for the
    modulus at every recorded time -- using the identical occupancy
    trajectory, so only the constitutive assumption differs.

    Returns a dict with G_scaffold_mf, G_ecm_mf, G_union_mf (union modelled
    as the additive mean-field prediction of whichever phase is present,
    scaffold-priority as in Akalp et al.'s single-network treatment, plus an
    ECM contribution once ECM itself is mean-field-rigid).
    """
    p_scaf = traj["p_scaffold"]
    p_ecm = traj["p_ecm"]

    G_scaf_mf = affine_meanfield_modulus(p_scaf, p_c_scaffold, G0_scaffold)
    G_ecm_mf = affine_meanfield_modulus(p_ecm, p_c_ecm, G0_ecm)
    # union: mean-field treatments of composite/interpenetrating networks
    # (e.g. Akalp et al. Eq. 3.8) additively decompose the strain energy,
    # which for small-strain shear modulus corresponds to additive moduli.
    G_union_mf = G_scaf_mf + G_ecm_mf

    return {
        "t": traj["t"],
        "G_scaffold_mf": G_scaf_mf,
        "G_ecm_mf": G_ecm_mf,
        "G_union_mf": G_union_mf,
    }


def rigidity_gap_bias(traj, p_c_scaffold, G_target_frac=0.2, G0_scaffold=1.0):
    """Quantify how badly the mean-field law mis-times mechanical failure
    relative to the measured solver, on the SAME trajectory.

    Returns dict with:
      t_fail_measured : first recorded t at which measured G_scaffold <= G_target
      t_fail_meanfield: first recorded t at which mean-field G_scaffold_mf <= G_target
      lead_time       : t_fail_meanfield - t_fail_measured (positive = mean
                         field is optimistic / late to flag failure, i.e. it
                         UNDER-predicts risk during the interval)
    """
    G_target = G_target_frac * G0_scaffold
    t = traj["t"]
    G_meas = traj["G_scaffold"]
    mf = meanfield_union_trajectory(traj, p_c_scaffold, p_c_scaffold, G0_scaffold, 0.0)
    G_mf = mf["G_scaffold_mf"]

    def first_below(G):
        idx = np.where(G <= G_target)[0]
        return float(t[idx[0]]) if len(idx) else float("nan")

    t_meas = first_below(G_meas)
    t_mf = first_below(G_mf)
    return {
        "t_fail_measured": t_meas,
        "t_fail_meanfield": t_mf,
        "lead_time": t_mf - t_meas if not (np.isnan(t_meas) or np.isnan(t_mf)) else float("nan"),
    }
