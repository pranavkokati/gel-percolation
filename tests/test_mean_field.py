"""Tests for gelrigidity.mean_field — the classical mean-field baseline
constitutive law (Akalp, Bryant & Vernerey, Soft Matter 12, 7505 (2016)),
used as a quantitative comparison against the measured rigidity-percolation
solver on identical occupancy trajectories.
"""

import numpy as np
import pytest

from gelrigidity.mean_field import (
    affine_meanfield_modulus, meanfield_union_trajectory, rigidity_gap_bias,
)


def test_affine_meanfield_modulus_zero_below_pc():
    p = np.array([0.0, 0.05, 0.1, 0.11264436241422231])
    G = affine_meanfield_modulus(p, p_c=0.11264436241422231, G0=1.0)
    assert np.all(G == 0.0)


def test_affine_meanfield_modulus_linear_above_pc():
    p_c = 0.2
    G0 = 3.0
    p = np.array([0.2, 0.4, 0.6, 1.0])
    G = affine_meanfield_modulus(p, p_c=p_c, G0=G0)
    expected = G0 * (p - p_c) / (1 - p_c)
    assert G == pytest.approx(expected)


def test_affine_meanfield_modulus_no_critical_scaling():
    """The defining behavioural difference from the measured solver: the
    mean-field law is exactly linear (no exponent, no rigidity gap) --
    slope is constant across the whole occupied range."""
    p_c = 0.1
    p = np.linspace(p_c, 1.0, 50)
    G = affine_meanfield_modulus(p, p_c=p_c, G0=1.0)
    dG = np.diff(G) / np.diff(p)
    assert np.allclose(dG, dG[0], atol=1e-12)


def test_meanfield_union_trajectory_additive():
    traj = {
        "t": np.array([0.0, 1.0, 2.0]),
        "p_scaffold": np.array([1.0, 0.5, 0.05]),
        "p_ecm": np.array([0.0, 0.3, 0.6]),
    }
    out = meanfield_union_trajectory(traj, p_c_scaffold=0.1, p_c_ecm=0.2,
                                      G0_scaffold=1.0, G0_ecm=2.0)
    assert out["G_union_mf"] == pytest.approx(out["G_scaffold_mf"] + out["G_ecm_mf"])
    # below scaffold p_c at t=2 -> scaffold contribution vanishes
    assert out["G_scaffold_mf"][2] == pytest.approx(0.0)


def test_rigidity_gap_bias_reports_lead_time():
    # Measured solver loses rigidity at t=20 (critical scaling drives G to
    # zero well before mean-field's linear law would predict failure).
    t = np.linspace(0, 100, 101)
    G_meas = np.where(t < 20, 1.0, 0.0)
    p_scaf = np.where(t < 60, 1.0, 0.05)  # connectivity lost at t=60
    traj = {"t": t, "G_scaffold": G_meas, "p_scaffold": p_scaf,
            "p_ecm": np.zeros_like(t)}
    out = rigidity_gap_bias(traj, p_c_scaffold=0.1, G_target_frac=0.2, G0_scaffold=1.0)
    assert out["t_fail_measured"] == pytest.approx(20.0)
    # mean-field stays fully rigid (G0=1.0, linear, target=0.2) until p drops
    # close to p_c -- with p a step function it only reaches G_target once
    # p <= 0.1*1.8+0.1... in this construction p steps straight past p_c, so
    # mean-field predicts failure at the same step as connectivity loss (t=60)
    assert out["t_fail_meanfield"] == pytest.approx(60.0)
    assert out["lead_time"] == pytest.approx(40.0)
