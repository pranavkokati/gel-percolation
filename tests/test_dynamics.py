"""Tests for gelrigidity.dynamics.CoupledNetwork.

Checks the coupled scaffold-degradation / ECM-deposition trajectory on the
RGG topology: monotonic scaffold decay, non-negative ECM growth, and the
qualitative rigidity-gap phenomenon (scaffold shear modulus reaches zero
while the scaffold is still geometrically connected).
"""

import numpy as np
import pytest

from gelrigidity.dynamics import CoupledNetwork


def _short_rgg_run(seed=1, n_steps=40, k_base=0.03, k_dep=0.02):
    net = CoupledNetwork(topology="rgg", rho_x=1.0, box_size=8.0, r_cut=1.5,
                          k_scaffold=1.0, k_ecm=1.0, seed=seed)
    net.seed_scaffold(p0=1.0)
    net.seed_cells(n_cells=10, secretion_radius=2.0)
    return net.run(n_steps=n_steps, dt=1.0, record_every=5,
                   mmp_level=1.0, k_base=k_base, k_dep=k_dep)


def test_scaffold_occupation_decays():
    rec = _short_rgg_run()
    p = np.asarray(rec["p_scaffold"])
    assert p[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(p) <= 1e-9)  # monotonically non-increasing


def test_ecm_occupation_grows():
    rec = _short_rgg_run()
    p_ecm = np.asarray(rec["p_ecm"])
    assert p_ecm[0] == pytest.approx(0.0, abs=1e-6)
    assert p_ecm[-1] > p_ecm[0]


def test_rigidity_gap_scaffold_channel():
    """Scaffold shear modulus must vanish before scaffold connectivity does —
    the rigidity-gap phenomenon at the heart of this project's claims."""
    rec = _short_rgg_run(k_base=0.04, n_steps=60)
    G = np.asarray(rec["G_scaffold"])
    Pinf = np.asarray(rec["Pinf_scaffold"])
    G0 = G[0]
    idx_floppy = np.flatnonzero(G / G0 < 0.02)
    assert idx_floppy.size > 0, "scaffold never loses rigidity in this short run"
    i0 = idx_floppy[0]
    # At the moment rigidity is lost, connectivity should still be substantially intact
    assert Pinf[i0] > 0.5


def test_union_modulus_nonnegative():
    rec = _short_rgg_run()
    assert np.all(np.asarray(rec["G_union"]) >= -1e-9)
