"""Tests for the transverse-stiffness (pre-stress / entropic) extension, eps = k_perp/k_par."""
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

from gelrigidity.rigidity import periodic_poisson_rgg, ElasticNetwork


@pytest.fixture(scope="module")
def net():
    pos, box, bonds, rhat, rvec = periodic_poisson_rgg(1.0, 6.0, 1.5, seed=5)
    return ElasticNetwork(pos, box, bonds, rhat, rvec)


def _occ(net, p, seed=0):
    return np.random.default_rng(seed).random(net.M) < p


def test_eps_zero_is_central_force_path(net):
    """eps=0 must be the unchanged central-force solve (bitwise reproducible)."""
    occ = _occ(net, 0.7)
    assert net.shear_modulus(occ) == net.shear_modulus(occ, eps=0.0)


def test_eps_invalid_raises(net):
    with pytest.raises(ValueError):
        net.shear_modulus(_occ(net, 0.7), eps=-0.1)
    with pytest.raises(ValueError):
        net.shear_modulus(_occ(net, 0.7), eps=1.5)


def test_direct_and_lsmr_agree(net):
    occ = _occ(net, 0.5)
    for eps in (1e-3, 0.1, 1.0):
        a = net.shear_modulus(occ, eps=eps, method="direct")
        b = net.shear_modulus(occ, eps=eps, method="lsmr")
        assert a == pytest.approx(b, rel=1e-6)


def test_eps_one_equals_scalar_conductance(net):
    """At eps=1 the three displacement components decouple and G equals the
    effective conductivity of the bond network (independent scalar solver)."""
    for p in (0.2, 0.45, 0.8):
        occ = _occ(net, p, seed=1)
        assert net.shear_modulus(occ, eps=1.0) == pytest.approx(
            net.conductance_modulus(occ), rel=1e-7)


def test_eps_one_full_network_is_affine(net):
    """Full network, all bonds, eps=1: modulus equals the conductance and
    is bounded above by the affine (Voigt) value  (1/V) sum_b k r_g^2."""
    occ = np.ones(net.M, bool)
    G = net.shear_modulus(occ, eps=1.0)
    voigt = float((net.rvec[:, 2] ** 2).sum() / net.V)
    assert 0 < G <= voigt * (1 + 1e-9)


def test_eps_one_onset_at_connectivity(net):
    """Phantom network: G>0 as soon as a spanning cluster exists (p_c), with no
    rigidity gap; central-force network at same p is floppy."""
    p = 0.25  # above p_c ~ 0.114 for this RGG, below p_r ~ 0.44
    occ = _occ(net, p, seed=2)
    assert net.shear_modulus(occ, eps=1.0) > 1e-3
    assert net.shear_modulus(occ, eps=0.0) < 1e-8


def test_below_pc_zero_for_all_eps(net):
    occ = _occ(net, 0.04, seed=3)
    for eps in (0.0, 0.01, 1.0):
        assert net.shear_modulus(occ, eps=eps) < 1e-8


def test_monotone_in_eps(net):
    occ = _occ(net, 0.6, seed=4)
    Gs = [net.shear_modulus(occ, eps=e) for e in (0.0, 1e-3, 1e-2, 1e-1, 1.0)]
    assert all(b > a for a, b in zip(Gs, Gs[1:]))


def test_floppy_window_response_linear_in_eps(net):
    """In the floppy window (p_c<p<p_r) the modulus is eps-controlled: G ~ eps^1
    for small eps (stiffness is supplied only by the transverse term)."""
    occ = _occ(net, 0.25, seed=2)
    g1 = net.shear_modulus(occ, eps=1e-4)
    g2 = net.shear_modulus(occ, eps=1e-3)
    assert 0.5 < np.log10(g2 / g1) < 1.5


def test_modulus_scales_linearly_with_bond_stiffness(net):
    """Doubling every bond stiffness doubles G at any eps (dimensional consistency)."""
    occ = _occ(net, 0.6, seed=2)
    k = np.where(occ, 2.0, 0.0)
    for eps in (0.0, 0.01, 0.3):
        g1 = net.shear_modulus(occ, eps=eps)
        g2 = net.shear_modulus(occ, eps=eps, kvec=k)
        assert g2 == pytest.approx(2 * g1, rel=1e-6)


def test_gap_is_nonincreasing_in_eps(net):
    """The rigidity onset (first p at which G > threshold along a nested removal sequence) can only move
    to lower p as eps increases: G is monotone in eps for every fixed occupancy (energy argument)."""
    rng = np.random.default_rng(3)
    order = rng.permutation(net.M)
    for p in (0.25, 0.35, 0.5):
        occ = np.zeros(net.M, bool); occ[order[: int(p * net.M)]] = True
        gs = [net.shear_modulus(occ, eps=e) for e in (0.0, 1e-3, 1e-2, 0.1, 1.0)]
        assert all(b >= a - 1e-12 for a, b in zip(gs, gs[1:]))


def test_eps_one_below_rigidity_threshold_is_finite_while_eps_zero_vanishes(net):
    """Above connectivity but below rigidity percolation: G(eps=0) = 0 (floppy), G(eps=1) > 0."""
    occ = _occ(net, 0.30, seed=4)
    assert net.shear_modulus(occ, eps=0.0) < 1e-8
    assert net.shear_modulus(occ, eps=1.0) > 1e-4
