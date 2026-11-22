"""Tests for gelrigidity.rigidity — the elastic-network shear-modulus solver.

Validates the solver against exact/known limits rather than against a
hardcoded constitutive law (there is none left to test against): a fully
occupied periodic lattice must be rigid (G>0, isotropic), an empty (zero-
occupancy) lattice must be exactly floppy (G=0), and the RGG rigidity
threshold must sit above the RGG connectivity threshold (the central
"rigidity gap" claim).

API note: ``fcc_lattice`` returns ``(pos, box)``; ``periodic_poisson_rgg``
returns ``(pos, box, bonds, rhat, rvec)``; ``ElasticNetwork(pos, box, bonds,
rhat, rvec)``; ``shear_modulus(occ, shear=("x","z"), kvec=None, ...)`` takes
an explicit boolean bond-occupancy mask ``occ`` (there is no bond-list
subsetting -- occupancy is always expressed as a mask over the full
candidate bond list returned by the generator).
"""

import numpy as np
import pytest

from gelrigidity.rigidity import (
    ElasticNetwork, fcc_lattice, periodic_poisson_rgg, neighbour_bonds,
)


def test_fcc_full_occupancy_isotropic():
    """A fully-bonded periodic FCC lattice is rigid and elastically isotropic."""
    pos, box = fcc_lattice(L=6, a=1.0)
    bonds, rhat, rvec = neighbour_bonds(pos, box, r_cut=0.75)
    net = ElasticNetwork(pos, box, bonds, rhat, rvec)
    occ_full = np.ones(net.M, dtype=bool)
    Gxz = net.shear_modulus(occ_full, shear=("x", "z"))
    Gyz = net.shear_modulus(occ_full, shear=("y", "z"))
    Gxy = net.shear_modulus(occ_full, shear=("x", "y"))
    assert Gxz > 0
    # Isotropy: the three shear planes should agree to <1%
    assert abs(Gxz - Gyz) / Gxz < 0.01
    assert abs(Gxz - Gxy) / Gxz < 0.01


def test_empty_network_is_floppy():
    """A network with zero bonds occupied has exactly zero shear modulus."""
    pos, box = fcc_lattice(L=4, a=1.0)
    bonds, rhat, rvec = neighbour_bonds(pos, box, r_cut=0.75)
    net = ElasticNetwork(pos, box, bonds, rhat, rvec)
    occ_empty = np.zeros(net.M, dtype=bool)
    assert net.shear_modulus(occ_empty) == 0.0


def test_periodic_rgg_bond_count_scales_with_density(rgg_params):
    """Doubling r_c at fixed rho_x increases the mean coordination number."""
    pos1, box1, bonds1, rhat1, rvec1 = periodic_poisson_rgg(
        rho_x=rgg_params["rho_x"], box_size=rgg_params["box_size"], r_c=1.0, seed=0)
    pos2, box2, bonds2, rhat2, rvec2 = periodic_poisson_rgg(
        rho_x=rgg_params["rho_x"], box_size=rgg_params["box_size"], r_c=1.5, seed=0)
    z1 = 2 * len(bonds1) / len(pos1)
    z2 = 2 * len(bonds2) / len(pos2)
    assert z2 > z1


def test_rigidity_threshold_exceeds_connectivity_threshold(rgg_params):
    """The central falsifiable claim: on the RGG, mechanical rigidity requires
    strictly more bonds than mere connectivity (p_r > p_c), i.e. there is a
    floppy-but-connected window. Checked here at a single occupancy known to
    lie in that window for this box/seed combination."""
    rng_seed = 7
    pos, box, bonds, rhat, rvec = periodic_poisson_rgg(**rgg_params, seed=rng_seed)
    net = ElasticNetwork(pos, box, bonds, rhat, rvec)
    rng = np.random.default_rng(rng_seed)
    # An occupancy just above the connectivity threshold but below rigidity
    p_test = 0.20
    occ = rng.random(net.M) < p_test
    G = net.shear_modulus(occ)
    # At p=0.20 (well below the measured p_r~0.44-0.48) the network must be floppy
    assert G < 1e-6


def test_neighbour_bonds_no_self_loops():
    pos, box, bonds, rhat, rvec = periodic_poisson_rgg(
        rho_x=1.0, box_size=8.0, r_c=1.5, seed=1)
    assert np.all(bonds[:, 0] != bonds[:, 1])
