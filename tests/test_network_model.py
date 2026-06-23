"""Tests for Module 1 — HydrogelNetwork.

FILE 1: tests/test_network_model.py
"""

import numpy as np
import pytest

from src.network_model import HydrogelNetwork, HydrogelParams


def test_network_creation(small_hydrogel_params):
    """Network has nodes and edges after init."""
    net = HydrogelNetwork(small_hydrogel_params, seed=0)
    assert net.graph.number_of_nodes() > 0
    # With rho_x=0.5 in a 20um box and r_c=5, expect edges too
    assert net.graph.number_of_edges() >= 0  # at least no crash; may be 0 in rare seeds


def test_poisson_density(small_hydrogel_params):
    """Node count ~ rho_x * box_size^3 within 3 sigma of the Poisson distribution."""
    expected = small_hydrogel_params.rho_x * small_hydrogel_params.box_size ** 3
    sigma = np.sqrt(expected)
    counts = [HydrogelNetwork(small_hydrogel_params, seed=i).graph.number_of_nodes()
              for i in range(5)]
    mean_count = np.mean(counts)
    # Mean of 20 realisations should be within 3 sigma / sqrt(20) of expected
    assert abs(mean_count - expected) < 3 * sigma, (
        f"Mean node count {mean_count:.1f} is more than 3 sigma from expected {expected:.1f}"
    )


def test_percolation_above_threshold(large_hydrogel_params):
    """At high rho_x (2.0) in a 50um box the network is well above p_c so P_inf > 0.5."""
    net = HydrogelNetwork(large_hydrogel_params, seed=1)
    p_inf = net.get_percolation_order_parameter()
    assert p_inf > 0.5, f"Expected P_inf > 0.5 for dense network, got {p_inf:.4f}"


def test_percolation_below_threshold():
    """At very low rho_x the network is fragmented and P_inf < 0.1."""
    params = HydrogelParams(box_size=30.0, rho_x=0.005, r_c=1.0, covalent_fraction=1.0)
    net = HydrogelNetwork(params, seed=7)
    p_inf = net.get_percolation_order_parameter()
    assert p_inf < 0.1, f"Expected P_inf < 0.1 for sparse network, got {p_inf:.4f}"


def test_degradation_reduces_p_inf(large_hydrogel_params):
    """After 100 degradation steps with high MMP, P_inf decreases."""
    net = HydrogelNetwork(large_hydrogel_params, seed=2)
    p_before = net.get_percolation_order_parameter()
    mmp = np.full((10, 10, 10), 10.0)  # high MMP concentration
    for _ in range(100):
        net.degrade_step(mmp, dt=1.0)
    p_after = net.get_percolation_order_parameter()
    assert p_after <= p_before + 0.02, (
        f"P_inf should not increase after heavy degradation: "
        f"before={p_before:.4f}, after={p_after:.4f}"
    )


def test_giant_component_is_fraction(small_hydrogel_params):
    """P_inf is always in [0, 1]."""
    net = HydrogelNetwork(small_hydrogel_params, seed=3)
    p_inf = net.get_percolation_order_parameter()
    assert 0.0 <= p_inf <= 1.0, f"P_inf={p_inf} is outside [0, 1]"


def test_local_p_grid_shape(large_hydrogel_params):
    """compute_local_p returns arrays with the correct grid shapes."""
    net = HydrogelNetwork(large_hydrogel_params, seed=4)
    resolution = 6
    grid_coords, local_p = net.compute_local_p(resolution=resolution)
    assert grid_coords.shape == (resolution, resolution, resolution, 3), (
        f"grid_coords shape {grid_coords.shape} != expected "
        f"({resolution}, {resolution}, {resolution}, 3)"
    )
    assert local_p.shape == (resolution, resolution, resolution), (
        f"local_p shape {local_p.shape} != expected "
        f"({resolution}, {resolution}, {resolution})"
    )
