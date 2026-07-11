"""Shared pytest fixtures for the gelrigidity test suite.

tests/conftest.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package root is importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

from gelrigidity.network import HydrogelNetwork, HydrogelParams


# ---------------------------------------------------------------------------
# HydrogelParams fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_hydrogel_params():
    """HydrogelParams for a small fast-running network.

    box_size = 20 µm, rho_x = 0.5 µm⁻³ → expected N = 0.5 × 20³ = 100 nodes.
    covalent_fraction=1.0 so degradation tests use only MMP, not thermal rupture.
    """
    return HydrogelParams(
        box_size=20.0,
        rho_x=0.5,
        r_c=1.0,
        covalent_fraction=1.0,
    )


@pytest.fixture
def large_hydrogel_params():
    """HydrogelParams for a dense, well-connected network clearly above p_c.

    box_size = 15 µm, rho_x = 2.0 µm⁻³ → N ≈ 6750 nodes, r_c=1 µm → z≈8.4, p_c≈0.15.
    With all bonds present (p=1) the giant component is essentially the whole network.
    """
    return HydrogelParams(
        box_size=15.0,
        rho_x=2.0,
        r_c=1.0,
        covalent_fraction=1.0,
    )


# ---------------------------------------------------------------------------
# Pre-built network fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_network(small_hydrogel_params):
    """Pre-built HydrogelNetwork from small_hydrogel_params (seed=42)."""
    return HydrogelNetwork(small_hydrogel_params, seed=42)


@pytest.fixture
def large_network(large_hydrogel_params):
    """Pre-built HydrogelNetwork from large_hydrogel_params (seed=42)."""
    return HydrogelNetwork(large_hydrogel_params, seed=42)


# ---------------------------------------------------------------------------
# Elastic-solver / RGG fixtures (rigidity.py, dynamics.py, handoff.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def rgg_params():
    """Small periodic-RGG parameters for fast solver tests.

    rho_x=1.0, box_size=8.0, r_c=1.5 -> N~500, well above the isostatic
    coordination number at full occupancy (z_full~9, G_full>0).
    """
    return dict(rho_x=1.0, box_size=8.0, r_c=1.5)
