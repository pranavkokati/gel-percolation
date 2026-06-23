"""Shared pytest fixtures for gel-percolation tests.

FILE 5: tests/conftest.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cell_invasion import CellParams, SimParams
from src.mechanical_properties import MechanicsParams, PercolationMechanics
from src.network_model import HydrogelNetwork, HydrogelParams


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
# Mechanics fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mechanics():
    """PercolationMechanics with default alginate parameters."""
    return PercolationMechanics(MechanicsParams())


# ---------------------------------------------------------------------------
# Cell / simulation parameter fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cell_params():
    """CellParams with all defaults."""
    return CellParams()


@pytest.fixture
def minimal_sim_params():
    """SimParams configured for a fast 50-step simulation.

    n_steps=50, record_interval=5 → 10 recorded snapshots.
    """
    return SimParams(
        n_steps=50,
        record_interval=5,
        n_cells=5,
        grid_resolution=10,
        box_size=20.0,
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# Miscellaneous convenience fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_mmp_field():
    """10×10×10 uniform MMP concentration field at 0.1 (arbitrary units)."""
    return np.full((10, 10, 10), 0.1)
