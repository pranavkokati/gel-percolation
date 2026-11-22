"""Shared pytest fixtures for the gelrigidity test suite.

tests/conftest.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package root is importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def rgg_params():
    """Small periodic-RGG parameters for fast solver tests.

    rho_x=1.0, box_size=8.0, r_c=1.5 -> N~500, well above the isostatic
    coordination number at full occupancy (z_full~9, G_full>0).
    """
    return dict(rho_x=1.0, box_size=8.0, r_c=1.5)
