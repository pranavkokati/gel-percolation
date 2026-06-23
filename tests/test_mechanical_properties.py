"""Tests for Module 2 — PercolationMechanics.

FILE 2: tests/test_mechanical_properties.py
"""

import numpy as np
import pytest

from src.mechanical_properties import MechanicsParams, PercolationMechanics


@pytest.fixture
def mech():
    return PercolationMechanics(MechanicsParams(p_c=0.2593, E_ref=1000.0))


def test_modulus_above_p_c(mech):
    """G' > 0 when p > p_c."""
    G = mech.compute_shear_modulus(p=0.5, omega=1.0)
    assert G > 0.0, f"Expected G' > 0 for p=0.5 > p_c, got {G}"


def test_modulus_below_p_c(mech):
    """G' ≈ 0 when p << p_c."""
    G = mech.compute_shear_modulus(p=0.05, omega=1.0)
    assert G == pytest.approx(0.0), f"Expected G' = 0 for p=0.05 << p_c, got {G}"


def test_modulus_power_law(mech):
    """G' ~ (p - p_c)^f near p_c; fitted exponent within 20% of expected value."""
    p_c = mech.params.p_c
    expected_f = mech.params.exponents.f_elastic

    # Only fit within the pure critical regime (eps < p_crossover = 0.05).
    # The affine term dominates above p_crossover, which would skew the slope.
    eps_values = np.linspace(0.003, 0.025, 20)
    p_values = p_c + eps_values
    G_values = np.array([mech.compute_shear_modulus(p, omega=1.0) for p in p_values])

    valid = G_values > 0
    if valid.sum() < 4:
        pytest.skip("Fewer than 4 valid G' points; cannot fit power law.")

    log_eps = np.log(eps_values[valid])
    log_G = np.log(G_values[valid])
    coeffs = np.polyfit(log_eps, log_G, 1)
    f_fit = float(coeffs[0])

    rel_error = abs(f_fit - expected_f) / expected_f
    assert rel_error < 0.20, (
        f"Power-law exponent fit {f_fit:.3f} deviates more than 20% "
        f"from expected {expected_f:.3f} (rel error = {rel_error:.2%})"
    )


def test_stiffness_gradient_shape(mech):
    """compute_stiffness_gradient returns shape (Nx, Ny, Nz, 3)."""
    Nx, Ny, Nz = 6, 7, 8
    field = np.random.default_rng(0).uniform(100.0, 1000.0, (Nx, Ny, Nz))
    grad = mech.compute_stiffness_gradient(field, dx=1.0)
    assert grad.shape == (Nx, Ny, Nz, 3), (
        f"Expected gradient shape ({Nx}, {Ny}, {Nz}, 3), got {grad.shape}"
    )


def test_gel_point_detection(mech):
    """get_gel_point finds p_c within a reasonable tolerance on synthetic data."""
    true_p_c = mech.params.p_c
    f = mech.params.exponents.f_elastic

    # Synthetic G'(p) = 1000 * (p - p_c)^f for p > p_c
    p_values = np.linspace(true_p_c + 0.02, 0.95, 40)
    moduli = 1000.0 * (p_values - true_p_c) ** f

    detected_p_c = mech.get_gel_point(p_values, moduli)
    assert abs(detected_p_c - true_p_c) < 0.05, (
        f"Detected p_c = {detected_p_c:.4f}, expected near {true_p_c:.4f} "
        f"(tolerance 0.05)"
    )
