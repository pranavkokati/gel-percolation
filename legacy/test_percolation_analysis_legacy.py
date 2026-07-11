"""Tests for Module 5 — DualPercolationTracker and CriticalExponentFitter.

FILE 4: tests/test_percolation_analysis.py
"""

import numpy as np
import pytest

from src.percolation_analysis import CriticalExponentFitter, DualPercolationTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_handoff_data(n: int = 100):
    """Return (t, p_hyd, p_col) where collagen rises fast and hydrogel falls slowly.

    At the handoff time t*, dp_col/dt >> |dp_hyd/dt|, so Q = dp_col - dp_hyd > 0.
    """
    t = np.linspace(0.0, 100.0, n)
    p_hyd = 0.55 * np.exp(-0.005 * t) + 0.02   # slow fall starting near 0.57
    p_col = 0.80 * (1.0 - np.exp(-0.08 * t))    # fast rise to 0.80
    return t, p_hyd, p_col


def _catastrophic_handoff_data(n: int = 100):
    """Return (t, p_hyd, p_col) where hydrogel recovers while collagen declines.

    At the handoff time t*, dp_hyd/dt > 0 and dp_col/dt < 0, so
    Q = dp_col - dp_hyd < 0.
    """
    t = np.linspace(0.0, 100.0, n)
    p_hyd = 0.10 + 0.70 * (1.0 - np.exp(-0.05 * t))   # hydrogel rises
    p_col = 0.80 * np.exp(-0.02 * t) + 0.20             # collagen falls
    return t, p_hyd, p_col


# ---------------------------------------------------------------------------
# DualPercolationTracker
# ---------------------------------------------------------------------------

def test_dual_tracker_records():
    """After recording 10 points the internal history has exactly 10 entries."""
    tracker = DualPercolationTracker()
    for i in range(10):
        tracker.record(float(i), 0.9 - 0.05 * i, 0.05 * i)

    times, p_hyd, p_col = tracker.get_arrays()
    assert len(times) == 10, f"Expected 10 time entries, got {len(times)}"
    assert len(p_hyd) == 10, f"Expected 10 hydrogel entries, got {len(p_hyd)}"
    assert len(p_col) == 10, f"Expected 10 collagen entries, got {len(p_col)}"


def test_handoff_quality_positive():
    """Q > 0 when collagen rises fast and hydrogel falls slowly."""
    tracker = DualPercolationTracker()
    t, p_hyd, p_col = _smooth_handoff_data()
    for ti, ph, pc in zip(t, p_hyd, p_col):
        tracker.record(ti, ph, pc)

    Q = tracker.compute_handoff_quality()
    assert np.isfinite(Q), f"Q is not finite: {Q}"
    assert Q > 0.0, (
        f"Expected Q > 0 (smooth handoff: collagen rises fast, hydrogel slow), "
        f"got Q={Q:.6f}"
    )


def test_handoff_quality_negative():
    """Q < 0 when hydrogel recovers (rises) while collagen declines."""
    tracker = DualPercolationTracker()
    t, p_hyd, p_col = _catastrophic_handoff_data()
    for ti, ph, pc in zip(t, p_hyd, p_col):
        tracker.record(ti, ph, pc)

    Q = tracker.compute_handoff_quality()
    assert np.isfinite(Q), f"Q is not finite: {Q}"
    assert Q < 0.0, (
        f"Expected Q < 0 (catastrophic: hydrogel rises, collagen falls at t*), "
        f"got Q={Q:.6f}"
    )


def test_exponent_fitter_beta():
    """CriticalExponentFitter fits beta=0.418 within 30% on synthetic P_inf(p) data."""
    p_c = 0.2593
    beta_true = 0.418
    rng = np.random.default_rng(42)

    p_range = np.linspace(p_c + 0.01, 1.0, 60)
    # Synthetic P_inf with small noise
    P_inf = (p_range - p_c) ** beta_true + 0.005 * rng.standard_normal(60)
    P_inf = np.maximum(P_inf, 0.0)

    result = CriticalExponentFitter.fit_percolation_exponent(
        p_range, P_inf, p_c_guess=p_c
    )

    assert "beta_fit" in result, "Result dict must contain 'beta_fit'."
    beta_fit = result["beta_fit"]
    rel_error = abs(beta_fit - beta_true) / beta_true

    assert rel_error < 0.30, (
        f"Fitted beta={beta_fit:.4f} deviates more than 30% from "
        f"true beta={beta_true} (rel error = {rel_error:.2%})"
    )


def test_crossover_type():
    """compute_crossover_type returns 'smooth' when Q > 0, 'catastrophic' when Q < 0."""
    # Smooth scenario
    tracker_smooth = DualPercolationTracker()
    t, p_hyd, p_col = _smooth_handoff_data()
    for ti, ph, pc in zip(t, p_hyd, p_col):
        tracker_smooth.record(ti, ph, pc)

    Q_smooth = tracker_smooth.compute_handoff_quality()
    crossover_smooth = tracker_smooth.compute_crossover_type()
    if Q_smooth > 0.0:
        assert crossover_smooth == "smooth", (
            f"Q={Q_smooth:.6f} > 0 but crossover_type='{crossover_smooth}'; "
            f"expected 'smooth'."
        )

    # Catastrophic scenario
    tracker_cat = DualPercolationTracker()
    t, p_hyd, p_col = _catastrophic_handoff_data()
    for ti, ph, pc in zip(t, p_hyd, p_col):
        tracker_cat.record(ti, ph, pc)

    Q_cat = tracker_cat.compute_handoff_quality()
    crossover_cat = tracker_cat.compute_crossover_type()
    if Q_cat < 0.0:
        assert crossover_cat == "catastrophic", (
            f"Q={Q_cat:.6f} < 0 but crossover_type='{crossover_cat}'; "
            f"expected 'catastrophic'."
        )

    # Verify both Q values are indeed on opposite sides
    assert Q_smooth > 0.0, (
        f"Smooth scenario did not produce Q > 0 (got Q={Q_smooth:.6f}); "
        "check the test fixture."
    )
    assert Q_cat < 0.0, (
        f"Catastrophic scenario did not produce Q < 0 (got Q={Q_cat:.6f}); "
        "check the test fixture."
    )
