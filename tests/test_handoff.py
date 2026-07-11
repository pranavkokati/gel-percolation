"""Tests for gelrigidity.handoff — the unified Q design metric.

Q = min_t G_union(t) / G_target is the ONLY definition of the handoff
quality metric in this codebase (the original repository's two conflicting
definitions have been removed; see legacy/percolation_analysis.py for the
deprecated code retained purely for audit purposes).
"""

import numpy as np
import pytest

from gelrigidity.handoff import (
    load_path_continuity_Q, rigidity_connectivity_lag, summarize_trajectory,
)


def test_Q_safe_when_union_never_drops_below_target():
    t = np.linspace(0, 100, 11)
    G_union = np.full_like(t, 2.0)
    out = load_path_continuity_Q(t, G_union, G_target=1.0)
    assert out["Q"] == pytest.approx(2.0)
    assert out["safe"] is True


def test_Q_unsafe_when_union_dips_below_target():
    t = np.linspace(0, 100, 11)
    G_union = np.array([2.0, 1.5, 0.3, 0.2, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0])
    out = load_path_continuity_Q(t, G_union, G_target=1.0)
    assert out["Q"] == pytest.approx(0.2)
    assert out["safe"] is False
    assert out["t_valley"] == pytest.approx(30.0)


def test_rigidity_connectivity_lag_positive_when_rigidity_lost_first():
    t = np.linspace(0, 100, 101)
    # Connectivity (Pinf) stays high until t=60; rigidity (G) collapses at t=20
    Pinf = np.where(t < 60, 1.0, 0.3)
    G = np.where(t < 20, 1.0, 0.0)
    out = rigidity_connectivity_lag(t, Pinf, G)
    assert out["t_rigidity_lost"] == pytest.approx(20.0)
    assert out["t_connectivity_lost"] == pytest.approx(60.0)
    assert out["tau_gap"] == pytest.approx(40.0)


def test_rigidity_connectivity_lag_nan_when_never_crossed():
    t = np.linspace(0, 100, 101)
    Pinf = np.ones_like(t)  # never drops below threshold
    G = np.ones_like(t)
    out = rigidity_connectivity_lag(t, Pinf, G)
    assert np.isnan(out["tau_gap"])


def test_summarize_trajectory_matches_direct_calls():
    rec = {
        "t": np.array([0.0, 1.0, 2.0]),
        "G_union": np.array([1.0, 0.4, 1.0]),
        "Pinf_scaffold": np.array([1.0, 1.0, 0.4]),
        "G_scaffold": np.array([1.0, 0.01, 0.0]),
    }
    out = summarize_trajectory(rec, G_target=0.5)
    assert out["Q"] == pytest.approx(0.8)
    assert out["safe"] is False
    assert "lag_scaffold" in out
