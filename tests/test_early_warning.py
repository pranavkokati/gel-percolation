"""Tests for Module 3 — EarlyWarningSignalDetector and TopologicalDataAnalyzer.

FILE 3: tests/test_early_warning.py
"""

import numpy as np
import pytest
from scipy import stats

from src.early_warning import EarlyWarningSignalDetector, TopologicalDataAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decaying_g_prime(n: int = 300, seed: int = 42) -> tuple:
    """Simulate G'(t) with critical slowing down: rising AR1 AND rising variance.

    Residuals are an AR1(rho(t)) process where both rho (autocorrelation) and
    sigma_innov (innovation amplitude) increase monotonically with time.
    After Savitzky-Golay detrending removes the smooth trend, the residuals
    show clear positive Kendall tau for both AR1 and rolling variance.
    """
    t = np.linspace(0.0, 300.0, n)
    G_trend = 1000.0 / (1.0 + np.exp(0.04 * (t - 220.0)))
    rng = np.random.default_rng(seed)
    # Both AR1 coefficient and innovation σ increase: classical CSD signature
    rho = 0.05 + 0.80 * (t / 300.0)                   # 0.05 → 0.85
    sigma_innov = 3.0 + 80.0 * (t / 300.0)             # 3 → 83 Pa
    eps = rng.standard_normal(n)
    noise = np.zeros(n)
    noise[0] = eps[0] * sigma_innov[0]
    for i in range(1, n):
        noise[i] = rho[i] * noise[i - 1] + sigma_innov[i] * eps[i]
    G = G_trend + noise
    G = np.maximum(G, 0.0)
    return t, G


def _k4_positions() -> np.ndarray:
    """Corners of a unit square in 3D — all six pairwise distances <= sqrt(2)."""
    return np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [1.0, 1.0, 0.0]],
        dtype=float,
    )


def _tree_positions() -> np.ndarray:
    """Six collinear points — forms a path graph (tree, no loops)."""
    return np.column_stack(
        [np.linspace(0.0, 5.0, 6), np.zeros(6), np.zeros(6)]
    )


# ---------------------------------------------------------------------------
# EarlyWarningSignalDetector tests
# ---------------------------------------------------------------------------

def test_ar1_increases_near_transition():
    """AR1 trend is positive (Kendall tau > 0) for a decaying G'(t)."""
    t, G = _make_decaying_g_prime(n=300, seed=0)
    det = EarlyWarningSignalDetector(window_size=40, detrend=True)
    ar1 = det.compute_ar1(G)

    valid = ~np.isnan(ar1)
    assert valid.sum() >= 4, "Too few non-NaN AR1 values to compute Kendall tau."

    tau, _ = stats.kendalltau(t[valid], ar1[valid])
    assert tau > 0, (
        f"Expected positive Kendall tau for AR1 trend near transition; got tau={tau:.4f}"
    )


def test_variance_increases_near_transition():
    """Variance trend is positive (Kendall tau > 0) for a decaying G'(t)."""
    t, G = _make_decaying_g_prime(n=300, seed=1)
    det = EarlyWarningSignalDetector(window_size=40, detrend=True)
    var = det.compute_variance(G)

    valid = ~np.isnan(var)
    assert valid.sum() >= 4, "Too few non-NaN variance values to compute Kendall tau."

    tau, _ = stats.kendalltau(t[valid], var[valid])
    assert tau > 0, (
        f"Expected positive Kendall tau for variance trend near transition; "
        f"got tau={tau:.4f}"
    )


# ---------------------------------------------------------------------------
# TopologicalDataAnalyzer tests
# ---------------------------------------------------------------------------

def test_h1_statistics_on_simple_graph():
    """H1 count > 0 for a complete graph K4 (4 nodes, 6 edges — contains 3 loops)."""
    tda = TopologicalDataAnalyzer(max_edge_length=5.0, max_dimension=1)
    pos = _k4_positions()
    # K4: all 6 pairs connected
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    dgm = tda.compute_persistence_diagram(pos, edges)
    h1_stats = tda.compute_h1_statistics(dgm, lifetime_threshold=0.01)

    # The Vietoris-Rips filtration on K4 must produce at least one H1 loop
    n_loops = h1_stats["n_long_lived_h1"]
    assert n_loops > 0, (
        f"Expected H1 count > 0 for K4 graph, got {n_loops}.  "
        f"H1 diagram: {dgm['H1']}"
    )


def test_h1_statistics_on_tree():
    """H1 count = 0 for a path graph (tree — no topological loops)."""
    tda = TopologicalDataAnalyzer(max_edge_length=2.5, max_dimension=1)
    pos = _tree_positions()
    # Path edges: 0-1, 1-2, 2-3, 3-4, 4-5  (max distance = 1.0 << 2.5)
    edges = [(i, i + 1) for i in range(5)]

    dgm = tda.compute_persistence_diagram(pos, edges)
    h1_stats = tda.compute_h1_statistics(dgm, lifetime_threshold=0.1)

    n_loops = h1_stats["n_long_lived_h1"]
    assert n_loops == 0, (
        f"Expected H1 count = 0 for a path / tree graph, got {n_loops}.  "
        f"H1 diagram: {dgm['H1']}"
    )


def test_persistence_diagram_shape():
    """compute_persistence_diagram returns a dict with H0 and H1 keys."""
    tda = TopologicalDataAnalyzer(max_edge_length=10.0, max_dimension=1)
    pos = _k4_positions()
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    dgm = tda.compute_persistence_diagram(pos, edges)

    assert isinstance(dgm, dict), "Return value must be a dict."
    assert "H0" in dgm, "Persistence diagram must contain 'H0' key."
    assert "H1" in dgm, "Persistence diagram must contain 'H1' key."

    # Each entry must be a 2-column array (birth, death)
    for key in ("H0", "H1"):
        arr = np.asarray(dgm[key])
        assert arr.ndim == 2, (
            f"dgm['{key}'] must be 2-D (n_features, 2); got shape {arr.shape}"
        )
        assert arr.shape[1] == 2, (
            f"dgm['{key}'] must have 2 columns; got {arr.shape[1]}"
        )
