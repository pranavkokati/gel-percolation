"""Module 3 — Early Warning Signal Detection for the Gel–Sol Percolation Transition.

The central prediction of this project: critical slowing down near p_c manifests
as measurable early warning signals (EWS) in bulk rheology G'(t) before the
actual gel-sol transition.  Two classes of EWS are tracked:

Classical EWS (from critical slowing down theory):
  1. Lag-1 autocorrelation (AR1) -> 1 as p -> p_c
  2. Variance of G' fluctuations diverges: chi ~ |p - p_c|^{-gamma}, gamma = 1.8
  3. Spatial correlation length diverges: xi ~ |p - p_c|^{-nu}, nu = 0.88

Topological EWS (novel contribution):
  4. H1 persistent homology loop count peaks BEFORE G' variance — because
     topological loops (mesoscale pores) are destroyed before the bulk
     percolation order parameter shifts.

The predicted ranking of lead times:
    t_H1_peak  <  t_G'_variance_onset  <  t_actual_transition

Critical exponents used throughout:
    gamma = 1.8  (susceptibility / variance divergence)
    nu    = 0.88 (correlation length divergence)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Optional TDA backends
# ---------------------------------------------------------------------------

try:
    from ripser import ripser as _ripser_func
    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False

try:
    import gudhi as _gudhi
    _HAS_GUDHI = True
except ImportError:
    _HAS_GUDHI = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

__all__ = [
    "EarlyWarningSignalDetector",
    "TopologicalDataAnalyzer",
    "SpatialCorrelationAnalyzer",
    "plot_ews_panel",
    "plot_persistence_diagram",
]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_GAMMA = 1.8   # susceptibility / variance critical exponent
_NU = 0.88     # correlation-length critical exponent


# ===========================================================================
# Class 1 — Classical EWS detector
# ===========================================================================

class EarlyWarningSignalDetector:
    """Detect critical slowing down in a rheological time series G'(t).

    Near the gel-sol percolation threshold p_c, the characteristic relaxation
    time diverges.  This is observable as:
      * AR1 rising toward 1 (the system loses its memory-restoring capacity)
      * Variance diverging as chi ~ |p - p_c|^{-gamma}

    Parameters
    ----------
    window_size : int
        Rolling window length (number of data points) for AR1 and variance.
    lag : int
        Autocorrelation lag (typically 1).
    detrend : bool
        Subtract a smooth trend before computing EWS to isolate fluctuations.
    baseline_fraction : float
        Fraction of the (valid) series used to define the quiet-time baseline
        for threshold detection.
    """

    def __init__(
        self,
        window_size: int = 50,
        lag: int = 1,
        detrend: bool = True,
        baseline_fraction: float = 0.2,
    ) -> None:
        if window_size < 4:
            raise ValueError("window_size must be at least 4.")
        if lag < 1:
            raise ValueError("lag must be >= 1.")
        self.window_size = window_size
        self.lag = lag
        self.detrend = detrend
        self.baseline_fraction = baseline_fraction

    # ------------------------------------------------------------------
    # Core rolling statistics
    # ------------------------------------------------------------------

    def compute_ar1(self, time_series: np.ndarray) -> np.ndarray:
        """Rolling lag-1 autocorrelation coefficient.

        Parameters
        ----------
        time_series : np.ndarray, shape (T,)
            Raw or pre-processed G'(t) values.

        Returns
        -------
        np.ndarray, shape (T,)
            AR1 at each time point; the first ``window_size`` entries are NaN
            because a full window is required.
        """
        ts = self._maybe_detrend(time_series)
        n = len(ts)
        ar1 = np.full(n, np.nan)
        w = self.window_size
        lag = self.lag

        for i in range(w, n):
            chunk = ts[i - w: i]
            sigma = np.std(chunk)
            if sigma < 1e-15:
                continue
            x = chunk[: -lag]
            y = chunk[lag:]
            r = np.corrcoef(x, y)[0, 1]
            ar1[i] = float(r)

        return ar1

    def compute_variance(self, time_series: np.ndarray) -> np.ndarray:
        """Rolling variance of fluctuations around the local mean.

        Parameters
        ----------
        time_series : np.ndarray, shape (T,)

        Returns
        -------
        np.ndarray, shape (T,)
            Rolling variance; leading ``window_size`` entries are NaN.
        """
        ts = self._maybe_detrend(time_series)
        n = len(ts)
        var = np.full(n, np.nan)
        w = self.window_size

        for i in range(w, n):
            var[i] = float(np.var(ts[i - w: i], ddof=1))

        return var

    # ------------------------------------------------------------------
    # Composite EWS computation
    # ------------------------------------------------------------------

    def compute_ews_indicators(
        self,
        G_prime_timeseries: np.ndarray,
        times: np.ndarray,
    ) -> Dict:
        """Compute all EWS indicators and their trend statistics.

        Parameters
        ----------
        G_prime_timeseries : np.ndarray, shape (T,)
            Storage modulus time series [Pa].
        times : np.ndarray, shape (T,)
            Corresponding time stamps [s].

        Returns
        -------
        dict with keys:
            ar1              -- np.ndarray, rolling AR1 values
            variance         -- np.ndarray, rolling variance values
            kendall_tau_ar1  -- float, Kendall tau for AR1 trend (positive = rising)
            kendall_tau_var  -- float, Kendall tau for variance trend
            ar1_pvalue       -- float, two-sided p-value for AR1 tau
            var_pvalue       -- float, two-sided p-value for variance tau
            ews_onset_time   -- float or None, time when AR1 first exceeds threshold
            transition_time  -- float, estimated time of gel-sol transition
            lead_time        -- float or None, transition_time - ews_onset_time [s]
        """
        G_prime_timeseries = np.asarray(G_prime_timeseries, dtype=float)
        times = np.asarray(times, dtype=float)

        ar1 = self.compute_ar1(G_prime_timeseries)
        var = self.compute_variance(G_prime_timeseries)

        # Kendall's tau trend test — monotonically increasing EWS is the signal
        valid_ar1 = ~np.isnan(ar1)
        if valid_ar1.sum() >= 4:
            tau_ar1, p_ar1 = stats.kendalltau(times[valid_ar1], ar1[valid_ar1])
        else:
            tau_ar1, p_ar1 = np.nan, np.nan

        valid_var = ~np.isnan(var)
        if valid_var.sum() >= 4:
            tau_var, p_var = stats.kendalltau(times[valid_var], var[valid_var])
        else:
            tau_var, p_var = np.nan, np.nan

        # Estimate gel-sol transition: first time G' drops below 1 % of its
        # early-time maximum (proxy for p crossing p_c).
        n_early = max(10, self.window_size)
        g_peak = np.nanmax(G_prime_timeseries[:n_early])
        threshold_gp = 0.01 * g_peak
        below = G_prime_timeseries < threshold_gp
        if np.any(below):
            transition_idx = int(np.argmax(below))
            transition_time = float(times[transition_idx])
        else:
            transition_idx = len(times) - 1
            transition_time = float(times[-1])

        # EWS onset index and time
        ews_idx = self.detect_transition_time(ar1, threshold_sigma=2.0)
        ews_onset_time = float(times[ews_idx]) if ews_idx is not None else None

        lead_time = (
            float(transition_time - ews_onset_time)
            if ews_onset_time is not None
            else None
        )

        return {
            "ar1": ar1,
            "variance": var,
            "kendall_tau_ar1": float(tau_ar1) if np.isfinite(tau_ar1) else None,
            "kendall_tau_var": float(tau_var) if np.isfinite(tau_var) else None,
            "ar1_pvalue": float(p_ar1) if np.isfinite(p_ar1) else None,
            "var_pvalue": float(p_var) if np.isfinite(p_var) else None,
            "ews_onset_time": ews_onset_time,
            "transition_time": transition_time,
            "lead_time": lead_time,
        }

    def detect_transition_time(
        self,
        indicator: np.ndarray,
        threshold_sigma: float = 2.0,
    ) -> Optional[int]:
        """Return the index where an indicator first exceeds its baseline level.

        The baseline is defined as the first ``baseline_fraction`` of the
        non-NaN values.  The alarm threshold is
        ``baseline_mean + threshold_sigma * baseline_std``.

        Parameters
        ----------
        indicator : np.ndarray, shape (T,)
            AR1, variance, or any scalar EWS indicator.
        threshold_sigma : float
            Number of standard deviations above the baseline mean.

        Returns
        -------
        int or None
            Index (into the original ``indicator`` array) of the first
            exceedance, or None if the threshold is never crossed.
        """
        indicator = np.asarray(indicator, dtype=float)
        valid_mask = ~np.isnan(indicator)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 4:
            return None

        n_baseline = max(int(self.baseline_fraction * len(valid_indices)), 2)
        baseline_vals = indicator[valid_indices[:n_baseline]]
        bl_mean = float(np.mean(baseline_vals))
        bl_std = float(np.std(baseline_vals, ddof=1)) if n_baseline > 1 else 0.0
        threshold = bl_mean + threshold_sigma * bl_std

        for idx in valid_indices[n_baseline:]:
            if indicator[idx] > threshold:
                return int(idx)

        return None

    def compute_lead_time(
        self,
        G_prime_series: np.ndarray,
        times: np.ndarray,
        transition_time_idx: int,
    ) -> float:
        """Lead time between the EWS onset (AR1 threshold crossing) and the
        actual gel-sol transition.

        Parameters
        ----------
        G_prime_series : np.ndarray, shape (T,)
        times : np.ndarray, shape (T,)
        transition_time_idx : int
            Index in ``times`` corresponding to the actual gel-sol transition.

        Returns
        -------
        float
            Lead time in the same units as ``times``; 0.0 if no EWS detected.
        """
        ar1 = self.compute_ar1(np.asarray(G_prime_series, dtype=float))
        onset_idx = self.detect_transition_time(ar1, threshold_sigma=2.0)
        if onset_idx is None:
            return 0.0
        return float(times[transition_time_idx] - times[onset_idx])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_detrend(self, ts: np.ndarray) -> np.ndarray:
        """Subtract a Savitzky-Golay smooth trend to isolate fluctuations."""
        ts = np.asarray(ts, dtype=float)
        if not self.detrend or len(ts) < self.window_size + 2:
            return ts.copy()
        # Window length must be odd and >= 5
        wl = self.window_size
        if wl % 2 == 0:
            wl += 1
        wl = min(wl, len(ts) - (0 if len(ts) % 2 == 1 else 1))
        if wl < 5:
            return ts.copy()
        try:
            trend = savgol_filter(ts, window_length=wl, polyorder=2)
            return ts - trend
        except Exception:
            return ts.copy()


# ===========================================================================
# Class 2 — Topological Data Analysis
# ===========================================================================

class TopologicalDataAnalyzer:
    """Compute persistent homology of network snapshots.

    The key novel prediction: H1 (1-cycle / loop) count peaks BEFORE G'
    variance, providing an earlier topological warning of the impending
    gel-sol transition.  Mesoscale pores (captured by H1 loops) are destroyed
    progressively as fibres are cleaved, but this topological restructuring
    precedes the bulk rheological response.

    Parameters
    ----------
    max_edge_length : float
        Maximum filtration distance for Vietoris-Rips complex [same units as
        node positions, typically µm].
    max_dimension : int
        Maximum homology dimension.  1 yields H0 (connected components) and
        H1 (loops).
    """

    def __init__(
        self,
        max_edge_length: float = 10.0,
        max_dimension: int = 1,
    ) -> None:
        self.max_edge_length = float(max_edge_length)
        self.max_dimension = int(max_dimension)

        if not _HAS_RIPSER and not _HAS_GUDHI:
            warnings.warn(
                "Neither ripser nor gudhi is installed.  "
                "TDA computations will return empty persistence diagrams.  "
                "Install with:  pip install ripser   OR   pip install gudhi",
                ImportWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Persistence diagram computation
    # ------------------------------------------------------------------

    def compute_persistence_diagram(
        self,
        node_positions: np.ndarray,
        adjacency: List[Tuple[int, int]],
    ) -> Dict:
        """Compute H0 and H1 persistence diagrams via Vietoris-Rips filtration.

        The filtration is built from pairwise Euclidean distances of node
        positions.  The ``adjacency`` list is provided for context (e.g. to
        restrict to a subgraph) but the VR complex uses all pairwise distances
        within ``max_edge_length``.

        Parameters
        ----------
        node_positions : np.ndarray, shape (N, 3)
            3D coordinates of network nodes [µm].
        adjacency : list of (int, int) tuples
            Current active edges in the network snapshot (used for subgraph
            sampling when the full pointcloud is too large).

        Returns
        -------
        dict
            ``'H0'``: np.ndarray of shape (n0, 2) — (birth, death) pairs for
                      connected-component features (infinite deaths removed).
            ``'H1'``: np.ndarray of shape (n1, 2) — (birth, death) pairs for
                      loop features.
        """
        node_positions = np.asarray(node_positions, dtype=float)

        if node_positions.ndim == 1:
            node_positions = node_positions[:, np.newaxis]

        if len(node_positions) < 2:
            return {"H0": np.empty((0, 2)), "H1": np.empty((0, 2))}

        # Sub-sample for computational tractability (O(N^2) distance matrix)
        max_nodes = 500
        if len(node_positions) > max_nodes:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(node_positions), max_nodes, replace=False)
            node_positions = node_positions[idx]

        if _HAS_RIPSER:
            return self._compute_with_ripser(node_positions)
        elif _HAS_GUDHI:
            return self._compute_with_gudhi(node_positions)
        else:
            return {"H0": np.empty((0, 2)), "H1": np.empty((0, 2))}

    def _compute_with_ripser(self, points: np.ndarray) -> Dict:
        """Backend using the ripser package."""
        result = _ripser_func(
            points,
            maxdim=self.max_dimension,
            thresh=self.max_edge_length,
            metric="euclidean",
        )
        dgms = result["dgms"]

        H0 = np.array(dgms[0], dtype=float) if len(dgms) > 0 else np.empty((0, 2))
        H1 = np.array(dgms[1], dtype=float) if len(dgms) > 1 else np.empty((0, 2))

        # Remove H0 features with infinite death (the single persistent component)
        if len(H0) > 0:
            H0 = H0[np.isfinite(H0[:, 1])]

        return {"H0": H0, "H1": H1}

    def _compute_with_gudhi(self, points: np.ndarray) -> Dict:
        """Backend using the gudhi package (fallback)."""
        rips = _gudhi.RipsComplex(
            points=points.tolist(),
            max_edge_length=self.max_edge_length,
        )
        st = rips.create_simplex_tree(max_dimension=self.max_dimension + 1)
        st.compute_persistence()

        H0_list: List[List[float]] = []
        H1_list: List[List[float]] = []

        for dim, (birth, death) in st.persistence():
            if dim == 0 and np.isfinite(death):
                H0_list.append([float(birth), float(death)])
            elif dim == 1:
                H1_list.append([float(birth), float(death)])

        return {
            "H0": np.array(H0_list, dtype=float) if H0_list else np.empty((0, 2)),
            "H1": np.array(H1_list, dtype=float) if H1_list else np.empty((0, 2)),
        }

    # ------------------------------------------------------------------
    # H1 statistics
    # ------------------------------------------------------------------

    def compute_h1_statistics(
        self,
        persistence_diagram: Dict,
        lifetime_threshold: float = 0.5,
    ) -> Dict:
        """Compute summary statistics of the H1 persistence diagram.

        Parameters
        ----------
        persistence_diagram : dict
            Output of ``compute_persistence_diagram``.
        lifetime_threshold : float
            Minimum feature lifetime to be counted as a 'long-lived' loop.
            Short-lived features are typically topological noise.

        Returns
        -------
        dict with keys:
            n_long_lived_h1  -- int,   count of H1 loops with lifetime > threshold
            mean_h1_lifetime -- float, mean lifetime of all finite H1 features
            max_h1_lifetime  -- float, maximum H1 feature lifetime
            h1_entropy       -- float, Shannon entropy of the lifetime distribution
        """
        H1 = persistence_diagram.get("H1", np.empty((0, 2)))
        H1 = np.asarray(H1, dtype=float)

        # Retain only finite-death features
        if len(H1) > 0:
            finite_mask = np.isfinite(H1[:, 1])
            H1_finite = H1[finite_mask]
        else:
            H1_finite = np.empty((0, 2))

        if len(H1_finite) == 0:
            return {
                "n_long_lived_h1": 0,
                "mean_h1_lifetime": 0.0,
                "max_h1_lifetime": 0.0,
                "h1_entropy": 0.0,
            }

        lifetimes = H1_finite[:, 1] - H1_finite[:, 0]
        lifetimes = lifetimes[lifetimes >= 0.0]  # guard against numerical noise

        n_long = int(np.sum(lifetimes > lifetime_threshold))
        mean_lt = float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0
        max_lt = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0

        # Shannon entropy — measure of topological complexity
        if len(lifetimes) >= 2:
            n_bins = max(5, min(50, len(lifetimes) // 3))
            hist, _ = np.histogram(lifetimes, bins=n_bins)
            total = hist.sum()
            if total > 0:
                probs = hist / total
                probs = probs[probs > 0]
                entropy = float(-np.sum(probs * np.log(probs)))
            else:
                entropy = 0.0
        else:
            entropy = 0.0

        return {
            "n_long_lived_h1": n_long,
            "mean_h1_lifetime": mean_lt,
            "max_h1_lifetime": max_lt,
            "h1_entropy": entropy,
        }

    # ------------------------------------------------------------------
    # Time-series processing
    # ------------------------------------------------------------------

    def compute_topology_timeseries(
        self,
        network_snapshots: List[Tuple[np.ndarray, List]],
    ) -> Dict:
        """Process a list of network snapshots and return H1 statistics over time.

        Parameters
        ----------
        network_snapshots : list of (node_positions, adjacency_list) tuples
            Each element represents the network state at one time point.
            ``node_positions`` is an (N, 3) array; ``adjacency_list`` is a list
            of (i, j) edge tuples.

        Returns
        -------
        dict with 1-D np.ndarray values (length = len(network_snapshots)):
            n_long_lived_h1, mean_h1_lifetime, max_h1_lifetime, h1_entropy
        """
        n = len(network_snapshots)
        n_long = np.zeros(n, dtype=float)
        mean_lt = np.zeros(n, dtype=float)
        max_lt = np.zeros(n, dtype=float)
        entropy = np.zeros(n, dtype=float)

        for i, (pos, edges) in enumerate(network_snapshots):
            try:
                dgm = self.compute_persistence_diagram(
                    np.asarray(pos, dtype=float), list(edges)
                )
                s = self.compute_h1_statistics(dgm)
                n_long[i] = s["n_long_lived_h1"]
                mean_lt[i] = s["mean_h1_lifetime"]
                max_lt[i] = s["max_h1_lifetime"]
                entropy[i] = s["h1_entropy"]
            except Exception as exc:
                warnings.warn(
                    f"Snapshot {i}: TDA failed with {exc!r}.  "
                    "Filling with zeros.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return {
            "n_long_lived_h1": n_long,
            "mean_h1_lifetime": mean_lt,
            "max_h1_lifetime": max_lt,
            "h1_entropy": entropy,
        }

    def detect_h1_peak_time(
        self,
        h1_timeseries: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """Return the time of the H1 loop count peak.

        The H1 loop count peaks before G' variance begins to diverge.  This
        is the topological early-warning lead advantage: mesoscale ring
        structures (H1 loops) are severed by enzymatic cleavage before the
        bulk percolation order parameter shifts.

        Parameters
        ----------
        h1_timeseries : np.ndarray, shape (T,)
            Time series of H1 loop counts (or n_long_lived_h1).
        times : np.ndarray, shape (T,)
            Corresponding time stamps [s].

        Returns
        -------
        float
            Time of the H1 peak [same units as times].
        """
        h1_timeseries = np.asarray(h1_timeseries, dtype=float)
        times = np.asarray(times, dtype=float)

        if len(h1_timeseries) == 0:
            return float(times[-1]) if len(times) > 0 else 0.0

        peak_idx = int(np.argmax(h1_timeseries))
        return float(times[peak_idx])

    def compare_ews_lead_times(
        self,
        G_prime_series: np.ndarray,
        h1_series: np.ndarray,
        times: np.ndarray,
        window_size: int = 30,
    ) -> Dict:
        """Compare topological vs classical EWS lead times.

        Quantifies the topological advantage: by how many seconds does the H1
        peak precede the classical AR1/variance EWS onset?

        Parameters
        ----------
        G_prime_series : np.ndarray, shape (T,)
            Storage modulus time series [Pa].
        h1_series : np.ndarray, shape (T,)
            H1 loop count time series.
        times : np.ndarray, shape (T,)
            Time stamps [s].
        window_size : int
            Rolling window size for the classical EWS detector.

        Returns
        -------
        dict with keys:
            g_prime_ews_onset   -- float or None, time of AR1 threshold crossing
            h1_peak_time        -- float, time of H1 loop count peak
            topology_lead_time  -- float or None, transition_time - h1_peak_time [s]
            transition_time     -- float, estimated gel-sol transition time [s]
            g_prime_lead_time   -- float or None, transition_time - g_prime_ews_onset [s]
            topology_advantage  -- float or None,
                                   g_prime_ews_onset - h1_peak_time (positive means
                                   H1 peaks earlier than classical EWS onset)
        """
        detector = EarlyWarningSignalDetector(window_size=window_size)
        ews_info = detector.compute_ews_indicators(
            np.asarray(G_prime_series, dtype=float),
            np.asarray(times, dtype=float),
        )

        h1_peak_time = self.detect_h1_peak_time(h1_series, times)
        g_prime_onset = ews_info["ews_onset_time"]
        transition_time = ews_info["transition_time"]

        topology_advantage: Optional[float] = None
        if g_prime_onset is not None:
            topology_advantage = float(g_prime_onset - h1_peak_time)

        topology_lead_time: Optional[float] = None
        if h1_peak_time is not None:
            topology_lead_time = float(transition_time - h1_peak_time)

        return {
            "g_prime_ews_onset": g_prime_onset,
            "h1_peak_time": h1_peak_time,
            "topology_lead_time": topology_lead_time,
            "transition_time": transition_time,
            "g_prime_lead_time": ews_info.get("lead_time"),
            "topology_advantage": topology_advantage,
        }


# ===========================================================================
# Class 3 — Spatial correlation analyser
# ===========================================================================

class SpatialCorrelationAnalyzer:
    """Measure spatial correlation lengths and structure factors from the
    stiffness field — direct evidence of the diverging correlation length
    xi ~ |p - p_c|^{-nu} (nu = 0.88).
    """

    # ------------------------------------------------------------------

    @staticmethod
    def compute_correlation_length(
        stiffness_field: np.ndarray,
        dx: float,
    ) -> float:
        """Extract the spatial correlation length xi from the stiffness field.

        The spatial autocorrelation function C(r) is computed via FFT and then
        fit to the exponential decay model C(r) ~ exp(-r / xi).

        Parameters
        ----------
        stiffness_field : np.ndarray, shape (Nx, Ny) or (Nx, Ny, Nz)
            Local stiffness (e.g. G' mapped on a grid) [Pa].  NaN values are
            replaced by the field mean before the FFT.
        dx : float
            Voxel / pixel size [µm].

        Returns
        -------
        float
            Correlation length xi [µm].  Returns 1.0 on fitting failure.
        """
        stiffness_field = np.asarray(stiffness_field, dtype=float)

        if stiffness_field.ndim == 2:
            # Extend to 3D trivially for uniform treatment
            stiffness_field = stiffness_field[:, :, np.newaxis]

        field = stiffness_field - np.nanmean(stiffness_field)
        field = np.nan_to_num(field, nan=0.0)

        # 3-D autocorrelation via the Wiener-Khinchin theorem
        f_fft = np.fft.fftn(field)
        power = np.abs(f_fft) ** 2
        acf_3d = np.real(np.fft.ifftn(power))
        acf_3d = np.fft.fftshift(acf_3d)

        # Normalise
        center_val = float(acf_3d[tuple(np.array(acf_3d.shape) // 2)])
        if abs(center_val) < 1e-20:
            return 1.0
        acf_3d /= center_val

        # Build radial distance array
        shape = np.array(acf_3d.shape)
        center = shape // 2
        axes = [np.arange(s) - c for s, c in zip(shape, center)]
        grids = np.meshgrid(*axes, indexing="ij")
        r = np.sqrt(sum(g**2 for g in grids)) * dx

        r_flat = r.ravel()
        acf_flat = acf_3d.ravel()

        # Radial binning
        r_max = np.max(r_flat) / 2.0
        r_bins = np.linspace(dx, r_max, 40)
        r_centers: List[float] = []
        acf_binned: List[float] = []

        for i in range(len(r_bins) - 1):
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i + 1])
            if mask.sum() > 0:
                r_centers.append(float(0.5 * (r_bins[i] + r_bins[i + 1])))
                acf_binned.append(float(np.mean(acf_flat[mask])))

        r_arr = np.array(r_centers)
        c_arr = np.array(acf_binned)
        valid = c_arr > 0.05  # fit only where ACF is still positive

        if valid.sum() < 3:
            return 1.0

        def _exp_decay(r: np.ndarray, xi: float) -> np.ndarray:
            return np.exp(-r / xi)

        try:
            popt, _ = curve_fit(
                _exp_decay,
                r_arr[valid],
                c_arr[valid],
                p0=[float(r_arr[valid].mean())],
                bounds=(1e-3, 1e4),
                maxfev=5000,
            )
            return float(popt[0])
        except Exception:
            return 1.0

    @staticmethod
    def compute_structure_factor(
        positions: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the static structure factor S(q).

        S(q) = (1/N) |sum_j v_j exp(i q . r_j)|^2

        averaged over q-vectors at each magnitude |q|.  A peak at small q
        signals long-range spatial correlations.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
            Node positions [µm].
        values : np.ndarray, shape (N,)
            Scalar field values at each node (e.g. local stiffness) [Pa].

        Returns
        -------
        (q_values, S_q) : (np.ndarray, np.ndarray), both 1-D.
            q in units of [1/µm], S_q dimensionless.
        """
        positions = np.asarray(positions, dtype=float)
        values = np.asarray(values, dtype=float)
        n = len(positions)

        if n == 0 or positions.ndim != 2 or positions.shape[1] < 2:
            return np.array([]), np.array([])

        # Estimate a sensible q range from the spatial extent
        extent = np.ptp(positions, axis=0)
        extent = np.where(extent > 0, extent, 1.0)
        q_min = 2.0 * np.pi / np.max(extent)
        q_max = 2.0 * np.pi / (np.min(extent) / max(n, 1) * 5.0)
        q_max = max(q_max, q_min * 10)

        q_vals = np.linspace(q_min, q_max, 50)
        S_q = np.zeros(len(q_vals), dtype=float)
        v_norm = values - np.mean(values)

        for iq, q_mag in enumerate(q_vals):
            sq_accum = 0.0
            # Average over three Cartesian q-directions (isotropic estimate)
            ndim = positions.shape[1]
            for axis in range(min(ndim, 3)):
                q_vec = np.zeros(ndim)
                q_vec[axis] = q_mag
                phase = np.exp(1j * (positions @ q_vec))
                sq_accum += float(np.abs(np.dot(v_norm, phase)) ** 2) / n
            S_q[iq] = sq_accum / min(ndim, 3)

        return q_vals, S_q


# ===========================================================================
# Plotting utilities
# ===========================================================================

def plot_ews_panel(
    times: np.ndarray,
    G_prime: np.ndarray,
    ar1: np.ndarray,
    variance: np.ndarray,
    h1_counts: np.ndarray,
    transition_time: float,
    figsize: Tuple = (14, 10),
):
    """Four-panel summary plot of EWS indicators.

    Panels (top to bottom):
      1. G'(t) — storage modulus
      2. Rolling AR1 — approaches 1 as p -> p_c
      3. Rolling variance — diverges near p_c
      4. H1 loop count — peaks before G' variance

    Parameters
    ----------
    times : np.ndarray, shape (T,)
    G_prime : np.ndarray, shape (T,)
    ar1 : np.ndarray, shape (T,)
    variance : np.ndarray, shape (T,)
    h1_counts : np.ndarray, shape (T,)
    transition_time : float
        Time of the gel-sol transition [s] — drawn as a dashed red line.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure or None (if matplotlib is unavailable).
    """
    if not _HAS_MPL:
        warnings.warn(
            "matplotlib is not installed; cannot produce EWS panel.",
            ImportWarning,
            stacklevel=2,
        )
        return None

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Panel 1 — G'(t)
    axes[0].plot(times, G_prime, color="black", lw=1.5, label="G'(t)")
    axes[0].axvline(
        transition_time, color="red", ls="--", lw=1.2,
        label=f"Transition t* = {transition_time:.0f} s",
    )
    axes[0].set_ylabel("G' [Pa]", fontsize=11)
    axes[0].set_title(
        "Early Warning Signals for Gel–Sol Percolation Transition",
        fontsize=12, fontweight="bold",
    )
    axes[0].legend(fontsize=9)

    # Panel 2 — AR1
    axes[1].plot(times, ar1, color="steelblue", lw=1.5, label="AR1")
    axes[1].axvline(transition_time, color="red", ls="--", lw=1.2)
    axes[1].axhline(0.0, color="gray", ls=":", lw=0.8)
    axes[1].axhline(1.0, color="gray", ls=":", lw=0.8)
    axes[1].set_ylabel("AR1", fontsize=11)
    axes[1].set_ylim(-0.1, 1.15)
    axes[1].legend(fontsize=9)

    # Panel 3 — Variance
    axes[2].plot(times, variance, color="forestgreen", lw=1.5, label="Var(G')")
    axes[2].axvline(transition_time, color="red", ls="--", lw=1.2)
    axes[2].set_ylabel("Var(G') [Pa²]", fontsize=11)
    axes[2].legend(fontsize=9)

    # Panel 4 — H1 topology
    axes[3].plot(times, h1_counts, color="darkorchid", lw=1.5, label="H₁ loops")
    axes[3].axvline(transition_time, color="red", ls="--", lw=1.2)
    if len(h1_counts) > 0:
        h1_peak_idx = int(np.argmax(h1_counts))
        axes[3].axvline(
            times[h1_peak_idx], color="darkorchid", ls=":", lw=1.0,
            label=f"H₁ peak t = {times[h1_peak_idx]:.0f} s",
        )
    axes[3].set_ylabel("H₁ count", fontsize=11)
    axes[3].set_xlabel("Time [s]", fontsize=11)
    axes[3].legend(fontsize=9)

    for ax in axes:
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=9)

    fig.tight_layout(h_pad=0.5)
    return fig


def plot_persistence_diagram(
    H0: np.ndarray,
    H1: np.ndarray,
    title: str = "",
    ax=None,
):
    """Plot H0 and H1 persistence diagrams on a birth-death plane.

    Points above the diagonal correspond to genuine topological features;
    H0 = connected components (blue circles), H1 = loops (red triangles).

    Parameters
    ----------
    H0 : np.ndarray, shape (n0, 2)
    H1 : np.ndarray, shape (n1, 2)
    title : str
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _HAS_MPL:
        warnings.warn(
            "matplotlib is not installed; cannot produce persistence diagram.",
            ImportWarning,
            stacklevel=2,
        )
        return None

    H0 = np.asarray(H0, dtype=float) if len(H0) > 0 else np.empty((0, 2))
    H1 = np.asarray(H1, dtype=float) if len(H1) > 0 else np.empty((0, 2))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Collect all finite values to set axis limits
    all_finite: List[float] = []
    for arr in [H0, H1]:
        if len(arr) > 0:
            all_finite.extend(arr[np.isfinite(arr)].tolist())

    if not all_finite:
        ax.set_title(title or "Persistence Diagram (empty)")
        return fig

    lo = min(all_finite) - 0.05 * abs(min(all_finite))
    hi = max(all_finite) + 0.05 * abs(max(all_finite))

    # Diagonal line (birth = death)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)

    # H0 features
    if len(H0) > 0:
        finite_H0 = H0[np.isfinite(H0[:, 1])]
        if len(finite_H0) > 0:
            ax.scatter(
                finite_H0[:, 0], finite_H0[:, 1],
                c="steelblue", s=35, alpha=0.75, zorder=3,
                label=f"H₀ ({len(finite_H0)} components)",
            )

    # H1 features
    if len(H1) > 0:
        finite_H1 = H1[np.isfinite(H1[:, 1])]
        if len(finite_H1) > 0:
            ax.scatter(
                finite_H1[:, 0], finite_H1[:, 1],
                c="firebrick", s=35, marker="^", alpha=0.75, zorder=3,
                label=f"H₁ ({len(finite_H1)} loops)",
            )

    ax.set_xlabel("Birth", fontsize=11)
    ax.set_ylabel("Death", fontsize=11)
    ax.set_title(title or "Persistence Diagram", fontsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect("equal")

    return fig
