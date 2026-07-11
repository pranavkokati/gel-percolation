"""
================================================================================
 DEPRECATED -- retained for archival/audit purposes only. NOT part of the
 public API and NOT imported by any current figure, test, or the manuscript.

 This module was flagged during the pre-publication audit (see REPORT.md,
 Section 2) for one or more of: (a) closed-form/hardcoded critical exponents
 presented as if independently measured, (b) a hand-drawn synthetic collagen
 deposition curve, (c) an Ornstein-Uhlenbeck noise term engineered to
 reproduce a target Kendall tau rather than derived from network dynamics,
 or (d) an internally-inconsistent Q metric definition superseded by
 gelrigidity.handoff.load_path_continuity_Q.

 The replacement, physically-grounded implementation lives in gelrigidity/:
   - measured (not closed-form) exponents      -> gelrigidity.rigidity
   - network-driven (not synthetic) ECM growth -> gelrigidity.dynamics
   - single, consistent Q definition           -> gelrigidity.handoff

 Do not import this module in new code. It is kept only so the original
 repository's provenance and the audit trail remain reproducible.
================================================================================
"""

"""Module 5 — Dual Percolation Crossover and Parameter Space Analysis.

The core design tool of this project: the "percolation handoff" metric Q
that quantifies whether mechanical load transfers smoothly from the degrading
hydrogel scaffold to the growing collagen ECM network.

Physical picture
----------------
At each timestep the simulation tracks two competing percolation order
parameters:

    P_inf_hydrogel(t)  --  decreasing as MMP enzymes cleave crosslinks
    P_inf_collagen(t)  --  increasing as fibroblasts deposit ECM fibres

The *handoff time* t* is defined as the moment both networks are near their
respective percolation thresholds and the two order parameters are closest
in value:

    t* = argmin_t |P_inf_hydrogel(t) - P_inf_collagen(t)|
         subject to: P_inf_hydrogel near p_c  AND  P_inf_collagen near p_c

The *handoff quality* Q quantifies the rate of mechanical transfer:

    Q = dP_inf_collagen/dt|_{t=t*}  -  dP_inf_hydrogel/dt|_{t=t*}

    Q > 0  --> collagen self-supports faster than hydrogel fails  --> SMOOTH
    Q <= 0 --> scaffold fails before collagen percolates          --> CATASTROPHIC

A parameter-space sweep maps Q over a 2-D grid of formulation parameters,
producing a design map with a phase boundary at Q = 0.

References
----------
* Stauffer & Aharony, "Introduction to Percolation Theory" (1994).
* Winter & Chambon, J. Rheol. 30, 367 (1986).
* Rubinstein & Colby, "Polymer Physics", OUP (2003).
"""

from __future__ import annotations

import logging
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    Figure = None  # type: ignore[assignment,misc]

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

logger = logging.getLogger("gel_percolation.percolation_analysis")

__all__ = [
    "DualPercolationTracker",
    "CriticalExponentFitter",
    "ParameterSpaceSweeper",
    "RheologyValidator",
    "SummaryReporter",
]

# 3-D bond percolation threshold (simple cubic lattice)
_P_C_3D: float = 0.2593

# Canonical 3-D percolation critical exponents
_BETA_3D: float = 0.418    # order parameter: P_inf ~ (p - p_c)^beta
_NU_3D: float = 0.88       # correlation length: xi ~ |p - p_c|^{-nu}
_F_BOND_BENDING: float = 2.1    # elastic modulus (bond-bending class, e.g. collagen)
_F_CENTRAL_FORCE: float = 3.75  # elastic modulus (central-force class, e.g. PEG)


# ===========================================================================
# 1.  DualPercolationTracker
# ===========================================================================


class DualPercolationTracker:
    """Track P_inf_hydrogel(t) and P_inf_collagen(t) and compute the handoff.

    Record percolation order parameters at each timestep, then compute the
    crossover time t*, handoff quality Q, and produce publication-quality
    diagnostic plots.

    Parameters
    ----------
    p_c_hydrogel : float
        Percolation threshold for the hydrogel network.  Default 0.2593 (3-D).
    p_c_collagen : float
        Percolation threshold for the collagen network.  Default 0.2593.

    Examples
    --------
    >>> tracker = DualPercolationTracker()
    >>> for t, p_hyd, p_col in zip(times, hydrogel_p_inf_arr, collagen_p_inf_arr):
    ...     tracker.record(t, p_hyd, p_col)
    >>> Q = tracker.compute_handoff_quality()
    >>> fig = tracker.plot_dual_percolation()
    """

    def __init__(self, p_c_hydrogel: float = 0.33, p_c_collagen: float = 0.33) -> None:
        """
        Parameters
        ----------
        p_c_hydrogel : float
            Percolation threshold for the hydrogel network.  Default 0.33 (Bethe
            estimate for z≈4 RGG); use measured value from
            HydrogelNetwork.measure_percolation_threshold() for accuracy.
        p_c_collagen : float
            Percolation threshold for the collagen network.  Default 0.33.
        """
        self.p_c_hydrogel = p_c_hydrogel
        self.p_c_collagen = p_c_collagen
        self._times: List[float] = []
        self._p_hyd: List[float] = []
        self._p_col: List[float] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record(
        self,
        time: float,
        p_inf_hydrogel: float,
        p_inf_collagen: float,
    ) -> None:
        """Append one time-point to the internal buffers.

        Parameters
        ----------
        time : float
            Simulation time [s].
        p_inf_hydrogel : float
            Hydrogel percolation order parameter P_inf in [0, 1].
        p_inf_collagen : float
            Collagen percolation order parameter P_inf in [0, 1].
        """
        self._times.append(float(time))
        self._p_hyd.append(float(p_inf_hydrogel))
        self._p_col.append(float(p_inf_collagen))

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(times, p_hydrogel, p_collagen)`` as numpy arrays."""
        return (
            np.asarray(self._times, dtype=float),
            np.asarray(self._p_hyd, dtype=float),
            np.asarray(self._p_col, dtype=float),
        )

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_handoff_time(self) -> Optional[float]:
        """Find t* = time where |P_inf_hydrogel - P_inf_collagen| is minimised.

        The search is restricted to the *critical window* where both networks
        are near their respective percolation thresholds:

          * P_inf_hydrogel is below 0.6  (hydrogel is partially degraded)
          * P_inf_collagen is above 0.05 (some collagen has been deposited)

        This prevents the trivial minimum at t = 0 (when both are at extreme
        values far from p_c).  If no points satisfy the constraint the global
        minimum of |P_inf_hyd - P_inf_col| is returned.

        Returns
        -------
        float or None
            The handoff time t* in the same units as recorded times, or
            None if fewer than 3 data points have been recorded.
        """
        times, p_hyd, p_col = self.get_arrays()
        if len(times) < 3:
            logger.warning(
                "compute_handoff_time: fewer than 3 data points; returning None."
            )
            return None

        diff = np.abs(p_hyd - p_col)

        # Prefer the region where both networks are near their p_c
        near_critical = (p_hyd < 0.6) & (p_col > 0.05)
        if np.any(near_critical):
            sub_diff = diff[near_critical]
            sub_times = times[near_critical]
            t_star = float(sub_times[np.argmin(sub_diff)])
        else:
            # Fall back to global minimum
            t_star = float(times[np.argmin(diff)])

        logger.debug("compute_handoff_time: t* = %.3f s", t_star)
        return t_star

    def compute_handoff_quality(self) -> float:
        """Compute Q = dP_inf_collagen/dt|_{t*} - dP_inf_hydrogel/dt|_{t*}.

        Uses ``numpy.gradient`` for numerical differentiation on the
        unevenly-spaced time grid, then linearly interpolates both
        derivatives at t*.

        Returns
        -------
        float
            Handoff quality Q.  Positive indicates smooth mechanical
            transfer; negative indicates catastrophic scaffold failure.
            Returns 0.0 if fewer than 4 data points have been recorded
            or if no handoff time can be determined.
        """
        times, p_hyd, p_col = self.get_arrays()
        if len(times) < 4:
            logger.warning(
                "compute_handoff_quality: fewer than 4 data points; returning 0.0."
            )
            return 0.0

        t_star = self.compute_handoff_time()
        if t_star is None:
            return 0.0

        # Numerical derivatives on the full time grid
        dp_hyd_dt = np.gradient(p_hyd, times)
        dp_col_dt = np.gradient(p_col, times)

        # Interpolate at t*
        dp_hyd_tstar = float(np.interp(t_star, times, dp_hyd_dt))
        dp_col_tstar = float(np.interp(t_star, times, dp_col_dt))

        Q = dp_col_tstar - dp_hyd_tstar
        logger.debug(
            "compute_handoff_quality: dP_col/dt|t* = %.4g, "
            "dP_hyd/dt|t* = %.4g, Q = %.4g",
            dp_col_tstar, dp_hyd_tstar, Q,
        )
        return float(Q)

    def compute_crossover_type(self) -> str:
        """Return ``'smooth'`` if Q > 0, else ``'catastrophic'``.

        Returns
        -------
        str
            ``'smooth'`` -- wound heals; mechanical load transfers successfully.
            ``'catastrophic'`` -- scaffold fails before collagen percolates.
        """
        return "smooth" if self.compute_handoff_quality() > 0.0 else "catastrophic"

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_dual_percolation(self, figsize: Tuple[int, int] = (12, 6)) -> Optional["Figure"]:
        """Plot P_inf_hydrogel(t) and P_inf_collagen(t) on the same axes.

        The figure annotates:
          * Both percolation curves in contrasting colours
          * Horizontal dashed lines at the p_c thresholds
          * A vertical line at t* (handoff time)
          * A text box with the Q value and outcome

        Parameters
        ----------
        figsize : (width, height) in inches.  Default (12, 6).

        Returns
        -------
        matplotlib.figure.Figure or None
            None if matplotlib is unavailable.
        """
        if not _HAS_MPL:
            warnings.warn(
                "matplotlib is not installed; cannot produce dual-percolation plot.",
                ImportWarning,
                stacklevel=2,
            )
            return None

        times, p_hyd, p_col = self.get_arrays()
        if len(times) < 2:
            warnings.warn(
                "plot_dual_percolation: insufficient data (< 2 points).",
                UserWarning,
                stacklevel=2,
            )
            return None

        t_star = self.compute_handoff_time()
        Q = self.compute_handoff_quality()
        crossover = self.compute_crossover_type()

        fig, ax = plt.subplots(figsize=figsize)

        # --- Percolation curves ---
        ax.plot(
            times, p_hyd,
            color="steelblue", lw=2.5, ls="-",
            label="Hydrogel $P_\\infty$ (decreasing)",
        )
        ax.plot(
            times, p_col,
            color="tomato", lw=2.5, ls="--",
            label="Collagen $P_\\infty$ (increasing)",
        )

        # --- p_c threshold lines ---
        ax.axhline(
            self.p_c_hydrogel,
            color="steelblue", ls=":", lw=1.4, alpha=0.7,
            label=f"$p_{{c,\\mathrm{{hyd}}}}$ = {self.p_c_hydrogel:.4f}",
        )
        ax.axhline(
            self.p_c_collagen,
            color="tomato", ls=":", lw=1.4, alpha=0.7,
            label=f"$p_{{c,\\mathrm{{col}}}}$ = {self.p_c_collagen:.4f}",
        )

        # --- Handoff time marker ---
        if t_star is not None:
            ax.axvline(
                t_star,
                color="forestgreen", ls="--", lw=2.0,
                label=f"$t^*$ = {t_star:.1f} s",
            )

            # Q annotation box
            outcome_str = "Smooth   (wound heals)" if Q > 0 else "Catastrophic (failure)"
            q_color = "forestgreen" if Q > 0 else "firebrick"
            t_range = times[-1] - times[0]
            x_annot = t_star + 0.04 * t_range
            y_annot = 0.5
            bbox_props = dict(
                boxstyle="round,pad=0.4",
                facecolor="lightyellow",
                edgecolor=q_color,
                alpha=0.9,
            )
            ax.annotate(
                f"$Q$ = {Q:+.4f}\n{outcome_str}",
                xy=(t_star, y_annot),
                xytext=(x_annot, y_annot),
                fontsize=11,
                color=q_color,
                fontweight="bold",
                bbox=bbox_props,
                arrowprops=dict(
                    arrowstyle="->",
                    color=q_color,
                    lw=1.5,
                ),
            )

        ax.set_xlabel("Time [s]", fontsize=13)
        ax.set_ylabel("Percolation order parameter $P_\\infty$", fontsize=13)
        ax.set_title(
            "Dual Percolation Dynamics: Hydrogel → Collagen Handoff\n"
            f"Crossover type: {crossover.upper()}",
            fontsize=13,
        )
        ax.legend(fontsize=10, loc="center right")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        fig.tight_layout()
        return fig


# ===========================================================================
# 2.  ParameterSpaceSweeper
# ===========================================================================


class ParameterSpaceSweeper:
    """Map the handoff quality Q across a 2-D grid of formulation parameters.

    Each grid point is a full WoundHealingSimulation run, so this class is
    the computationally intensive "design loop" of the project.  Parallel
    execution is supported via joblib.

    Parameters
    ----------
    base_hydrogel_params : HydrogelParams or None
        Base hydrogel network parameters (Module 1).  If None a default
        HydrogelParams() is used inside each run.
    base_cell_params : CellParams or None
        Base fibroblast parameters (Module 4).  If None a default
        CellParams() is used.
    base_sim_params : SimParams or None
        Base simulation time-stepping parameters (Module 4).  If None
        a default SimParams() is used.
    """

    def __init__(
        self,
        base_hydrogel_params=None,
        base_cell_params=None,
        base_sim_params=None,
    ) -> None:
        self.base_hydrogel_params = base_hydrogel_params
        self.base_cell_params = base_cell_params
        self.base_sim_params = base_sim_params

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    def sweep_2d(
        self,
        param1_name: str,
        param1_values: Union[np.ndarray, List[float]],
        param2_name: str,
        param2_values: Union[np.ndarray, List[float]],
        n_jobs: int = 4,
    ):
        """Run the simulation for every (param1, param2) combination.

        Parameter names are matched against the fields of CellParams,
        SimParams, and HydrogelParams (in that order) by attribute name.
        The first object that owns the attribute receives the override.

        Parameters
        ----------
        param1_name : str
            Attribute name of the first swept parameter.
        param1_values : array-like
            Values to sweep for param1.
        param2_name : str
            Attribute name of the second swept parameter.
        param2_values : array-like
            Values to sweep for param2.
        n_jobs : int
            Number of parallel joblib workers.  Default 4.  Set 1 to
            disable parallelism (useful for debugging).

        Returns
        -------
        pandas.DataFrame (if pandas is available) or list of dicts
            Columns: ``param1_name``, ``param2_name``, ``Q``,
            ``t_star``, ``invasion_depth``, ``outcome``.
        """
        param1_values = list(param1_values)
        param2_values = list(param2_values)
        combos = [(p1, p2) for p1 in param1_values for p2 in param2_values]

        def _set_param(obj, name: str, value) -> bool:
            """Set attribute on obj if it exists; return True on success."""
            if obj is not None and hasattr(obj, name):
                object.__setattr__(obj, name, value)
                return True
            return False

        def _run_one(p1_val: float, p2_val: float) -> Dict[str, Any]:
            """Execute one simulation and return the result record."""
            try:
                from .cell_invasion import (
                    CellParams, SimParams, WoundHealingSimulation,
                )

                cp = deepcopy(self.base_cell_params) if self.base_cell_params is not None else CellParams()
                sp = deepcopy(self.base_sim_params) if self.base_sim_params is not None else SimParams()

                try:
                    from .network_model import HydrogelParams
                    hp = deepcopy(self.base_hydrogel_params) if self.base_hydrogel_params is not None else None
                except ImportError:
                    hp = None

                # Apply parameter overrides: search CellParams, SimParams, HydrogelParams
                for name, val in [(param1_name, p1_val), (param2_name, p2_val)]:
                    set_cp = _set_param(cp, name, val)
                    set_sp = _set_param(sp, name, val) if not set_cp else False
                    if hp is not None and not set_cp and not set_sp:
                        _set_param(hp, name, val)

                sim = WoundHealingSimulation(
                    hydrogel_params=hp,
                    cell_params=cp,
                    sim_params=sp,
                )
                sim.initialize()
                history = sim.run()

                tracker = DualPercolationTracker()
                for state in history:
                    tracker.record(state.time, state.hydrogel_p_inf, state.collagen_p_inf)

                Q = tracker.compute_handoff_quality()
                t_star = tracker.compute_handoff_time()
                inv_depth = sim.get_invasion_depth()

                return {
                    param1_name: p1_val,
                    param2_name: p2_val,
                    "Q": float(Q),
                    "t_star": float(t_star) if t_star is not None else float("nan"),
                    "invasion_depth": float(inv_depth),
                    "outcome": "smooth" if Q > 0.0 else "catastrophic",
                }

            except Exception as exc:
                logger.warning(
                    "Sweep point (%s=%s, %s=%s) failed: %s",
                    param1_name, p1_val, param2_name, p2_val, exc,
                )
                return {
                    param1_name: p1_val,
                    param2_name: p2_val,
                    "Q": float("nan"),
                    "t_star": float("nan"),
                    "invasion_depth": float("nan"),
                    "outcome": "error",
                }

        # Parallel or serial execution
        if _HAS_JOBLIB and n_jobs != 1 and len(combos) > 1:
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_run_one)(p1, p2) for p1, p2 in combos
            )
        else:
            results = [_run_one(p1, p2) for p1, p2 in combos]

        if _HAS_PANDAS:
            return pd.DataFrame(results)
        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_Q_heatmap(
        self,
        sweep_results,
        param1_name: str,
        param2_name: str,
    ) -> Optional["Figure"]:
        """2-D colourmap of the handoff quality Q across the parameter grid.

        The colourmap uses a diverging palette (red = catastrophic, blue =
        smooth) centred at Q = 0.  A bold black dashed contour at Q = 0
        marks the phase boundary between the two outcomes.

        Parameters
        ----------
        sweep_results : pandas.DataFrame or list of dicts
            Output of :py:meth:`sweep_2d`.
        param1_name : str
            Column name for the first swept parameter (y-axis).
        param2_name : str
            Column name for the second swept parameter (x-axis).

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if not _HAS_MPL:
            warnings.warn(
                "matplotlib is not installed; cannot produce Q heatmap.",
                ImportWarning,
                stacklevel=2,
            )
            return None

        if not _HAS_PANDAS or not isinstance(sweep_results, pd.DataFrame):
            warnings.warn(
                "pandas DataFrame is required for plot_Q_heatmap; skipping.",
                UserWarning,
                stacklevel=2,
            )
            return None

        p1_vals = sorted(sweep_results[param1_name].unique())
        p2_vals = sorted(sweep_results[param2_name].unique())

        Q_grid = np.full((len(p1_vals), len(p2_vals)), np.nan)
        for i, p1 in enumerate(p1_vals):
            for j, p2 in enumerate(p2_vals):
                mask = (
                    (sweep_results[param1_name] == p1) &
                    (sweep_results[param2_name] == p2)
                )
                rows = sweep_results.loc[mask, "Q"]
                if len(rows) > 0:
                    Q_grid[i, j] = float(rows.values[0])

        fig, ax = plt.subplots(figsize=(9, 7))

        vmax = float(np.nanmax(np.abs(Q_grid)))
        vmax = vmax if vmax > 0 else 1.0

        im = ax.imshow(
            Q_grid,
            extent=[
                float(min(p2_vals)), float(max(p2_vals)),
                float(min(p1_vals)), float(max(p1_vals)),
            ],
            origin="lower",
            aspect="auto",
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Handoff quality Q", fontsize=12)

        # Q = 0 phase boundary
        try:
            ax.contour(
                p2_vals,
                p1_vals,
                Q_grid,
                levels=[0.0],
                colors=["black"],
                linewidths=[2.5],
                linestyles=["--"],
            )
        except Exception:
            pass  # contour fails if all-NaN or no zero-crossing

        ax.set_xlabel(param2_name, fontsize=12)
        ax.set_ylabel(param1_name, fontsize=12)
        ax.set_title(
            "Parameter Space Design Map: Handoff Quality Q\n"
            "--- = Q = 0 phase boundary (smooth/catastrophic)",
            fontsize=12,
        )
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Optimal formulation
    # ------------------------------------------------------------------

    def identify_optimal_formulation(
        self,
        sweep_results,
        min_invasion_depth: float = 10.0,
    ) -> Dict[str, Any]:
        """Return the parameter set that maximises Q.

        The optimum is constrained to rows where ``invasion_depth``
        exceeds ``min_invasion_depth`` µm, reflecting the biological
        requirement that fibroblasts actually invade the scaffold.  If no
        row satisfies the constraint the best unconstrainted row is returned.

        Parameters
        ----------
        sweep_results : pandas.DataFrame or list of dicts
        min_invasion_depth : float
            Minimum required invasion depth [µm].  Default 10.0.

        Returns
        -------
        dict
            Row of ``sweep_results`` with the highest Q (subject to the
            invasion-depth constraint).
        """
        if _HAS_PANDAS and isinstance(sweep_results, pd.DataFrame):
            feasible = sweep_results[
                sweep_results["invasion_depth"] >= min_invasion_depth
            ]
            if len(feasible) == 0:
                logger.warning(
                    "identify_optimal_formulation: no rows satisfy "
                    "invasion_depth >= %.1f; using unconstrained optimum.",
                    min_invasion_depth,
                )
                feasible = sweep_results

            best_idx = feasible["Q"].idxmax()
            return dict(feasible.loc[best_idx])

        elif isinstance(sweep_results, list):
            valid = [
                r for r in sweep_results
                if not np.isnan(float(r.get("Q", float("nan"))))
                and float(r.get("invasion_depth", 0.0)) >= min_invasion_depth
            ]
            if not valid:
                logger.warning(
                    "identify_optimal_formulation: no feasible rows found; "
                    "using unconstrained optimum."
                )
                valid = [r for r in sweep_results if not np.isnan(float(r.get("Q", float("nan"))))]
            if not valid:
                return {}
            return max(valid, key=lambda r: float(r.get("Q", float("-inf"))))

        logger.warning(
            "identify_optimal_formulation: unrecognised sweep_results type %s.",
            type(sweep_results),
        )
        return {}


# ===========================================================================
# 3.  CriticalExponentFitter
# ===========================================================================


class CriticalExponentFitter:
    """Fit power-law critical exponents to simulation data near p_c.

    Implements three complementary fits:

    * Order parameter:       P_inf ~ |p - p_c|^beta   (beta_3d = 0.418)
    * Elastic modulus:       G'    ~ |p - p_c|^f      (f = 2.1 or 3.75)
    * Correlation length:    xi    ~ |p - p_c|^{-nu}  (nu_3d = 0.88)

    All fit methods return dictionaries containing the fitted parameter,
    the expected 3-D value for comparison, and a goodness-of-fit R^2.

    Examples
    --------
    >>> fitter = CriticalExponentFitter()
    >>> result = fitter.fit_percolation_exponent(p_arr, P_inf_arr)
    >>> print(result['beta_fit'], result['beta_3d_expected'])
    """

    # ------------------------------------------------------------------
    # Order-parameter exponent
    # ------------------------------------------------------------------

    @staticmethod
    def fit_percolation_exponent(
        p_values: np.ndarray,
        order_parameter_values: np.ndarray,
        p_c_guess: Optional[float] = _P_C_3D,
    ) -> Dict[str, float]:
        """Fit P_inf ~ |p - p_c|^beta near p_c.

        If ``p_c_guess`` is provided (not None), it is used as a fixed
        threshold and only data points with p > p_c_guess enter the fit
        (existing behaviour).  If ``p_c_guess`` is None, both p_c and beta
        are fitted simultaneously as free parameters, with p_c constrained
        to [0.05, 0.5].

        Parameters
        ----------
        p_values : np.ndarray, shape (N,)
            Bond occupation probabilities.
        order_parameter_values : np.ndarray, shape (N,)
            Corresponding P_inf values.
        p_c_guess : float or None
            Initial guess for the percolation threshold.  Default 0.2593
            (simple cubic lattice).  Pass None to fit p_c as a free
            parameter alongside beta (preferred for RGG topologies).

        Returns
        -------
        dict with keys:
            p_c_fit             -- float, fitted threshold
            beta_fit            -- float, fitted exponent
            beta_3d_expected    -- float, canonical 3-D value (0.418)
            fit_quality         -- float, R^2 of the fit
        """
        p_values = np.asarray(p_values, dtype=float)
        order_parameter_values = np.asarray(order_parameter_values, dtype=float)

        if p_c_guess is None:
            # Fit both p_c and beta simultaneously
            p_vals = p_values
            P_inf_vals = order_parameter_values

            def model_free(p, p_c_fit, beta_fit):
                eps = np.maximum(p - p_c_fit, 0.0)
                return eps ** beta_fit

            try:
                popt, _ = curve_fit(
                    model_free, p_vals, P_inf_vals,
                    p0=[0.3, 0.418],
                    bounds=([0.05, 0.1], [0.5, 2.0]),
                    maxfev=5000,
                )
                p_c_fit, beta_fit = popt
            except RuntimeError:
                p_c_fit, beta_fit = p_c_guess or 0.3, 0.418

            p_c_fit = float(p_c_fit)
            beta_fit = float(beta_fit)

            P_pred = model_free(p_vals, p_c_fit, beta_fit)
            ss_res = float(np.sum((P_inf_vals - P_pred) ** 2))
            ss_tot = float(np.sum((P_inf_vals - P_inf_vals.mean()) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + 1e-30)

            return {
                "p_c_fit": p_c_fit,
                "beta_fit": beta_fit,
                "beta_3d_expected": _BETA_3D,
                "fit_quality": float(r2),
            }

        else:
            # Original fixed-p_c fit (existing code)
            above_pc = p_values > p_c_guess
            if above_pc.sum() < 3:
                logger.warning(
                    "fit_percolation_exponent: fewer than 3 points above p_c_guess; "
                    "returning defaults."
                )
                return {
                    "p_c_fit": float(p_c_guess),
                    "beta_fit": _BETA_3D,
                    "beta_3d_expected": _BETA_3D,
                    "fit_quality": 0.0,
                }

            p_fit = p_values[above_pc]
            P_fit = order_parameter_values[above_pc]

            def _model(p: np.ndarray, p_c: float, beta: float) -> np.ndarray:
                eps = np.abs(p - p_c)
                eps = np.where(eps < 1e-10, 1e-10, eps)
                return eps ** beta

            try:
                popt, _ = curve_fit(
                    _model,
                    p_fit,
                    P_fit,
                    p0=[p_c_guess, _BETA_3D],
                    bounds=([0.0, 0.05], [1.0, 3.0]),
                    maxfev=8000,
                )
                p_c_fit, beta_fit = float(popt[0]), float(popt[1])

                P_pred = _model(p_fit, *popt)
                ss_res = float(np.sum((P_fit - P_pred) ** 2))
                ss_tot = float(np.sum((P_fit - P_fit.mean()) ** 2))
                r2 = 1.0 - ss_res / (ss_tot + 1e-30)

                return {
                    "p_c_fit": p_c_fit,
                    "beta_fit": beta_fit,
                    "beta_3d_expected": _BETA_3D,
                    "fit_quality": float(r2),
                }
            except Exception as exc:
                logger.warning("fit_percolation_exponent failed: %s", exc)
                return {
                    "p_c_fit": float(p_c_guess),
                    "beta_fit": _BETA_3D,
                    "beta_3d_expected": _BETA_3D,
                    "fit_quality": 0.0,
                }

    # ------------------------------------------------------------------
    # Modulus scaling exponent
    # ------------------------------------------------------------------

    @staticmethod
    def fit_modulus_scaling(
        p_values: np.ndarray,
        modulus_values: np.ndarray,
        p_c: float = _P_C_3D,
    ) -> Dict[str, float]:
        """Fit G' ~ |p - p_c|^f near p_c.

        A log-log linear regression is used on points where p > p_c and
        G' > 0.  The fit exponent is compared to both universality classes:
        bond-bending (f = 2.1) and central-force (f = 3.75).

        Parameters
        ----------
        p_values : np.ndarray, shape (N,)
        modulus_values : np.ndarray, shape (N,)
            Storage modulus G' [Pa].
        p_c : float
            Known or estimated percolation threshold.

        Returns
        -------
        dict with keys:
            f_fit                       -- float, fitted exponent
            f_expected_bond_bending     -- float, 2.1 (alginate / collagen)
            f_expected_central_force    -- float, 3.75 (PEG / polyacrylamide)
            fit_quality                 -- float, R^2 on log-log scale
        """
        p_values = np.asarray(p_values, dtype=float)
        modulus_values = np.asarray(modulus_values, dtype=float)

        valid = (p_values > p_c) & (modulus_values > 0.0) & np.isfinite(modulus_values)
        if valid.sum() < 3:
            return {
                "f_fit": _F_BOND_BENDING,
                "f_expected_bond_bending": _F_BOND_BENDING,
                "f_expected_central_force": _F_CENTRAL_FORCE,
                "fit_quality": 0.0,
            }

        eps = np.abs(p_values[valid] - p_c) + 1e-15
        G = modulus_values[valid]

        try:
            log_eps = np.log(eps)
            log_G = np.log(G)
            coeffs = np.polyfit(log_eps, log_G, 1)
            f_fit = float(coeffs[0])

            log_G_pred = np.polyval(coeffs, log_eps)
            ss_res = float(np.sum((log_G - log_G_pred) ** 2))
            ss_tot = float(np.sum((log_G - log_G.mean()) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + 1e-30)

            return {
                "f_fit": f_fit,
                "f_expected_bond_bending": _F_BOND_BENDING,
                "f_expected_central_force": _F_CENTRAL_FORCE,
                "fit_quality": float(r2),
            }
        except Exception as exc:
            logger.warning("fit_modulus_scaling failed: %s", exc)
            return {
                "f_fit": _F_BOND_BENDING,
                "f_expected_bond_bending": _F_BOND_BENDING,
                "f_expected_central_force": _F_CENTRAL_FORCE,
                "fit_quality": 0.0,
            }

    # ------------------------------------------------------------------
    # Correlation-length exponent
    # ------------------------------------------------------------------

    @staticmethod
    def fit_correlation_length(
        p_values: np.ndarray,
        xi_values: np.ndarray,
        p_c: float = _P_C_3D,
    ) -> Dict[str, float]:
        """Fit xi ~ |p - p_c|^{-nu}.

        Both sides of p_c are used; the expected divergence (xi → inf as
        p → p_c) means only points with |p - p_c| > 0 and xi > 0 are
        included.

        Parameters
        ----------
        p_values : np.ndarray, shape (N,)
        xi_values : np.ndarray, shape (N,)
            Measured or estimated correlation lengths [µm].
        p_c : float
            Known or estimated percolation threshold.

        Returns
        -------
        dict with keys:
            nu_fit          -- float, fitted exponent (negated log-log slope)
            nu_expected     -- float, 0.88 (3-D random percolation)
            fit_quality     -- float, R^2 on log-log scale
        """
        p_values = np.asarray(p_values, dtype=float)
        xi_values = np.asarray(xi_values, dtype=float)

        valid = (p_values != p_c) & (xi_values > 0.0) & np.isfinite(xi_values)
        if valid.sum() < 3:
            return {
                "nu_fit": _NU_3D,
                "nu_expected": _NU_3D,
                "fit_quality": 0.0,
            }

        eps = np.abs(p_values[valid] - p_c) + 1e-15
        xi = xi_values[valid]

        try:
            log_eps = np.log(eps)
            log_xi = np.log(xi)
            # xi ~ eps^{-nu}  =>  log(xi) = -nu * log(eps) + const
            # Linear fit: slope = -nu
            coeffs = np.polyfit(log_eps, log_xi, 1)
            nu_fit = float(-coeffs[0])

            log_xi_pred = np.polyval(coeffs, log_eps)
            ss_res = float(np.sum((log_xi - log_xi_pred) ** 2))
            ss_tot = float(np.sum((log_xi - log_xi.mean()) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + 1e-30)

            return {
                "nu_fit": nu_fit,
                "nu_expected": _NU_3D,
                "fit_quality": float(r2),
            }
        except Exception as exc:
            logger.warning("fit_correlation_length failed: %s", exc)
            return {
                "nu_fit": _NU_3D,
                "nu_expected": _NU_3D,
                "fit_quality": 0.0,
            }

    # ------------------------------------------------------------------
    # Scaling collapse plot
    # ------------------------------------------------------------------

    @staticmethod
    def plot_scaling_collapse(
        p_values: np.ndarray,
        data: np.ndarray,
        exponent: float,
        p_c: float,
        xlabel: str = "$|p - p_c|$",
        ylabel: str = "Observable",
    ) -> Optional["Figure"]:
        """Log–log plot showing power-law scaling near p_c.

        The data points are plotted against |p - p_c| and a reference
        power-law guide line with the given exponent is overlaid.

        Parameters
        ----------
        p_values : np.ndarray, shape (N,)
        data : np.ndarray, shape (N,)
        exponent : float
            The expected power-law exponent for the guide line.
        p_c : float
            Percolation threshold.
        xlabel : str
            x-axis label.  Default r'$|p - p_c|$'.
        ylabel : str
            y-axis label.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if not _HAS_MPL:
            warnings.warn(
                "matplotlib is not installed; cannot produce scaling collapse plot.",
                ImportWarning,
                stacklevel=2,
            )
            return None

        p_values = np.asarray(p_values, dtype=float)
        data = np.asarray(data, dtype=float)

        eps = np.abs(p_values - p_c)
        valid = (eps > 0) & (data > 0) & np.isfinite(data)

        if valid.sum() < 2:
            warnings.warn(
                "plot_scaling_collapse: too few valid data points.",
                UserWarning,
                stacklevel=2,
            )
            return None

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.loglog(
            eps[valid],
            data[valid],
            "o",
            ms=7,
            alpha=0.75,
            color="steelblue",
            label="Simulation data",
        )

        # Guide line: y = A * eps^exponent  (A fitted from median)
        A = float(np.median(data[valid] / (eps[valid] ** exponent + 1e-15)))
        eps_range = np.logspace(
            np.log10(eps[valid].min() + 1e-12),
            np.log10(eps[valid].max()),
            60,
        )
        ax.loglog(
            eps_range,
            A * eps_range ** exponent,
            "r--",
            lw=2,
            label=f"$\\propto |p - p_c|^{{{exponent:.3f}}}$",
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"Power-Law Scaling Near Percolation Threshold\n"
            f"$p_c$ = {p_c:.4f},  exponent = {exponent:.3f}",
            fontsize=12,
        )
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
        fig.tight_layout()
        return fig


# ===========================================================================
# 4.  RheologyValidator
# ===========================================================================


class RheologyValidator:
    """Compare model predictions to experimental oscillatory rheometry data.

    The class handles:
      * Loading experimental CSV files (time, G', G'', frequency columns)
      * RMSE / R^2 comparison between model and experiment across a
        frequency sweep
      * Gel-point detection (where G' = G'', i.e. tan δ = 1)
      * Overlay plots with log–log frequency axes

    Examples
    --------
    >>> validator = RheologyValidator()
    >>> exp_df = validator.load_experimental_data("rheo_data.csv")
    >>> comparison = validator.compare_frequency_sweep(
    ...     model_G_prime, exp_df['G_prime'].values, exp_df['frequency'].values
    ... )
    >>> print(comparison['RMSE'], comparison['R_squared'])
    """

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_experimental_data(filepath: str) -> "pd.DataFrame":
        """Load experimental rheometry data from a CSV file.

        The CSV is expected to contain at least a subset of the columns:
        ``time``, ``G_prime``, ``G_double_prime``, ``frequency``.  Extra
        columns are preserved unchanged.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If pandas is not installed.
        FileNotFoundError
            If the file does not exist.
        """
        if not _HAS_PANDAS:
            raise ImportError(
                "pandas is required for load_experimental_data; "
                "install with: pip install pandas"
            )
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Experimental data file not found: {filepath!r}")
        df = pd.read_csv(path)
        logger.info(
            "Loaded experimental data from '%s': %d rows, columns: %s",
            filepath, len(df), list(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Quantitative comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_frequency_sweep(
        model_G_prime: np.ndarray,
        exp_G_prime: np.ndarray,
        omega_values: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare model and experimental G' across a frequency sweep.

        Gel-point detection is performed by searching for the frequency at
        which the input storage modulus is closest to a loss-modulus proxy
        (a flat tangent-loss curve).  For the model the gel point is
        estimated as the frequency at which G' is closest to its geometric
        mean; for the experiment the same heuristic is applied.

        Parameters
        ----------
        model_G_prime : np.ndarray, shape (N,)
            Model predictions of the storage modulus [Pa].
        exp_G_prime : np.ndarray, shape (N,)
            Experimental storage modulus [Pa].
        omega_values : np.ndarray, shape (N,)
            Angular frequencies [rad s^-1].

        Returns
        -------
        dict with keys:
            RMSE              -- float, root-mean-square error [Pa]
            R_squared         -- float, coefficient of determination
            gel_point_model   -- float or None, frequency at model gel point
            gel_point_exp     -- float or None, frequency at experimental gel point
        """
        model_G_prime = np.asarray(model_G_prime, dtype=float)
        exp_G_prime = np.asarray(exp_G_prime, dtype=float)
        omega_values = np.asarray(omega_values, dtype=float)

        if len(model_G_prime) == 0 or len(exp_G_prime) == 0:
            return {
                "RMSE": float("nan"),
                "R_squared": float("nan"),
                "gel_point_model": None,
                "gel_point_exp": None,
            }

        # Align lengths
        n = min(len(model_G_prime), len(exp_G_prime), len(omega_values))
        model_G_prime = model_G_prime[:n]
        exp_G_prime = exp_G_prime[:n]
        omega_values = omega_values[:n]

        rmse = float(np.sqrt(np.mean((model_G_prime - exp_G_prime) ** 2)))
        ss_res = float(np.sum((exp_G_prime - model_G_prime) ** 2))
        ss_tot = float(np.sum((exp_G_prime - exp_G_prime.mean()) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-30))

        def _gel_point(G_prime: np.ndarray, omega: np.ndarray) -> Optional[float]:
            """Frequency at which G' is closest to its log-geometric mean.

            At the Winter-Chambon gel point G' and G'' have equal
            power-law dependence on frequency.  As a proxy we look for the
            frequency at which G' is nearest to exp(mean(log(G'))).
            """
            valid = (G_prime > 0) & np.isfinite(G_prime)
            if valid.sum() < 2:
                return None
            G_log_mean = float(np.exp(np.mean(np.log(G_prime[valid]))))
            idx = int(np.argmin(np.abs(G_prime[valid] - G_log_mean)))
            return float(omega[valid][idx])

        return {
            "RMSE": rmse,
            "R_squared": r2,
            "gel_point_model": _gel_point(model_G_prime, omega_values),
            "gel_point_exp": _gel_point(exp_G_prime, omega_values),
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def plot_comparison(
        model_data: np.ndarray,
        exp_data: np.ndarray,
        title: str = "",
    ) -> Optional["Figure"]:
        """Overlay model and experimental G' on a log–log frequency plot.

        Parameters
        ----------
        model_data : np.ndarray, shape (N,) or shape (2, N)
            If 1-D: treated as G' values over an implicit frequency axis
            (1 ... N) rad s^-1.
            If 2-D: row 0 is omega values, row 1 is G' values.
        exp_data : np.ndarray, shape (N,) or shape (2, N)
            Experimental counterpart, same convention as model_data.
        title : str
            Figure title.  Default '' → auto-generated.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if not _HAS_MPL:
            warnings.warn(
                "matplotlib is not installed; cannot produce comparison plot.",
                ImportWarning,
                stacklevel=2,
            )
            return None

        model_data = np.asarray(model_data, dtype=float)
        exp_data = np.asarray(exp_data, dtype=float)

        # Unpack (omega, G') if 2-D input
        if model_data.ndim == 2 and model_data.shape[0] == 2:
            omega_model = model_data[0]
            G_model = model_data[1]
        else:
            model_data = model_data.ravel()
            omega_model = np.arange(1, len(model_data) + 1, dtype=float)
            G_model = model_data

        if exp_data.ndim == 2 and exp_data.shape[0] == 2:
            omega_exp = exp_data[0]
            G_exp = exp_data[1]
        else:
            exp_data = exp_data.ravel()
            omega_exp = np.arange(1, len(exp_data) + 1, dtype=float)
            G_exp = exp_data

        fig, ax = plt.subplots(figsize=(8, 5))

        valid_model = (G_model > 0) & np.isfinite(G_model)
        valid_exp = (G_exp > 0) & np.isfinite(G_exp)

        ax.loglog(
            omega_model[valid_model], G_model[valid_model],
            "b-", lw=2.5, label="Model G'",
        )
        ax.loglog(
            omega_exp[valid_exp], G_exp[valid_exp],
            "ro", ms=7, alpha=0.8, label="Experiment G'",
        )

        ax.set_xlabel("$\\omega$ [rad s$^{-1}$]", fontsize=12)
        ax.set_ylabel("G' [Pa]", fontsize=12)
        ax.set_title(title or "Model vs Experiment: Storage Modulus G'($\\omega$)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
        fig.tight_layout()
        return fig


# ===========================================================================
# 5.  SummaryReporter
# ===========================================================================


class SummaryReporter:
    """Generate text reports and export all simulation results to disk.

    The reporter aggregates results from all upstream modules (simulation
    history, dual percolation tracker, EWS analysis) into a formatted text
    summary, and optionally exports figures, arrays, and the report itself
    to a structured output directory.
    """

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_report(
        simulation_history: list,
        dual_percolation: Optional[DualPercolationTracker] = None,
        ews_results: Optional[Dict] = None,
    ) -> str:
        """Return a formatted text report summarising all key findings.

        Parameters
        ----------
        simulation_history : list of SimulationState
            History list returned by ``WoundHealingSimulation.run()``.
        dual_percolation : DualPercolationTracker or None
            Filled tracker for handoff analysis.
        ews_results : dict or None
            Output of ``EarlyWarningSignalDetector.compute_ews_indicators``.

        Returns
        -------
        str
            Multi-section formatted report.
        """
        sep = "=" * 72
        dash = "-" * 36
        lines: List[str] = []

        lines += [
            sep,
            "  GEL PERCOLATION SIMULATION REPORT",
            "  Percolation Inversion Dynamics in Enzymatically Degrading",
            "  Wound Hydrogels",
            sep,
            "",
        ]

        # ------ Simulation summary ------
        if simulation_history:
            final = simulation_history[-1]
            initial = simulation_history[0]
            n_steps = len(simulation_history)

            p_hyd_arr = np.asarray(
                [s.hydrogel_p_inf for s in simulation_history], dtype=float
            )
            p_col_arr = np.asarray(
                [s.collagen_p_inf for s in simulation_history], dtype=float
            )

            lines += [
                "SIMULATION SUMMARY",
                dash,
                f"  Recorded snapshots:         {n_steps}",
                f"  Simulation duration:        {final.time:.2f} s",
                f"  Initial hydrogel P_inf:     {initial.hydrogel_p_inf:.4f}",
                f"  Final   hydrogel P_inf:     {final.hydrogel_p_inf:.4f}",
                f"  Initial collagen P_inf:     {initial.collagen_p_inf:.4f}",
                f"  Final   collagen P_inf:     {final.collagen_p_inf:.4f}",
                f"  Final invasion depth:       {final.invasion_depth:.2f} um",
                f"  Total collagen fibres:      {final.n_collagen_fibers}",
                f"  Peak hydrogel P_inf:        {p_hyd_arr.max():.4f}",
                f"  Minimum collagen P_inf:     {p_col_arr.min():.4f}",
                "",
            ]
        else:
            lines += ["SIMULATION SUMMARY", dash, "  No simulation history provided.", ""]

        # ------ Percolation handoff ------
        if dual_percolation is not None:
            Q = dual_percolation.compute_handoff_quality()
            t_star = dual_percolation.compute_handoff_time()
            crossover = dual_percolation.compute_crossover_type()
            outcome_str = "WOUND HEALS (smooth mechanical transfer)" if Q > 0.0 \
                else "WOUND FAILURE (scaffold fails before collagen percolates)"

            lines += [
                "PERCOLATION HANDOFF ANALYSIS",
                dash,
                f"  Handoff time t*:            {t_star:.2f} s" if t_star is not None
                else "  Handoff time t*:            N/A (insufficient data)",
                f"  Handoff quality Q:          {Q:+.6f}",
                f"  Crossover type:             {crossover.upper()}",
                f"  Clinical outcome:           {outcome_str}",
                f"  p_c (hydrogel):             {dual_percolation.p_c_hydrogel:.4f}",
                f"  p_c (collagen):             {dual_percolation.p_c_collagen:.4f}",
                "",
            ]
        else:
            lines += ["PERCOLATION HANDOFF ANALYSIS", dash, "  Not available.", ""]

        # ------ Early warning signals ------
        if ews_results is not None:
            def _fmt(v: Any) -> str:
                if v is None:
                    return "N/A"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            lines += [
                "EARLY WARNING SIGNAL ANALYSIS",
                dash,
                f"  Kendall tau (AR1):          {_fmt(ews_results.get('kendall_tau_ar1'))}",
                f"  AR1 p-value:                {_fmt(ews_results.get('ar1_pvalue'))}",
                f"  Kendall tau (variance):     {_fmt(ews_results.get('kendall_tau_var'))}",
                f"  Variance p-value:           {_fmt(ews_results.get('var_pvalue'))}",
                f"  EWS onset time:             {_fmt(ews_results.get('ews_onset_time'))} s",
                f"  Gel-sol transition time:    {_fmt(ews_results.get('transition_time'))} s",
                f"  EWS lead time:              {_fmt(ews_results.get('lead_time'))} s",
                "",
            ]
        else:
            lines += ["EARLY WARNING SIGNAL ANALYSIS", dash, "  Not available.", ""]

        lines += [sep]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_results(
        output_dir: str,
        simulation_history: list,
        all_results: Optional[Dict] = None,
    ) -> None:
        """Save all arrays, figures, and the text report to *output_dir*.

        Directory structure created:

          ``output_dir/``
            ``arrays/``
              ``times.npy``
              ``p_inf_hydrogel.npy``
              ``p_inf_collagen.npy``
              ``invasion_depth.npy``
              ``n_collagen_fibers.npy``
            ``figures/``
              ``<figure_name>.png``  (one per Figure in all_results)
            ``report.txt``
            ``<dataframe_name>.csv``  (one per DataFrame in all_results)

        Parameters
        ----------
        output_dir : str
            Root output directory (created if it does not exist).
        simulation_history : list of SimulationState
            History list from ``WoundHealingSimulation.run()``.
        all_results : dict or None
            Arbitrary dictionary mapping names to results objects.
            Recognised value types:
              * numpy.ndarray      → saved to arrays/
              * pandas.DataFrame   → saved as CSV in output_dir/
              * matplotlib.Figure  → saved as PNG in figures/
              * str                → appended to report.txt
            Unknown types are ignored with a debug log message.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        arr_dir = out / "arrays"
        arr_dir.mkdir(exist_ok=True)
        fig_dir = out / "figures"
        fig_dir.mkdir(exist_ok=True)

        # ---- Save time-series arrays ----
        if simulation_history:
            times_arr = np.array([s.time for s in simulation_history], dtype=float)
            p_hyd_arr = np.array([s.hydrogel_p_inf for s in simulation_history], dtype=float)
            p_col_arr = np.array([s.collagen_p_inf for s in simulation_history], dtype=float)
            inv_arr = np.array([s.invasion_depth for s in simulation_history], dtype=float)
            ncol_arr = np.array([s.n_collagen_fibers for s in simulation_history], dtype=int)

            np.save(arr_dir / "times.npy", times_arr)
            np.save(arr_dir / "p_inf_hydrogel.npy", p_hyd_arr)
            np.save(arr_dir / "p_inf_collagen.npy", p_col_arr)
            np.save(arr_dir / "invasion_depth.npy", inv_arr)
            np.save(arr_dir / "n_collagen_fibers.npy", ncol_arr)
            logger.info("Saved time-series arrays to '%s'.", str(arr_dir))

        # ---- Process all_results dictionary ----
        extra_report_lines: List[str] = []

        if all_results is not None:
            for name, value in all_results.items():
                if isinstance(value, np.ndarray):
                    np.save(arr_dir / f"{name}.npy", value)
                    logger.info("Saved array '%s' to arrays/.", name)

                elif _HAS_PANDAS and isinstance(value, pd.DataFrame):
                    csv_path = out / f"{name}.csv"
                    value.to_csv(csv_path, index=False)
                    logger.info("Saved DataFrame '%s' to '%s'.", name, str(csv_path))

                elif _HAS_MPL and isinstance(value, matplotlib.figure.Figure):
                    fig_path = fig_dir / f"{name}.png"
                    value.savefig(str(fig_path), dpi=150, bbox_inches="tight")
                    logger.info("Saved figure '%s' to '%s'.", name, str(fig_path))

                elif isinstance(value, str):
                    extra_report_lines.append(f"\n[{name}]\n{value}")
                    logger.debug("Appending string result '%s' to report.", name)

                else:
                    logger.debug(
                        "export_results: skipping '%s' (type %s).",
                        name, type(value).__name__,
                    )

        # ---- Write text report ----
        report_path = out / "report.txt"
        try:
            base_report = SummaryReporter.generate_report(simulation_history)
            full_report = base_report
            if extra_report_lines:
                full_report += "\n\nADDITIONAL RESULTS\n" + "=" * 72
                full_report += "\n".join(extra_report_lines)

            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(full_report)
            logger.info("Report written to '%s'.", str(report_path))
        except Exception as exc:
            logger.error("Failed to write report: %s", exc)
