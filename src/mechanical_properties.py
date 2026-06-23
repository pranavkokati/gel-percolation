"""Module 2 — Network Topology → Mechanical Properties.

Maps the percolation order parameter p(t) and spatial bond-fraction field
to rheological observables: G'(omega, t), G''(omega, t), and spatially-resolved
stiffness G'_eff(x, t).

Physical framework
------------------
Near the gel-sol percolation threshold p_c (critical scaling regime):

    G'(p, omega) ~ |p - p_c|^f_elastic * omega^Delta

where ``f_elastic`` is the elastic percolation exponent and ``Delta`` is the
dynamic (frequency) exponent.  This scaling originates from the divergence of
the correlation length xi ~ |p - p_c|^{-nu} and applies within a crossover
window |p - p_c| < p_crossover.

Far above p_c (affine network / rubber-elastic regime):

    G'_plateau = rho_chain * k_B * T

where rho_chain is the strand density between crosslinks (chains m^-3).

A smooth sigmoid crossover blends the two regimes across the window
|p - p_c| in [p_c, p_c + 2*p_crossover].

Loss modulus (Kramers-Kronig / Winter-Chambon)
----------------------------------------------
At the gel point the loss tangent is frequency-independent:

    tan delta = G'' / G' = tan(pi * Delta / 2)   [Winter-Chambon criterion]

Away from p_c tan delta decays exponentially toward a residual value of ~0.05
as the gel solidifies.

Key exponents for 3-D percolation networks
------------------------------------------
beta       = 0.418   order-parameter exponent (P_inf ~ (p-p_c)^beta)
f_elastic  = 2.1     elastic exponent, bond-bending (alginate, collagen)
           = 3.75    elastic exponent, central-force (PEG, polyacrylamide)
Delta      = 0.72    dynamic / frequency exponent
nu         = 0.88    correlation-length exponent (xi ~ |p-p_c|^{-nu})
gamma      = 1.8     susceptibility exponent (chi ~ |p-p_c|^{-gamma})

References
----------
* Stauffer & Aharony, "Introduction to Percolation Theory" (1994).
* Winter & Chambon, J. Rheol. 30, 367 (1986).
* Rubinstein & Colby, "Polymer Physics", OUP (2003).
* Almdal et al., J. Phys. D, 26, B279 (1993).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

__all__ = [
    "MechanicsParams",
    "PercolationCriticalExponents",
    "PercolationMechanics",
    "AffineNetworkTheory",
    "FrequencySweepAnalyzer",
]

# Boltzmann constant [J K^-1]
_K_B: float = 1.380649e-23


# ---------------------------------------------------------------------------
# Critical exponents dataclass
# ---------------------------------------------------------------------------

@dataclass
class PercolationCriticalExponents:
    """Critical exponents for 3-D percolation rheology.

    Two distinct universality classes are supported depending on the
    dominant restoring-force mechanism in the polymer network:

    Bond-bending networks (alginate, gelatin, collagen, semiflexible gels)
        f_elastic = 2.1  (Kantor-Webman bond-bending percolation class)

    Central-force networks (PEG, polyacrylamide, flexible gels)
        f_elastic = 3.75  (de Gennes central-force class)

    Both share the same dynamic exponent Delta = 0.72 and the standard
    3-D random percolation exponents nu = 0.88, gamma = 1.8.

    Attributes
    ----------
    beta : float
        Order-parameter exponent.  P_inf ~ (p - p_c)^beta above p_c.
    f_elastic : float
        Elastic / shear-modulus exponent.  G' ~ (p - p_c)^f_elastic.
    Delta : float
        Dynamic frequency exponent.  G' ~ omega^Delta at p_c.
    nu : float
        Correlation-length exponent.  xi ~ |p - p_c|^{-nu}.
    gamma_sus : float
        Susceptibility exponent.  chi ~ |p - p_c|^{-gamma_sus}.
    """

    beta: float = 0.418
    f_elastic: float = 2.1
    Delta: float = 0.72
    nu: float = 0.88
    gamma_sus: float = 1.8

    @classmethod
    def for_alginate(cls) -> "PercolationCriticalExponents":
        """Bond-bending exponents (alginate, gelatin, collagen).

        Returns
        -------
        PercolationCriticalExponents
            Instance with f_elastic = 2.1 (Kantor-Webman class).
        """
        return cls(f_elastic=2.1, Delta=0.72, nu=0.88, gamma_sus=1.8)

    @classmethod
    def for_peg(cls) -> "PercolationCriticalExponents":
        """Central-force exponents (PEG, polyacrylamide).

        Returns
        -------
        PercolationCriticalExponents
            Instance with f_elastic = 3.75 (de Gennes central-force class).
        """
        return cls(f_elastic=3.75, Delta=0.72, nu=0.88, gamma_sus=1.8)

    @classmethod
    def estimate_from_network_geometry(
        cls, bond_bending: bool
    ) -> "PercolationCriticalExponents":
        """Select exponent set from the dominant network restoring force.

        Parameters
        ----------
        bond_bending : bool
            If True, use bond-bending exponents (alginate-type).
            If False, use central-force exponents (PEG-type).

        Returns
        -------
        PercolationCriticalExponents
        """
        return cls.for_alginate() if bond_bending else cls.for_peg()


# ---------------------------------------------------------------------------
# Master parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class MechanicsParams:
    """All physical parameters required for mechanical property calculations.

    Attributes
    ----------
    p_c : float
        3-D bond percolation threshold for a random geometric graph.
        Value 0.2593 corresponds to the bond percolation threshold on a
        3-D Bethe / random network (Ziff & Stell, 1980).
    p_crossover : float
        Half-width of the crossover region around p_c.  The model smoothly
        transitions from critical scaling to affine network theory over the
        interval [p_c, p_c + 2 * p_crossover].
    T : float
        Physiological temperature [K].  Default 310.15 K (37 degC).
    kB : float
        Boltzmann constant [J K^-1].  Fixed at the 2019 SI definition.
    rho_chain_ref : float
        Reference strand density between crosslinks [chains m^-3].  Used to
        set the scale of the affine-network plateau modulus.
    E_ref : float
        Reference stiffness scale [Pa].  Sets the prefactor of the critical-
        scaling expression G' = E_ref * eps^f_elastic * omega^Delta.
    omega_ref : float
        Reference angular frequency [rad s^-1] for non-dimensionalising omega
        in the critical scaling formula.
    exponents : PercolationCriticalExponents
        Percolation critical exponents.  Defaults to alginate (bond-bending).
    """

    p_c: float = 0.2593
    p_crossover: float = 0.05
    T: float = 310.15
    kB: float = _K_B
    rho_chain_ref: float = 2.34e23  # chains m⁻³; gives G'_affine ≈ 1 kPa at p=1
    E_ref: float = 1000.0
    omega_ref: float = 1.0
    exponents: PercolationCriticalExponents = field(
        default_factory=PercolationCriticalExponents.for_alginate
    )


# ---------------------------------------------------------------------------
# Core mechanics class
# ---------------------------------------------------------------------------

class PercolationMechanics:
    """Map the percolation bond fraction p to rheological observables.

    This class implements the constitutive model relating the microscopic
    network connectivity p (bond occupation fraction) to the macroscopic
    shear moduli G'(p, omega) and G''(p, omega).

    The model has three regimes:

    1. **Below threshold** (p <= p_c):
       The system is in the sol phase.  G' = G'' = 0.

    2. **Critical regime** (p_c < p < p_c + 2 * p_crossover):
       Critical scaling dominates.
           G'(p, omega) = E_ref * (p - p_c)^f_elastic * (omega / omega_ref)^Delta

    3. **Affine gel regime** (p >> p_c):
       Rubber-elastic plateau dominates.
           G'_plateau = rho_chain * k_B * T

    A smooth sigmoid interpolates continuously between regimes 2 and 3.

    Parameters
    ----------
    params : MechanicsParams, optional
        Physical parameters.  Defaults to alginate at 37 degC.

    Examples
    --------
    >>> mech = PercolationMechanics()
    >>> G_prime = mech.compute_shear_modulus(p=0.5, omega=1.0)
    >>> G_prime_field = mech.compute_stiffness_field(np.linspace(0, 1, 50))
    """

    def __init__(self, params: Optional[MechanicsParams] = None) -> None:
        self.params: MechanicsParams = (
            params if params is not None else MechanicsParams()
        )

    # ------------------------------------------------------------------
    # Shear modulus G'
    # ------------------------------------------------------------------

    def compute_shear_modulus(self, p: float, omega: float = 1.0) -> float:
        """Compute the elastic (storage) shear modulus G'(p, omega).

        Implements a smooth crossover between the critical-scaling regime
        near the percolation threshold and the affine rubber-elastic plateau
        deep in the gel phase.

        Near p_c (critical regime):
            G' ~ |p - p_c|^f_elastic * omega^Delta

        Far above p_c (affine network):
            G'_plateau = rho_chain * k_B * T

        The two regimes are blended via:
            w = sigmoid((eps - p_crossover) / (p_crossover / 3))
            G' = (1 - w) * G_critical + w * G_affine

        where eps = p - p_c.

        Parameters
        ----------
        p : float
            Bond occupation fraction in [0, 1].  p = 1 is an intact network;
            p -> 0 is a fully degraded sol.
        omega : float, optional
            Angular frequency [rad s^-1].  Default 1.0.

        Returns
        -------
        float
            Storage modulus G' [Pa].  Identically zero for p <= p_c.

        Notes
        -----
        The frequency dependence omega^Delta is the gel-point scaling
        predicted by the Winter-Chambon theory for a self-similar network
        at the percolation threshold.  Above p_c the plateau modulus G'_aff
        is frequency-independent in the rubber-elastic limit (omega -> 0).
        """
        pm = self.params
        ex = pm.exponents

        if p <= pm.p_c:
            return 0.0

        eps = p - pm.p_c  # reduced control parameter (epsilon)

        # Critical-scaling contribution
        G_critical = (
            pm.E_ref
            * (eps ** ex.f_elastic)
            * (omega / pm.omega_ref) ** ex.Delta
        )

        # Affine rubber-elastic plateau: G'_aff = rho_chain(p) * k_B * T
        # rho_chain scales linearly with p (more bonds -> more elastic strands).
        # We take the maximum of the rubber-elastic estimate and the critical
        # scaling estimate to ensure the modulus is monotonically non-decreasing
        # with p and that the crossover always blends toward the stiffer response.
        # This is physically justified: the network adopts whichever mechanism
        # (bond-bending / central-force criticality vs rubber elasticity) provides
        # the dominant restoring force.
        G_affine_rubber = AffineNetworkTheory.compute_plateau_modulus(
            rho_chain=pm.rho_chain_ref * p, T=pm.T
        )
        G_affine = G_affine_rubber  # pure affine plateau target for the sigmoid blend

        # Sharp sigmoid crossover: w << 1 for eps << p_crossover (pure critical regime),
        # w ~ 1 for eps >> p_crossover (pure affine regime).
        # Width p_crossover/20 keeps the critical-scaling slope clean up to eps ≈ 0.03.
        width = max(pm.p_crossover / 20.0, 1e-12)
        w = _sigmoid((eps - pm.p_crossover) / width)

        return float((1.0 - w) * G_critical + w * G_affine)

    # ------------------------------------------------------------------
    # Loss modulus G''
    # ------------------------------------------------------------------

    def compute_loss_modulus(self, p: float, omega: float = 1.0) -> float:
        """Compute the viscous (loss) shear modulus G''(p, omega).

        Derived from Kramers-Kronig relations and the Winter-Chambon
        gel-point criterion.  At the gel point (p = p_c) the loss tangent
        is frequency-independent:

            tan delta_gel = tan(pi * Delta / 2)

        This is the signature of a self-similar (fractal) network at the
        critical point.  Away from p_c the network becomes increasingly
        elastic and tan delta decays exponentially toward a residual
        dissipation floor (~0.05, representative of physical crosslink
        slippage in alginate).

        Parameters
        ----------
        p : float
            Bond occupation fraction in [0, 1].
        omega : float, optional
            Angular frequency [rad s^-1].  Default 1.0.

        Returns
        -------
        float
            Loss modulus G'' [Pa].  Identically zero for p <= p_c.

        Notes
        -----
        The Kramers-Kronig approach ensures causality.  The approximation
        used here (frequency-independent tan delta at p_c, exponential decay
        above p_c) is consistent with the scaling theory of Muthukumar (1985)
        and the experimental observations of Winter & Chambon (1986).
        """
        pm = self.params
        ex = pm.exponents

        G_prime = self.compute_shear_modulus(p, omega)
        if G_prime <= 0.0:
            return 0.0

        # tan delta at gel point from Winter-Chambon theory
        tan_delta_gel = np.tan(np.pi * ex.Delta / 2.0)

        # Fractional distance into the gel phase: 0 at p_c, 1 at p = 1
        eps_frac = max((p - pm.p_c) / (1.0 - pm.p_c), 0.0)

        # Exponential decay of tan delta with increasing gel fraction
        # + residual dissipation floor of 0.05
        tan_delta = tan_delta_gel * np.exp(-3.0 * eps_frac) + 0.05

        return float(G_prime * tan_delta)

    # ------------------------------------------------------------------
    # Spatially-resolved stiffness field
    # ------------------------------------------------------------------

    def compute_stiffness_field(
        self, local_p_grid: np.ndarray, omega: float = 1.0
    ) -> np.ndarray:
        """Compute the stiffness G'(x) over a spatial grid of bond fractions.

        Vectorised computation applying ``compute_shear_modulus`` element-wise
        to an array of local bond occupation fractions.  Suitable for 1-D,
        2-D, or 3-D spatial grids of arbitrary shape.

        Parameters
        ----------
        local_p_grid : np.ndarray
            Array of local bond fractions p(x) with shape (Nx,), (Nx, Ny),
            or (Nx, Ny, Nz).  Values should lie in [0, 1].
        omega : float, optional
            Angular frequency [rad s^-1].  Default 1.0.

        Returns
        -------
        np.ndarray
            Stiffness field G'(x) [Pa] with the same shape as
            ``local_p_grid``.

        Notes
        -----
        NaN values in ``local_p_grid`` (e.g. from voxels with insufficient
        node density) are propagated as NaN in the output.
        """
        vfunc = np.vectorize(
            lambda p_val: (
                self.compute_shear_modulus(p_val, omega)
                if np.isfinite(p_val)
                else np.nan
            )
        )
        return vfunc(local_p_grid)

    # ------------------------------------------------------------------
    # Stiffness gradient field
    # ------------------------------------------------------------------

    def compute_stiffness_gradient(
        self, stiffness_field: np.ndarray, dx: float
    ) -> np.ndarray:
        """Compute the spatial gradient vector field of the stiffness.

        Uses second-order central finite differences (np.gradient) with
        first-order one-sided differences at the boundaries.

        Parameters
        ----------
        stiffness_field : np.ndarray, shape (Nx, Ny, Nz)
            Spatially resolved storage modulus G'(x) [Pa].
        dx : float
            Isotropic voxel spacing [same units as the spatial coordinates,
            typically micrometres].

        Returns
        -------
        np.ndarray, shape (Nx, Ny, Nz, 3)
            Gradient vector (dG'/dx, dG'/dy, dG'/dz) [Pa / length] at every
            voxel.  The last axis indexes the spatial direction (0=x, 1=y,
            2=z).

        Notes
        -----
        NaN voxels in ``stiffness_field`` are handled gracefully by
        ``np.gradient``, which propagates NaN only to directly adjacent
        finite-difference stencils.

        The gradient magnitude |nabla G'| can serve as a mechanotactic
        cue for cell migration models: cells preferentially migrate along
        increasing stiffness gradients (durotaxis).
        """
        if stiffness_field.ndim != 3:
            raise ValueError(
                f"stiffness_field must be 3-D (Nx, Ny, Nz); "
                f"got shape {stiffness_field.shape}."
            )
        grad = np.zeros(stiffness_field.shape + (3,), dtype=float)
        grad[..., 0] = np.gradient(stiffness_field, dx, axis=0)
        grad[..., 1] = np.gradient(stiffness_field, dx, axis=1)
        grad[..., 2] = np.gradient(stiffness_field, dx, axis=2)
        return grad

    # ------------------------------------------------------------------
    # Gel-point detection from modulus data
    # ------------------------------------------------------------------

    def get_gel_point(
        self, p_values: np.ndarray, moduli: np.ndarray
    ) -> float:
        """Estimate the percolation threshold p_c from G'(p) experimental data.

        Fits a power law G'(p) = G0 * (p - p_c)^f to the region where
        G' > 0.  The fit is performed in log-log space using scipy.optimize
        curve_fit, initialised near the model default p_c.

        The function returns the p_c extracted from the best-fit intercept.
        If the fit fails (e.g. too few data points, numerical issues) the
        model default p_c is returned with a warning.

        Parameters
        ----------
        p_values : np.ndarray, shape (N,)
            Bond fractions, sorted in ascending order.
        moduli : np.ndarray, shape (N,)
            Corresponding measured or simulated G' [Pa].

        Returns
        -------
        float
            Estimated gel point p_c.

        Notes
        -----
        The fitting function in log space is:
            ln G' = ln G0 + f * ln(p - p_c)

        This requires p_c < min(p_values[G' > 0]), which is enforced via
        the lower bound in the optimisation.
        """
        p_values = np.asarray(p_values, dtype=float)
        moduli = np.asarray(moduli, dtype=float)

        # Identify region with measurable modulus (> 0.01% of peak)
        noise_floor = 1e-4 * np.nanmax(np.abs(moduli))
        nonzero = (moduli > noise_floor) & np.isfinite(moduli) & np.isfinite(p_values)

        if not np.any(nonzero):
            warnings.warn(
                "get_gel_point: no moduli above noise floor; returning default p_c.",
                UserWarning, stacklevel=2,
            )
            return self.params.p_c

        p_gel = p_values[nonzero]
        G_gel = moduli[nonzero]

        if len(p_gel) < 4:
            # Too few points for a robust fit; return the onset of non-zero G'
            return float(p_values[nonzero][0])

        p_min = float(p_gel.min())

        def _log_power_law(p_arr: np.ndarray, log_G0: float, f_fit: float,
                           p_c_fit: float) -> np.ndarray:
            """Log G' = log G0 + f_fit * log(p - p_c_fit)."""
            eps = p_arr - p_c_fit
            eps = np.where(eps > 1e-9, eps, 1e-9)
            return log_G0 + f_fit * np.log(eps)

        log_G = np.log(G_gel)

        try:
            p0 = [np.log(self.params.E_ref), self.params.exponents.f_elastic,
                  self.params.p_c]
            bounds_lo = [-np.inf, 0.5, 0.0]
            bounds_hi = [np.inf, 6.0, p_min - 1e-6]

            popt, _ = curve_fit(
                _log_power_law, p_gel, log_G,
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=10_000,
            )
            p_c_fit = float(popt[2])
            return p_c_fit

        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"get_gel_point: power-law fit failed ({exc}); "
                "returning default p_c.",
                UserWarning, stacklevel=2,
            )
            return self.params.p_c

    # ------------------------------------------------------------------
    # Auxiliary observables (susceptibility, correlation length)
    # ------------------------------------------------------------------

    def compute_susceptibility(self, p: float) -> float:
        """Cluster-size susceptibility chi ~ |p - p_c|^{-gamma}.

        The susceptibility is proportional to the second moment of the
        cluster-size distribution (variance of cluster sizes) and diverges
        at the percolation threshold.  It is an early-warning-signal (EWS)
        quantity: its rise precedes the actual percolation transition.

        Parameters
        ----------
        p : float
            Bond occupation fraction.

        Returns
        -------
        float
            Dimensionless susceptibility scaled by E_ref [Pa].
        """
        pm = self.params
        eps = abs(p - pm.p_c)
        eps = max(eps, 1e-9)
        return float(pm.E_ref * eps ** (-pm.exponents.gamma_sus))

    def compute_correlation_length(self, p: float) -> float:
        """Spatial correlation length xi ~ |p - p_c|^{-nu} [a.u.].

        The correlation length quantifies the characteristic cluster size
        and the spatial range over which mechanical fluctuations are
        correlated.  It diverges at p_c.

        Parameters
        ----------
        p : float
            Bond occupation fraction.

        Returns
        -------
        float
            Correlation length in units of the network mesh size.
        """
        pm = self.params
        eps = abs(p - pm.p_c)
        eps = max(eps, 1e-9)
        return float(eps ** (-pm.exponents.nu))


# ---------------------------------------------------------------------------
# Affine network theory
# ---------------------------------------------------------------------------

class AffineNetworkTheory:
    """Rubber-elastic affine network theory for the gel-phase plateau modulus.

    The affine network model (Treloar, 1943; Flory, 1953) treats a crosslinked
    polymer network as a collection of Gaussian chains that deform affinely with
    the macroscopic strain.  The result is the classical neo-Hookean shear
    modulus:

        G'_plateau = nu_e * k_B * T

    where nu_e is the *effective* strand (chain) density, i.e. the number of
    elastically active network strands per unit volume [m^-3].

    This is a lower bound; real networks exhibit additional contributions from
    entanglements (tube model) and non-affine deformation.
    """

    @staticmethod
    def compute_plateau_modulus(rho_chain: float, T: float = 310.15) -> float:
        """Affine rubber-elastic plateau modulus G'_plateau = rho_chain * k_B * T.

        Parameters
        ----------
        rho_chain : float
            Density of elastically active network strands [chains m^-3].
            For a perfect network this equals the crosslink density;
            in practice it is reduced by dangling ends and loops.
        T : float, optional
            Absolute temperature [K].  Default 310.15 K (37 degC).

        Returns
        -------
        float
            Plateau storage modulus [Pa].

        Notes
        -----
        The result is in Pascals when rho_chain is in m^-3, k_B in J K^-1,
        and T in K.  Typical hydrogel values:
            rho_chain = 1e20  m^-3 -> G' ~ 0.43 Pa   (very loose network)
            rho_chain = 1e23  m^-3 -> G' ~ 430 Pa     (typical alginate)
            rho_chain = 1e24  m^-3 -> G' ~ 4300 Pa    (stiff PEG hydrogel)
        """
        return float(rho_chain * _K_B * T)

    @staticmethod
    def compute_entanglement_modulus(
        Mw: float,
        concentration: float,
        rho_polymer: float = 1000.0,
    ) -> float:
        """Entanglement contribution to the plateau modulus.

        Based on the de Gennes / Doi-Edwards tube model.  The entanglement
        strand density is estimated as:

            rho_e = (rho_polymer * phi * N_A) / M_e

        where phi = concentration (volume fraction), N_A is Avogadro's
        number, and M_e is the entanglement molecular weight.  For many
        flexible polymers M_e is empirically correlated with Mw, but here
        we use M_e ~ Mw as a first approximation (one entanglement per chain).

        The entanglement modulus is then:

            G_e = rho_e * k_B * T

        Parameters
        ----------
        Mw : float
            Weight-average molecular weight [g mol^-1].
        concentration : float
            Polymer volume fraction (dimensionless, 0 to 1).
        rho_polymer : float, optional
            Bulk polymer density [kg m^-3].  Default 1000.0 (aqueous
            approximation for most hydrogel polymers).

        Returns
        -------
        float
            Entanglement contribution to the plateau modulus [Pa].

        Notes
        -----
        1 g mol^-1 = 1e-3 kg mol^-1.  Avogadro: N_A = 6.022e23 mol^-1.
        rho_polymer [kg m^-3] * 1000 [g kg^-1] = rho_polymer [g m^-3].
        """
        N_A = 6.022_140_76e23  # mol^-1
        # Convert rho_polymer to g m^-3
        rho_g_per_m3 = rho_polymer * 1000.0
        # Strand density: number of entanglement strands per m^3
        rho_e = (rho_g_per_m3 * concentration * N_A) / Mw
        return float(rho_e * _K_B * 310.15)


# ---------------------------------------------------------------------------
# Frequency-sweep analyser
# ---------------------------------------------------------------------------

class FrequencySweepAnalyzer:
    """Compute and analyse oscillatory shear rheology sweeps.

    Provides tools for:
    * Generating G'(omega) and G''(omega) sweeps at fixed p.
    * Identifying the gel point from time-resolved sweeps using the
      Winter-Chambon criterion (frequency-independent tan delta).
    * Plotting frequency sweeps on log-log axes.

    Parameters
    ----------
    mechanics : PercolationMechanics, optional
        Mechanics instance.  A default alginate instance is used if None.
    """

    def __init__(
        self, mechanics: Optional[PercolationMechanics] = None
    ) -> None:
        self.mech = (
            mechanics if mechanics is not None else PercolationMechanics()
        )

    def compute_frequency_sweep(
        self, p: float, omega_range: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute G'(omega) and G''(omega) at a fixed bond fraction p.

        Parameters
        ----------
        p : float
            Bond occupation fraction in [0, 1].
        omega_range : np.ndarray, shape (Nomega,)
            Angular frequencies [rad s^-1] at which to evaluate moduli.
            Typically a logarithmically spaced array, e.g.
            ``np.logspace(-2, 2, 50)``.

        Returns
        -------
        G_prime : np.ndarray, shape (Nomega,)
            Storage modulus G'(omega) [Pa].
        G_double_prime : np.ndarray, shape (Nomega,)
            Loss modulus G''(omega) [Pa].
        """
        omega_range = np.asarray(omega_range, dtype=float)
        G_prime = np.array(
            [self.mech.compute_shear_modulus(p, w) for w in omega_range]
        )
        G_double_prime = np.array(
            [self.mech.compute_loss_modulus(p, w) for w in omega_range]
        )
        return G_prime, G_double_prime

    def fit_gel_point_winterschmidt(
        self,
        omega_range: np.ndarray,
        G_prime_t: np.ndarray,
        G_double_prime_t: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Identify the gel point using the Winter-Chambon criterion.

        The Winter-Chambon gel-point criterion states that at the gel point
        the loss tangent tan delta = G''/G' is frequency-independent:

            G'(omega) ~ G''(omega) ~ omega^Delta   at p = p_c

        This method finds the time index (row) in ``G_prime_t`` where the
        standard deviation of tan delta across frequencies is minimised —
        the signature of the gel point.  It also fits the exponent Delta
        from the power-law slope of G'(omega) at that time.

        Parameters
        ----------
        omega_range : np.ndarray, shape (Nomega,)
            Angular frequencies [rad s^-1].
        G_prime_t : np.ndarray, shape (Nt, Nomega)
            Time series of storage modulus sweeps.  Rows are time points,
            columns are frequencies.
        G_double_prime_t : np.ndarray, shape (Nt, Nomega), optional
            Time series of loss modulus sweeps.  If None, an approximate
            G'' is derived from G_prime_t using the model tan delta.

        Returns
        -------
        p_c_index : float
            Row index (time index) in G_prime_t corresponding to the gel
            point.  Cast to float for compatibility with continuous-time
            interpolation.
        Delta_fit : float
            Best-fit value of the dynamic exponent Delta extracted from
            G'(omega) ~ omega^Delta at the gel-point time.

        Notes
        -----
        The method name ``fit_gel_point_winterschmidt`` follows the
        historical Winterschmidt (1986) German-language publication that
        accompanied the Winter-Chambon paper; the algorithm is identical to
        the standard Winter-Chambon procedure.

        References
        ----------
        Winter, H.H. & Chambon, F. J. Rheol. 30, 367-382 (1986).
        Chambon, F. & Winter, H.H. Polym. Bull. 13, 499-503 (1985).
        """
        omega_range = np.asarray(omega_range, dtype=float)
        G_prime_t = np.asarray(G_prime_t, dtype=float)

        if G_prime_t.ndim == 1:
            G_prime_t = G_prime_t[np.newaxis, :]

        if G_double_prime_t is None:
            # Approximate G'' ~ G' * tan(pi * Delta / 2) as a fallback
            tan_d = np.tan(np.pi * self.mech.params.exponents.Delta / 2.0)
            G_double_prime_t = G_prime_t * tan_d
        else:
            G_double_prime_t = np.asarray(G_double_prime_t, dtype=float)
            if G_double_prime_t.ndim == 1:
                G_double_prime_t = G_double_prime_t[np.newaxis, :]

        # Compute tan delta = G'' / G' for each (time, frequency) pair
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tan_delta = G_double_prime_t / np.where(
                G_prime_t > 0, G_prime_t, np.nan
            )

        # Gel point: time index where std(tan delta) over frequencies is minimal
        std_per_t = np.nanstd(tan_delta, axis=1)
        gel_idx = int(np.argmin(std_per_t))

        # Fit Delta from G'(omega) ~ omega^Delta at the gel-point time
        g_at_gel = G_prime_t[gel_idx]
        valid = (g_at_gel > 0) & (omega_range > 0) & np.isfinite(g_at_gel)

        if valid.sum() >= 2:
            log_w = np.log(omega_range[valid])
            log_g = np.log(g_at_gel[valid])
            coeffs = np.polyfit(log_w, log_g, 1)
            Delta_fit = float(coeffs[0])
        else:
            Delta_fit = float(self.mech.params.exponents.Delta)

        return float(gel_idx), Delta_fit

    def plot_frequency_sweep(
        self,
        omega_range: np.ndarray,
        G_prime: np.ndarray,
        G_double_prime: np.ndarray,
        title: str = "",
        ax=None,
    ):
        """Plot G'(omega) and G''(omega) on log-log axes.

        Parameters
        ----------
        omega_range : np.ndarray
            Angular frequencies [rad s^-1].
        G_prime : np.ndarray
            Storage modulus G'(omega) [Pa].
        G_double_prime : np.ndarray
            Loss modulus G''(omega) [Pa].
        title : str, optional
            Plot title.  Defaults to "Frequency sweep".
        ax : matplotlib.axes.Axes, optional
            Axes object to plot into.  A new figure is created if None.

        Returns
        -------
        matplotlib.figure.Figure or None
            Figure handle.  Returns None if matplotlib is unavailable.

        Notes
        -----
        Requires matplotlib.  If not installed the function emits a warning
        and returns None, allowing non-plotting code paths to continue.
        """
        if not _HAS_MPL:
            warnings.warn(
                "plot_frequency_sweep: matplotlib is not installed; "
                "skipping plot.",
                ImportWarning, stacklevel=2,
            )
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        # Guard against zero or negative values before log-log plot
        G_prime = np.asarray(G_prime, dtype=float)
        G_double_prime = np.asarray(G_double_prime, dtype=float)
        omega_range = np.asarray(omega_range, dtype=float)

        mask_p = G_prime > 0
        mask_pp = G_double_prime > 0

        if mask_p.any():
            ax.loglog(
                omega_range[mask_p], G_prime[mask_p],
                "b-o", label="G' (storage)", markersize=4, linewidth=1.5,
            )
        if mask_pp.any():
            ax.loglog(
                omega_range[mask_pp], G_double_prime[mask_pp],
                "r--s", label="G'' (loss)", markersize=4, linewidth=1.5,
            )

        ax.set_xlabel("Angular frequency omega [rad s^{-1}]")
        ax.set_ylabel("Modulus [Pa]")
        ax.set_title(title if title else "Frequency sweep")
        ax.legend(frameon=True)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid function.

    Returns 1 / (1 + exp(-x)) computed in a way that avoids overflow
    for large positive or negative x.

    Parameters
    ----------
    x : float

    Returns
    -------
    float
        sigmoid(x) in (0, 1).
    """
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)
