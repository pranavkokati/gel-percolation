"""Module 4 — Agent-Based Fibroblast Invasion with MMP-Durotaxis Feedback.

Fibroblasts are modelled as continuous-space, continuous-time agents that:
  1. Migrate via standard durotaxis: cells follow the local stiffness
     gradient ∇E_eff toward the intact (stiff) scaffold ahead of them.
     MMP secretion softens already-visited matrix, so the gradient
     persistently points toward the remaining stiff front — fibroblasts
     follow the stiffness gradient toward the intact scaffold (standard
     durotaxis); MMP secretion accelerates the degradation front, creating
     a dynamic gradient that cells continue to chase.
  2. Secrete MMP at high rate when on stiff substrate (E > E_threshold),
     creating a positive feedback that accelerates degradation.
  3. Deposit collagen proportional to local stiffness (mechanostimulated
     ECM synthesis), building the competing collagen percolation network.

The combined MMP+collagen feedback creates the "percolation handoff" dynamics
analysed in Module 5.

Key physics
-----------
- Durotaxis (standard): v = mu_durotaxis * grad(E) / |grad(E)|
  Cells sense the stiffness gradient and move toward stiffer regions
  (the intact scaffold ahead). MMP secretion softens the matrix behind
  and beside the cell, so the gradient always points toward the stiff front.

- MMP secretion switch: rate = k_MMP_high when E_local > E_threshold,
  k_MMP_low otherwise.  This creates positive feedback: cells at the stiff
  invasion front secrete MMP vigorously, softening ahead, which draws them
  forward.

- Collagen synthesis: r_col * (E_local / E_ref) * dt — mechanostimulated
  production proportional to local substrate stiffness.

- PDGF chemotaxis: v += mu_chemotaxis * grad([PDGF]) — wound-derived
  chemoattractant gradient drives directed migration from wound edge inward.

Interfaces
----------
Module 1 (network_model.HydrogelNetwork): degrade_step(mmp_field, dt)
Module 2 (mechanical_properties.PercolationMechanics): compute_stiffness_field
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

logger = logging.getLogger("gel_percolation.cell_invasion")

__all__ = [
    "CellParams",
    "SimParams",
    "SimulationState",
    "Fibroblast",
    "CollagenNetwork",
    "MMPDiffusionSolver",
    "PDGFChemokineSolver",
    "WoundHealingSimulation",
]


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CellParams:
    """Fibroblast migration and signalling parameters.

    Attributes
    ----------
    mu_durotaxis : float
        Durotaxis speed coefficient [µm s⁻¹ per (Pa µm⁻¹)⁻¹].
        Cells move along unit stiffness gradient at this speed.
    mu_chemotaxis : float
        PDGF chemotaxis speed coefficient [µm s⁻¹ per (nM µm⁻¹)⁻¹].
    D_rand : float
        Diffusion coefficient for random motility [µm² s⁻¹].
    E_threshold : float
        Stiffness threshold [Pa] for MMP secretion switching.
    k_MMP_high : float
        MMP secretion rate on stiff substrate (E > E_threshold) [nM s⁻¹ cell⁻¹].
    k_MMP_low : float
        MMP secretion rate on soft substrate (E <= E_threshold) [nM s⁻¹ cell⁻¹].
    r_col : float
        Collagen fibre deposition rate coefficient [µm s⁻¹ per (E/E_ref)].
    E_ref : float
        Reference stiffness for normalisation [Pa].
    cell_radius : float
        Effective cell radius for exclusion [µm].
    """

    mu_durotaxis: float = 0.5
    mu_chemotaxis: float = 0.3
    D_rand: float = 0.1
    E_threshold: float = 500.0
    k_MMP_high: float = 1e-3
    k_MMP_low: float = 1e-5
    r_col: float = 0.01
    E_ref: float = 1000.0
    cell_radius: float = 10.0


@dataclass
class SimParams:
    """Simulation time-stepping and geometry parameters.

    Attributes
    ----------
    dt : float
        Timestep [s].
    n_steps : int
        Total number of simulation steps.
    record_interval : int
        Record state every this many steps.
    box_size : float
        Cubic simulation domain side length [µm].
    grid_resolution : int
        Number of grid points per spatial axis for field discretisation.
    n_cells : int
        Initial number of fibroblasts to seed.
    random_seed : int or None
        Seed for reproducibility.
    """

    dt: float = 1.0
    n_steps: int = 3600
    record_interval: int = 10
    box_size: float = 50.0
    grid_resolution: int = 20
    n_cells: int = 20
    random_seed: Optional[int] = 42


@dataclass
class SimulationState:
    """Snapshot of full simulation state at one time point.

    Attributes
    ----------
    time : float
        Simulation time [s].
    step : int
        Step index.
    cell_positions : np.ndarray, shape (N_cells, 3)
        3-D positions of all fibroblasts [µm].
    mmp_field : np.ndarray, shape (G, G, G)
        MMP concentration field [nM].
    pdgf_field : np.ndarray, shape (G, G, G)
        PDGF concentration field [nM].
    stiffness_field : np.ndarray, shape (G, G, G)
        Effective elastic modulus G'(x) [Pa].
    collagen_p_inf : float
        Percolation order parameter of the growing collagen network.
    hydrogel_p_inf : float
        Percolation order parameter of the degrading hydrogel network.
    invasion_depth : float
        95th percentile of cell x-coordinates [µm].
    n_collagen_fibers : int
        Total number of collagen fibre segments deposited.
    """

    time: float
    step: int
    cell_positions: np.ndarray
    mmp_field: np.ndarray
    pdgf_field: np.ndarray
    stiffness_field: np.ndarray
    collagen_p_inf: float
    hydrogel_p_inf: float
    invasion_depth: float
    n_collagen_fibers: int


# ---------------------------------------------------------------------------
# Fibroblast agent
# ---------------------------------------------------------------------------


class Fibroblast:
    """A single fibroblast agent undergoing MMP-mediated stiffness-guided migration.

    The cell moves via a superposition of three forces:
      1. Durotaxis (standard): unit stiffness gradient times mu_durotaxis.
         Cells move TOWARD high stiffness (intact scaffold); MMP softens
         already-visited matrix, so the gradient persistently points toward
         the stiff front the cell is chasing.
      2. PDGF chemotaxis: gradient of PDGF concentration times mu_chemotaxis.
      3. Stochastic noise: overdamped Langevin term with amplitude
         sqrt(2 * D_rand / dt) per coordinate (velocity units).

    Positive feedback mechanism:
        - Cell arrives at stiff region (E > E_threshold)
        - Secretes MMP at high rate (k_MMP_high) → matrix softens ahead
        - Stiffness gradient now points toward the stiff front
        - Cell chases gradient forward
        - Cycle repeats → accelerating invasion

    Parameters
    ----------
    position : np.ndarray, shape (3,)
        Initial 3-D position [µm].
    cell_id : int
        Unique cell identifier used to seed the cell's RNG.
    params : CellParams
        Physical parameters.
    rng : np.random.Generator, optional
        External RNG; a new one seeded by cell_id is created if None.
    """

    def __init__(
        self,
        position: np.ndarray,
        cell_id: int,
        params: CellParams,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.position: np.ndarray = position.copy().astype(float)
        self.cell_id: int = cell_id
        self.params: CellParams = params
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng(cell_id)
        )
        self.velocity: np.ndarray = np.zeros(3, dtype=float)
        self.mmp_secretion_rate: float = params.k_MMP_low  # current rate [nM s⁻¹]
        self.collagen_deposited: float = 0.0                # cumulative [µm]
        self._mmp_total_secreted: float = 0.0
        self._E_local: float = 0.0

    # ------------------------------------------------------------------
    # Private field utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _trilinear_interpolate(
        field: np.ndarray,
        position: np.ndarray,
        dx: float,
    ) -> float:
        """Trilinearly interpolate a 3-D field at an arbitrary position.

        Parameters
        ----------
        field : np.ndarray, shape (nx, ny, nz)
        position : np.ndarray, shape (3,)
        dx : float
            Isotropic grid spacing [µm].

        Returns
        -------
        float
            Interpolated field value.
        """
        nx_, ny_, nz_ = field.shape
        frac = position / dx
        frac = np.clip(frac, 0.0, np.array([nx_ - 1, ny_ - 1, nz_ - 1]) - 1e-9)

        i0, j0, k0 = frac.astype(int)
        i1 = min(i0 + 1, nx_ - 1)
        j1 = min(j0 + 1, ny_ - 1)
        k1 = min(k0 + 1, nz_ - 1)

        tx, ty, tz = frac[0] - i0, frac[1] - j0, frac[2] - k0

        return float(
            field[i0, j0, k0] * (1 - tx) * (1 - ty) * (1 - tz)
            + field[i1, j0, k0] * tx       * (1 - ty) * (1 - tz)
            + field[i0, j1, k0] * (1 - tx) * ty       * (1 - tz)
            + field[i1, j1, k0] * tx       * ty       * (1 - tz)
            + field[i0, j0, k1] * (1 - tx) * (1 - ty) * tz
            + field[i1, j0, k1] * tx       * (1 - ty) * tz
            + field[i0, j1, k1] * (1 - tx) * ty       * tz
            + field[i1, j1, k1] * tx       * ty       * tz
        )

    @staticmethod
    def _central_difference_gradient(
        field: np.ndarray,
        position: np.ndarray,
        dx: float,
    ) -> np.ndarray:
        """Compute the gradient of a 3-D field at a position using central differences.

        Interior positions use true central differences; positions near the
        boundary are shifted to the nearest valid stencil.

        Parameters
        ----------
        field : np.ndarray, shape (nx, ny, nz)
        position : np.ndarray, shape (3,)
        dx : float

        Returns
        -------
        np.ndarray, shape (3,)
            Gradient vector [field_units / µm].
        """
        nx_, ny_, nz_ = field.shape
        idx = (position / dx).astype(int)
        # Clamp so stencil [idx-1, idx+1] is always valid
        idx[0] = int(np.clip(idx[0], 1, nx_ - 2))
        idx[1] = int(np.clip(idx[1], 1, ny_ - 2))
        idx[2] = int(np.clip(idx[2], 1, nz_ - 2))

        grad = np.empty(3, dtype=float)
        grad[0] = (field[idx[0] + 1, idx[1], idx[2]]
                   - field[idx[0] - 1, idx[1], idx[2]]) / (2.0 * dx)
        grad[1] = (field[idx[0], idx[1] + 1, idx[2]]
                   - field[idx[0], idx[1] - 1, idx[2]]) / (2.0 * dx)
        grad[2] = (field[idx[0], idx[1], idx[2] + 1]
                   - field[idx[0], idx[1], idx[2] - 1]) / (2.0 * dx)
        return grad

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_velocity(
        self,
        stiffness_field: np.ndarray,
        pdgf_field: np.ndarray,
        grid_coords: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute cell velocity for the current timestep.

        Velocity model (all terms in µm s⁻¹):

            v = mu_durotaxis * grad_E / |grad_E|
              + mu_chemotaxis * grad_PDGF
              + noise

        where noise = sqrt(2 * D_rand * dt) / dt * N(0, 1)^3, i.e. a white
        noise term drawn from an overdamped Langevin equation giving a mean
        squared displacement of 2 * D_rand * dt per coordinate per step.

        Parameters
        ----------
        stiffness_field : np.ndarray, shape (G, G, G)
            Local elastic modulus G'(x) [Pa].
        pdgf_field : np.ndarray, shape (G, G, G)
            PDGF concentration field [nM].
        grid_coords : np.ndarray, shape (G, G, G, 3)
            3-D coordinates of each grid voxel centre [µm].
        dt : float
            Timestep [s].  Used to scale the noise amplitude.

        Returns
        -------
        np.ndarray, shape (3,)
            Velocity vector [µm s⁻¹].
        """
        p = self.params
        # Grid spacing inferred from grid_coords
        dx = float(grid_coords[1, 0, 0, 0] - grid_coords[0, 0, 0, 0])

        # Update cached local stiffness via trilinear interpolation
        self._E_local = self._trilinear_interpolate(stiffness_field, self.position, dx)

        # Update secretion rate based on current stiffness
        self.mmp_secretion_rate = (
            p.k_MMP_high if self._E_local > p.E_threshold else p.k_MMP_low
        )

        # --- Durotaxis term ---
        # Stiffness-guided migration toward intact scaffold (standard durotaxis):
        # cells move along the stiffness gradient direction toward stiffer regions.
        # MMP secretion softens already-visited matrix, so the gradient
        # persistently points toward the remaining stiff scaffold ahead.
        grad_E = self._central_difference_gradient(stiffness_field, self.position, dx)
        grad_E_mag = np.linalg.norm(grad_E)
        if grad_E_mag > 1e-10:
            v_durotaxis = p.mu_durotaxis * grad_E / grad_E_mag
        else:
            v_durotaxis = np.zeros(3, dtype=float)

        # --- PDGF chemotaxis term ---
        grad_pdgf = self._central_difference_gradient(pdgf_field, self.position, dx)
        v_chemo = p.mu_chemotaxis * grad_pdgf

        # --- Random motility (overdamped Langevin noise) ---
        # <v_noise^2> = 2 D_rand / dt  so that  <(v_noise * dt)^2> = 2 D_rand dt
        noise_amp = np.sqrt(2.0 * p.D_rand * dt) / max(dt, 1e-15)
        v_noise = noise_amp * self.rng.standard_normal(3)

        return v_durotaxis + v_chemo + v_noise

    def secrete_mmp(self, E_local: float, dt: float) -> float:
        """Compute the amount of MMP secreted in one timestep.

        Implements the positive-feedback stiffness switch:
          - E_local > E_threshold  →  high secretion rate (k_MMP_high)
          - E_local <= E_threshold →  basal secretion rate (k_MMP_low)

        Parameters
        ----------
        E_local : float
            Local elastic modulus at the cell's position [Pa].
        dt : float
            Timestep [s].

        Returns
        -------
        float
            MMP amount secreted in this step [nM].
        """
        p = self.params
        self._E_local = float(E_local)
        rate = p.k_MMP_high if E_local > p.E_threshold else p.k_MMP_low
        self.mmp_secretion_rate = rate
        secreted = rate * dt
        self._mmp_total_secreted += secreted
        return secreted

    def deposit_collagen(self, E_local: float, dt: float) -> float:
        """Compute the collagen fibre length deposited in one timestep.

        Mechanostimulated collagen production scales linearly with the local
        substrate stiffness normalised by E_ref:

            delta_L = r_col * (E_local / E_ref) * dt

        Parameters
        ----------
        E_local : float
            Local elastic modulus [Pa].
        dt : float
            Timestep [s].

        Returns
        -------
        float
            Collagen fibre length deposited [µm].
        """
        p = self.params
        deposition = p.r_col * (max(E_local, 0.0) / p.E_ref) * dt
        self.collagen_deposited += deposition
        return deposition

    def step(
        self,
        stiffness_field: np.ndarray,
        pdgf_field: np.ndarray,
        mmp_field: np.ndarray,
        grid_coords: np.ndarray,
        dt: float,
    ) -> None:
        """Advance the fibroblast agent by one timestep.

        Operations:
          1. Compute velocity (durotaxis + chemotaxis + noise).
          2. Update position (Euler integration, with reflective boundary).
          3. Secrete MMP.
          4. Deposit collagen.

        Parameters
        ----------
        stiffness_field : np.ndarray, shape (G, G, G)
        pdgf_field : np.ndarray, shape (G, G, G)
        mmp_field : np.ndarray, shape (G, G, G)
            Current MMP field (used for bookkeeping; secretion is applied
            externally via the diffusion solver).
        grid_coords : np.ndarray, shape (G, G, G, 3)
        dt : float
        """
        box_size = float(grid_coords[-1, 0, 0, 0] + (grid_coords[1, 0, 0, 0] - grid_coords[0, 0, 0, 0]))

        # Compute and store velocity
        v = self.compute_velocity(stiffness_field, pdgf_field, grid_coords, dt)
        self.velocity = v

        # Euler integration
        self.position = self.position + v * dt

        # Reflective boundary conditions: bounce back from domain walls
        self.position = np.clip(self.position, 0.0, box_size - 1e-9)

        # Secretion & deposition use the current (post-move) local stiffness
        E_here = self._E_local  # already updated in compute_velocity
        self.secrete_mmp(E_here, dt)
        self.deposit_collagen(E_here, dt)


# ---------------------------------------------------------------------------
# Collagen network — second percolation graph
# ---------------------------------------------------------------------------


class CollagenNetwork:
    """Growing collagen fibre network deposited by fibroblasts.

    The network is represented as an unordered list of fibre segments, each
    characterised by a position, orientation unit vector, and length.
    Percolation is computed by constructing a proximity graph in which two
    fibres are connected if their centre-to-centre distance is less than
    twice the mean fibre length (fibres can overlap / cross-link), then
    computing the giant connected component fraction.

    Parameters
    ----------
    box_size : float
        Cubic domain side length [µm].
    resolution : int
        Grid resolution used for the density field [voxels per side].
    """

    def __init__(self, box_size: float, resolution: int = 20) -> None:
        self.box_size = float(box_size)
        self.resolution = int(resolution)
        self._fiber_positions: List[np.ndarray] = []
        self._fiber_lengths: List[float] = []
        self._fiber_orientations: List[np.ndarray] = []
        # Graph cache: avoid rebuilding the KDTree/graph when no new fibers were added
        self._cached_graph: Optional[nx.Graph] = None
        self._cached_n_fibers: int = 0

    # ------------------------------------------------------------------
    # Fibre management
    # ------------------------------------------------------------------

    def add_fiber(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        length: float,
    ) -> None:
        """Register a new collagen fibre segment.

        Parameters
        ----------
        position : np.ndarray, shape (3,)
            Centre position of the fibre [µm].
        orientation : np.ndarray, shape (3,)
            Direction vector of the fibre (need not be normalised).
        length : float
            Fibre length [µm].
        """
        norm = np.linalg.norm(orientation)
        if norm < 1e-10:
            orientation = np.array([1.0, 0.0, 0.0])
        else:
            orientation = orientation / norm
        self._fiber_positions.append(position.copy())
        self._fiber_lengths.append(float(length))
        self._fiber_orientations.append(orientation)

    def get_n_fibers(self) -> int:
        """Return the number of fibre segments currently registered."""
        return len(self._fiber_positions)

    def get_fiber_positions(self) -> np.ndarray:
        """Return fibre centre positions as an array.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Returns an empty (0, 3) array if no fibres have been deposited.
        """
        if not self._fiber_positions:
            return np.empty((0, 3), dtype=float)
        return np.array(self._fiber_positions, dtype=float)

    # ------------------------------------------------------------------
    # Graph construction for percolation analysis
    # ------------------------------------------------------------------

    def build_graph(self) -> nx.Graph:
        """Convert fibre list to a NetworkX graph for percolation analysis.

        Two fibres are connected by an edge if their centre-to-centre
        distance is less than ``r_connect = max(2 * mean_length, 5.0) µm``.
        This threshold accounts for fibre overlap and physical cross-linking.

        Returns
        -------
        networkx.Graph
            Nodes are fibre indices; edges represent proximity connections.
            Returns an empty Graph if fewer than 2 fibres exist.
        """
        n = len(self._fiber_positions)
        G = nx.Graph()
        if n < 2:
            if n == 1:
                G.add_node(0)
            return G

        pos_arr = np.array(self._fiber_positions, dtype=float)
        lengths = np.array(self._fiber_lengths, dtype=float)
        mean_length = float(np.mean(lengths)) if len(lengths) > 0 else 1.0
        r_connect = max(2.0 * mean_length, 5.0)

        tree = KDTree(pos_arr)
        pairs = tree.query_pairs(r_connect, output_type="ndarray")

        G.add_nodes_from(range(n))
        if len(pairs) > 0:
            G.add_edges_from(pairs.tolist())
        return G

    def get_percolation_order_parameter(self) -> float:
        """Estimate P_inf = |giant component| / N for the collagen network.

        Uses the proximity graph built by :py:meth:`build_graph`.  The graph
        is cached: if no new fibres have been added since the last call the
        previous graph is reused, avoiding an O(N²) KDTree rebuild every step.

        Returns
        -------
        float
            Percolation order parameter in [0, 1].
            Returns 0.0 if fewer than 2 fibres exist.
        """
        n = len(self._fiber_positions)
        if n < 2:
            return 0.0
        try:
            # Rebuild graph only when new fibers have been added since last call
            if n != self._cached_n_fibers or self._cached_graph is None:
                self._cached_graph = self.build_graph()
                self._cached_n_fibers = n
            G = self._cached_graph
            if G.number_of_nodes() == 0:
                return 0.0
            components = list(nx.connected_components(G))
            largest = max(components, key=len)
            return float(len(largest)) / float(G.number_of_nodes())
        except Exception as exc:
            logger.warning("CollagenNetwork.get_percolation_order_parameter failed: %s", exc)
            return 0.0

    def get_local_density(self, x: float, y: float, z: float,
                          radius: float = 5.0) -> float:
        """Compute local collagen density (fibres per unit volume) near (x, y, z).

        Parameters
        ----------
        x, y, z : float
            Query point coordinates [µm].
        radius : float
            Neighbourhood radius [µm].

        Returns
        -------
        float
            Number of fibres per µm³ within the sphere of given radius.
        """
        if not self._fiber_positions:
            return 0.0
        pos_arr = np.array(self._fiber_positions, dtype=float)
        query = np.array([x, y, z], dtype=float)
        dists = np.linalg.norm(pos_arr - query, axis=1)
        n_nearby = int(np.sum(dists < radius))
        volume = (4.0 / 3.0) * np.pi * radius ** 3
        return float(n_nearby) / volume

    def get_density_field(self, resolution: Optional[int] = None) -> np.ndarray:
        """Compute a voxelised collagen density field.

        Each fibre contributes its length to the voxel containing its centre.

        Parameters
        ----------
        resolution : int, optional
            Grid resolution (voxels per side).  Defaults to ``self.resolution``.

        Returns
        -------
        np.ndarray, shape (res, res, res)
            Collagen length density [µm per voxel].
        """
        res = int(resolution) if resolution is not None else self.resolution
        field = np.zeros((res, res, res), dtype=float)
        if not self._fiber_positions:
            return field
        dx = self.box_size / res
        for pos, length in zip(self._fiber_positions, self._fiber_lengths):
            idx = np.clip((pos / dx).astype(int), 0, res - 1)
            field[idx[0], idx[1], idx[2]] += length
        return field


# ---------------------------------------------------------------------------
# MMP reaction-diffusion solver
# ---------------------------------------------------------------------------


class MMPDiffusionSolver:
    """Explicit finite-difference solver for MMP reaction-diffusion.

    Solves the PDE:

        d[MMP]/dt = D_mmp * laplacian([MMP])
                  - k_degradation * [MMP]
                  + sum_cells( k_secretion_i * delta(x - x_cell_i) )

    using an explicit (forward Euler) scheme with automatic sub-stepping to
    satisfy the von Neumann stability criterion:

        dt_sub < dx^2 / (6 * D_mmp)

    Parameters
    ----------
    grid_shape : tuple of int, (nx, ny, nz)
        Shape of the concentration field grid.
    dx : float
        Isotropic grid spacing [µm].
    D_mmp : float
        MMP diffusion coefficient [µm² s⁻¹].
    k_degradation : float
        First-order MMP degradation rate [s⁻¹].
    params : CellParams, optional
        Cell parameters (used for secretion rate defaults).
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        dx: float,
        D_mmp: float = 1.0,
        k_degradation: float = 1e-3,
        params: Optional[CellParams] = None,
    ) -> None:
        self.grid_shape = tuple(grid_shape)
        self.dx = float(dx)
        self.D_mmp = float(D_mmp)
        self.k_degradation = float(k_degradation)
        self.params = params if params is not None else CellParams()

        # Stability threshold: explicit scheme is stable when r = D dt / dx^2 <= 1/6
        if D_mmp > 0.0:
            self._dt_stability = self.dx ** 2 / (6.0 * self.D_mmp)
        else:
            self._dt_stability = float("inf")

        logger.debug(
            "MMPDiffusionSolver: dx=%.2f µm, D_mmp=%.2e µm²/s, "
            "k_deg=%.2e s⁻¹, dt_stability=%.4f s",
            self.dx, self.D_mmp, self.k_degradation, self._dt_stability,
        )

    def step(
        self,
        mmp_field: np.ndarray,
        cell_positions: np.ndarray,
        secretion_rates: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance the MMP concentration field by dt seconds.

        Parameters
        ----------
        mmp_field : np.ndarray, shape grid_shape
            Current MMP concentration field [nM].
        cell_positions : np.ndarray, shape (N_cells, 3)
            Cell positions [µm].
        secretion_rates : np.ndarray, shape (N_cells,)
            MMP secretion rate per cell [nM s⁻¹].  These are distributed
            into the nearest grid voxel as a volumetric source [nM s⁻¹ µm⁻³].
        dt : float
            Target time increment [s].

        Returns
        -------
        np.ndarray, shape grid_shape
            Updated MMP concentration field [nM].
        """
        # Sub-stepping for stability
        n_sub = max(1, int(np.ceil(dt / (self._dt_stability + 1e-15))))
        dt_sub = dt / float(n_sub)

        # Build volumetric source field [nM s⁻¹ µm⁻³]
        source = np.zeros(self.grid_shape, dtype=float)
        voxel_vol = self.dx ** 3
        grid_arr = np.array(self.grid_shape, dtype=int)
        for pos, rate in zip(cell_positions, secretion_rates):
            idx = np.clip((pos / self.dx).astype(int), 0, grid_arr - 1)
            source[idx[0], idx[1], idx[2]] += rate / voxel_vol

        field = mmp_field.copy()
        D = self.D_mmp
        k = self.k_degradation
        dx2 = self.dx ** 2

        for _ in range(n_sub):
            # 3-D Laplacian via roll (zero-flux BCs via zero-padding of boundary terms
            # is approximated by periodic roll here; boundary effects are small)
            lap = (
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
                + np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2)
                - 6.0 * field
            ) / dx2

            field = field + dt_sub * (D * lap - k * field + source)
            # Concentration must be non-negative
            np.maximum(field, 0.0, out=field)

        return field

    def get_mmp_at_position(
        self, mmp_field: np.ndarray, position: np.ndarray
    ) -> float:
        """Return MMP concentration at a position via nearest-neighbour lookup.

        Parameters
        ----------
        mmp_field : np.ndarray, shape grid_shape
        position : np.ndarray, shape (3,)
            Query position [µm].

        Returns
        -------
        float
            Interpolated MMP concentration [nM].
        """
        idx = np.clip(
            (position / self.dx).astype(int),
            0,
            np.array(self.grid_shape) - 1,
        )
        return float(mmp_field[idx[0], idx[1], idx[2]])


# ---------------------------------------------------------------------------
# PDGF chemokine solver
# ---------------------------------------------------------------------------


class PDGFChemokineSolver:
    """Steady-state PDGF gradient emanating from the wound boundary.

    PDGF diffuses significantly faster than MMP:

        D_pdgf = 10 * D_mmp

    The PDGF field is initialised as a static exponential gradient along the
    wound invasion direction (x-axis), decaying away from the wound edge at
    x = 0.  The field is optionally evolved by diffusion with a localised
    source at the wound centre, but the default usage is the static
    pre-computed gradient.

    Parameters
    ----------
    grid_shape : tuple of int
        Grid shape (nx, ny, nz).
    dx : float
        Isotropic grid spacing [µm].
    D_pdgf : float
        PDGF diffusion coefficient [µm² s⁻¹].  Default 10 * D_mmp (10.0).
    source_strength : float
        PDGF source amplitude [nM].
    source_radius : float
        Decay length of the exponential gradient from the wound edge [µm].
    box_size : float
        Domain side length [µm].
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        dx: float,
        D_pdgf: float = 10.0,
        source_strength: float = 1.0,
        source_radius: float = 5.0,
        box_size: float = 50.0,
    ) -> None:
        self.grid_shape = tuple(grid_shape)
        self.dx = float(dx)
        self.D_pdgf = float(D_pdgf)
        self.source_strength = float(source_strength)
        self.source_radius = float(source_radius)
        self.box_size = float(box_size)

        # Stability threshold for the explicit scheme (if time-stepping is needed)
        if D_pdgf > 0.0:
            self._dt_stability = self.dx ** 2 / (6.0 * self.D_pdgf)
        else:
            self._dt_stability = float("inf")

        self._field: Optional[np.ndarray] = None

    def build_steady_state(self) -> np.ndarray:
        """Construct the static PDGF gradient field.

        The gradient is an exponential decay from x = 0 (wound edge) along
        the x-axis, uniform in y and z:

            [PDGF](x) = S * exp(-x / lambda)

        where S = source_strength and lambda = source_radius.

        Returns
        -------
        np.ndarray, shape grid_shape
            PDGF concentration field [nM].
        """
        g = self.grid_shape[0]
        x_coords = np.linspace(0.0, self.box_size, g, endpoint=False) + self.dx / 2.0
        decay = self.source_strength * np.exp(
            -x_coords / max(self.source_radius, 1e-10)
        )
        field = np.zeros(self.grid_shape, dtype=float)
        for i in range(g):
            field[i, :, :] = decay[i]
        self._field = field
        return field

    def step(
        self,
        pdgf_field: np.ndarray,
        cell_positions: np.ndarray,
        secretion_rates: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance the PDGF field by dt seconds (diffusion only; no degradation).

        By default the PDGF field is treated as quasi-static (steady-state
        gradient maintained by wound boundary).  This method provides a full
        diffusion update for cases where the PDGF field should evolve.

        Parameters
        ----------
        pdgf_field : np.ndarray, shape grid_shape
        cell_positions : np.ndarray, shape (N_cells, 3)
        secretion_rates : np.ndarray, shape (N_cells,)
            PDGF secretion rates per cell [nM s⁻¹].
        dt : float

        Returns
        -------
        np.ndarray, shape grid_shape
            Updated PDGF field [nM].
        """
        n_sub = max(1, int(np.ceil(dt / (self._dt_stability + 1e-15))))
        dt_sub = dt / float(n_sub)

        source = np.zeros(self.grid_shape, dtype=float)
        voxel_vol = self.dx ** 3
        grid_arr = np.array(self.grid_shape, dtype=int)
        for pos, rate in zip(cell_positions, secretion_rates):
            idx = np.clip((pos / self.dx).astype(int), 0, grid_arr - 1)
            source[idx[0], idx[1], idx[2]] += rate / voxel_vol

        # Maintain wound-edge source
        source[0, :, :] += self.source_strength / (self.dx ** 2)

        field = pdgf_field.copy()
        D = self.D_pdgf
        dx2 = self.dx ** 2

        for _ in range(n_sub):
            lap = (
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
                + np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2)
                - 6.0 * field
            ) / dx2
            field = field + dt_sub * (D * lap + source)
            np.maximum(field, 0.0, out=field)

        return field

    def get_field(self) -> np.ndarray:
        """Return the current PDGF field, building it if not yet initialised."""
        if self._field is None:
            return self.build_steady_state()
        return self._field

    def get_pdgf_at_position(
        self, pdgf_field: np.ndarray, position: np.ndarray
    ) -> float:
        """Nearest-neighbour lookup of PDGF concentration."""
        idx = np.clip(
            (position / self.dx).astype(int),
            0,
            np.array(self.grid_shape) - 1,
        )
        return float(pdgf_field[idx[0], idx[1], idx[2]])


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------

SimulationHistory = List[SimulationState]


# ---------------------------------------------------------------------------
# Main wound healing simulation
# ---------------------------------------------------------------------------


class WoundHealingSimulation:
    """Orchestrator for coupled hydrogel degradation, cell invasion, and collagen growth.

    Implements the full "percolation inversion" dynamics:

    1. Fibroblasts invade via stiffness-guided migration toward the intact scaffold
       (standard durotaxis); MMP secretion accelerates the degradation front,
       creating a dynamic stiffness gradient that cells continue to chase.
    2. High MMP secretion on stiff substrate creates positive feedback → accelerated
       degradation at the invasion front.
    3. Cells deposit mechanostimulated collagen as the hydrogel degrades.
    4. Two competing percolation order parameters are tracked:
       P_inf_hydrogel (decreasing) and P_inf_collagen (increasing).
    5. The handoff quality Q = dP_col/dt|_{t*} - dP_hyd/dt|_{t*} is the design metric.

    Step ordering per timestep:
      1. Update MMP field (diffusion + cell secretion)
      2. Update PDGF field (static gradient or full diffusion)
      3. Degrade hydrogel network (Module 1 interface)
      4. Recompute stiffness field (Module 2 interface)
      5. Move cells (durotaxis + chemotaxis + noise)
      6. Deposit collagen from cells
      7. Update collagen percolation
      8. Record state (at record_interval)

    Parameters
    ----------
    hydrogel_params : HydrogelParams, optional
        Parameters for HydrogelNetwork (Module 1).  If None and no
        hydrogel_network is supplied, a simple linear decay proxy is used.
    cell_params : CellParams, optional
        Fibroblast parameters.  Defaults to CellParams().
    sim_params : SimParams, optional
        Simulation parameters.  Defaults to SimParams().
    mechanics : callable, optional
        Function (local_p_grid, omega) → np.ndarray of stiffness values.
        Wraps PercolationMechanics.compute_stiffness_field (Module 2).
        If None, stiffness = E_ref * p.
    hydrogel_network : HydrogelNetwork, optional
        Pre-built Module 1 network.  If None and hydrogel_params is given,
        a network is constructed automatically.
    """

    def __init__(
        self,
        hydrogel_params=None,
        cell_params: Optional[CellParams] = None,
        sim_params: Optional[SimParams] = None,
        mechanics=None,
        hydrogel_network=None,
    ) -> None:
        self.cp = cell_params if cell_params is not None else CellParams()
        self.sp = sim_params if sim_params is not None else SimParams()
        self._mechanics = mechanics

        # Module 1 interface: HydrogelNetwork
        if hydrogel_network is not None:
            self._network = hydrogel_network
        elif hydrogel_params is not None:
            try:
                from .network_model import HydrogelNetwork
                self._network = HydrogelNetwork(hydrogel_params, seed=self.sp.random_seed)
                logger.info("Constructed HydrogelNetwork from supplied HydrogelParams.")
            except ImportError:
                logger.warning(
                    "network_model not importable; falling back to proxy stiffness decay."
                )
                self._network = None
        else:
            self._network = None

        # Module 2 interface: PercolationMechanics
        if mechanics is None:
            try:
                from .mechanical_properties import PercolationMechanics
                self._mech_obj = PercolationMechanics()
                self._mechanics = lambda p_grid, omega=1.0: \
                    self._mech_obj.compute_stiffness_field(p_grid, omega)
                logger.info("Loaded PercolationMechanics for stiffness computation.")
            except ImportError:
                self._mech_obj = None
                self._mechanics = None
                logger.warning(
                    "mechanical_properties not importable; "
                    "using stiffness = E_ref * p proxy."
                )

        # RNG
        self._rng = np.random.default_rng(self.sp.random_seed)

        # Grid geometry
        g = self.sp.grid_resolution
        bs = self.sp.box_size
        self._dx = bs / float(g)
        # grid_coords[i, j, k] = (x_i, y_j, z_k) voxel centres
        lin = np.linspace(0.0, bs, g, endpoint=False) + self._dx / 2.0
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        self._grid_coords: np.ndarray = np.stack([gx, gy, gz], axis=-1)  # (g, g, g, 3)
        self._grid_1d = lin
        self._grid_shape = (g, g, g)

        # Initialise concentration fields
        self._mmp_field: np.ndarray = np.zeros(self._grid_shape, dtype=float)

        self._pdgf_solver = PDGFChemokineSolver(
            self._grid_shape,
            self._dx,
            D_pdgf=10.0,
            source_strength=1.0,
            source_radius=max(bs / 10.0, 5.0),
            box_size=bs,
        )
        self._pdgf_field: np.ndarray = self._pdgf_solver.build_steady_state()

        self._mmp_solver = MMPDiffusionSolver(
            self._grid_shape,
            self._dx,
            D_mmp=1.0,
            k_degradation=1e-3,
            params=self.cp,
        )

        # Initial stiffness field
        p_init = 1.0
        if self._network is not None:
            p_init = self._network.get_percolation_order_parameter()
        self._stiffness_field: np.ndarray = np.full(
            self._grid_shape, self.cp.E_ref * p_init, dtype=float
        )

        # Collagen network
        self._collagen = CollagenNetwork(bs, g)

        # Cells and history
        self._cells: List[Fibroblast] = []
        self._history: SimulationHistory = []
        self._step: int = 0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(
        self,
        n_cells: Optional[int] = None,
        seeding_positions: Optional[np.ndarray] = None,
    ) -> None:
        """Seed fibroblasts and record initial state.

        Parameters
        ----------
        n_cells : int, optional
            Number of cells to seed.  Defaults to sp.n_cells.
        seeding_positions : np.ndarray, shape (n_cells, 3), optional
            Explicit starting positions [µm].  If None, cells are placed
            uniformly in the x ∈ [0, 5] µm strip (the wound edge).
        """
        n = int(n_cells) if n_cells is not None else self.sp.n_cells
        bs = self.sp.box_size

        if seeding_positions is not None:
            positions = np.asarray(seeding_positions, dtype=float)[:n]
        else:
            positions = np.column_stack([
                self._rng.uniform(0.0, min(5.0, bs * 0.1), n),
                self._rng.uniform(0.0, bs, n),
                self._rng.uniform(0.0, bs, n),
            ])

        seed_base = self.sp.random_seed if self.sp.random_seed is not None else 0
        self._cells = [
            Fibroblast(
                positions[i],
                i,
                self.cp,
                rng=np.random.default_rng(seed_base + i),
            )
            for i in range(n)
        ]
        self._initialized = True
        logger.info("Seeded %d fibroblasts at wound edge (x ∈ [0, %.1f] µm).", n, positions[:, 0].max())
        self._record_state()

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, dt: Optional[float] = None) -> SimulationState:
        """Advance the simulation by one timestep.

        Parameters
        ----------
        dt : float, optional
            Timestep [s].  Defaults to sp.dt.

        Returns
        -------
        SimulationState
            The most recently recorded state (may be from a previous record
            step if the current step does not land on record_interval).
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before step().")

        dt_ = float(dt) if dt is not None else self.sp.dt

        # --- Step 1: Update MMP field ---
        cell_positions = np.array([c.position for c in self._cells], dtype=float)
        secretion_rates = np.array(
            [c.mmp_secretion_rate for c in self._cells], dtype=float
        )
        self._mmp_field = self._mmp_solver.step(
            self._mmp_field, cell_positions, secretion_rates, dt_
        )

        # --- Step 2: Update PDGF field (static; field maintained by solver) ---
        # The PDGF field is quasi-static (wound-edge source); we leave it unchanged.
        # If dynamic evolution is desired, call self._pdgf_solver.step(...) here.

        # --- Step 3: Degrade hydrogel network (Module 1 interface) ---
        if self._network is not None:
            self._network.degrade_step(self._mmp_field, dt_)

        # --- Step 4: Recompute stiffness field (Module 2 interface) ---
        if self._network is not None:
            _, local_p = self._network.compute_local_p(
                resolution=self.sp.grid_resolution
            )
            local_p_grid = np.nan_to_num(
                local_p.reshape(self._grid_shape), nan=0.0
            )
        else:
            # Proxy: uniform linear decay
            t_total = self.sp.n_steps * self.sp.dt
            t_now = self._step * self.sp.dt
            p_uniform = max(0.0, 1.0 - t_now / (t_total + 1e-15))
            local_p_grid = np.full(self._grid_shape, p_uniform, dtype=float)

        if self._mechanics is not None:
            self._stiffness_field = np.asarray(
                self._mechanics(local_p_grid, 1.0), dtype=float
            )
        else:
            self._stiffness_field = self.cp.E_ref * local_p_grid

        # --- Step 5: Move cells (durotaxis + chemotaxis + noise) ---
        # --- Step 6: Deposit collagen ---
        for cell in self._cells:
            # Cache secretion rate before moving (uses stiffness from step 4)
            cell.step(
                self._stiffness_field,
                self._pdgf_field,
                self._mmp_field,
                self._grid_coords,
                dt_,
            )
            # Deposit collagen fibre if significant length was produced
            col = cell.collagen_deposited
            # Only add a fibre for incremental deposition (per-step amount)
            dx_ = self._dx
            E_local = cell._E_local
            col_increment = self.cp.r_col * (max(E_local, 0.0) / self.cp.E_ref) * dt_
            if col_increment > 1e-4:
                # Random fibre orientation with mild x-bias (invasion direction)
                orient = self._rng.standard_normal(3)
                orient[0] = abs(orient[0])  # bias toward +x
                self._collagen.add_fiber(cell.position, orient, col_increment)

        self._step += 1

        # --- Step 7 & 8: Record state at specified interval ---
        if self._step % self.sp.record_interval == 0:
            self._record_state()

        return self._history[-1]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        n_steps: Optional[int] = None,
        record_interval: Optional[int] = None,
    ) -> SimulationHistory:
        """Run the full simulation.

        Parameters
        ----------
        n_steps : int, optional
            Number of timesteps.  Defaults to sp.n_steps.
        record_interval : int, optional
            Override sp.record_interval for this run.

        Returns
        -------
        SimulationHistory
            List of :py:class:`SimulationState` snapshots recorded at each
            record_interval step.
        """
        if not self._initialized:
            self.initialize()

        total = int(n_steps) if n_steps is not None else self.sp.n_steps
        if record_interval is not None:
            old_interval = self.sp.record_interval
            self.sp.record_interval = int(record_interval)

        logger.info(
            "Running %d steps (dt=%.2f s, T_total=%.1f s, record_interval=%d).",
            total, self.sp.dt, total * self.sp.dt, self.sp.record_interval,
        )

        for i in range(total):
            self.step()
            if (i + 1) % max(total // 10, 1) == 0:
                logger.debug(
                    "  step %d/%d  invasion_depth=%.2f µm  collagen_P_inf=%.4f",
                    i + 1, total,
                    self.get_invasion_depth(),
                    self._collagen.get_percolation_order_parameter(),
                )

        if record_interval is not None:
            self.sp.record_interval = old_interval

        return self._history

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def get_invasion_depth(self, time_idx: int = -1) -> float:
        """Return the 95th percentile of cell x-coordinates as invasion depth.

        Parameters
        ----------
        time_idx : int
            Index into simulation history.  Default -1 (latest).

        Returns
        -------
        float
            95th-percentile cell x-coordinate [µm].
        """
        if not self._history:
            if self._cells:
                xs = np.array([c.position[0] for c in self._cells])
                return float(np.percentile(xs, 95))
            return 0.0
        state = self._history[time_idx]
        if len(state.cell_positions) == 0:
            return 0.0
        return float(np.percentile(state.cell_positions[:, 0], 95))

    def get_history(self) -> SimulationHistory:
        """Return the full simulation history."""
        return self._history

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_snapshot(
        self,
        time_idx: int = -1,
        figsize: Tuple[int, int] = (16, 12),
    ):
        """Four-panel visualisation of simulation state.

        Panels:
          (0,0) Stiffness field G'(x,y) at mid-z slice [Pa], heatmap
          (0,1) Fibroblast x-y positions with 95th-pct invasion depth line
          (1,0) MMP concentration field [nM], mid-z slice
          (1,1) Dual percolation dynamics P_inf_hydrogel and P_inf_collagen vs time

        Parameters
        ----------
        time_idx : int
            Index into simulation history.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if not _HAS_MPL:
            warnings.warn("matplotlib not available; cannot plot snapshot.", ImportWarning)
            return None
        if not self._history:
            warnings.warn("No simulation history; run the simulation first.", UserWarning)
            return None

        state = self._history[time_idx]
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        mid = self.sp.grid_resolution // 2
        bs = self.sp.box_size
        extent = [0, bs, 0, bs]

        # --- Panel (0,0): Stiffness heatmap ---
        im0 = axes[0, 0].imshow(
            state.stiffness_field[:, :, mid].T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            aspect="auto",
        )
        axes[0, 0].set_title(f"Stiffness G' [Pa]  t = {state.time:.0f} s")
        axes[0, 0].set_xlabel("x [µm]")
        axes[0, 0].set_ylabel("y [µm]")
        plt.colorbar(im0, ax=axes[0, 0], label="G' [Pa]")

        # --- Panel (0,1): Cell positions ---
        if len(state.cell_positions) > 0:
            axes[0, 1].scatter(
                state.cell_positions[:, 0],
                state.cell_positions[:, 1],
                c="crimson",
                s=25,
                alpha=0.75,
                edgecolors="darkred",
                linewidths=0.4,
                label="Fibroblasts",
            )
        axes[0, 1].axvline(
            state.invasion_depth,
            color="royalblue",
            ls="--",
            lw=1.5,
            label=f"95th pct: {state.invasion_depth:.1f} µm",
        )

        # Collagen density overlay
        col_field = self._collagen.get_density_field()
        col_2d = col_field[:, :, mid]
        if col_2d.max() > 0:
            axes[0, 1].imshow(
                col_2d.T,
                origin="lower",
                extent=extent,
                cmap="YlOrBr",
                alpha=0.3,
                aspect="auto",
                vmin=0,
                vmax=col_2d.max(),
            )

        axes[0, 1].set_xlim(0, bs)
        axes[0, 1].set_ylim(0, bs)
        axes[0, 1].set_title("Fibroblast positions + collagen density")
        axes[0, 1].set_xlabel("x [µm] (invasion direction)")
        axes[0, 1].set_ylabel("y [µm]")
        axes[0, 1].legend(fontsize=8, loc="upper right")

        # --- Panel (1,0): MMP field ---
        im2 = axes[1, 0].imshow(
            state.mmp_field[:, :, mid].T,
            origin="lower",
            extent=extent,
            cmap="hot",
            aspect="auto",
        )
        axes[1, 0].set_title("MMP concentration [nM]")
        axes[1, 0].set_xlabel("x [µm]")
        axes[1, 0].set_ylabel("y [µm]")
        plt.colorbar(im2, ax=axes[1, 0], label="[MMP] [nM]")

        # --- Panel (1,1): Dual percolation dynamics ---
        times_arr = np.array([s.time for s in self._history])
        p_hyd = np.array([s.hydrogel_p_inf for s in self._history])
        p_col = np.array([s.collagen_p_inf for s in self._history])

        ax_perc = axes[1, 1]
        ax_perc.plot(times_arr, p_hyd, "b-", lw=2, label="Hydrogel P∞")
        ax_perc.plot(times_arr, p_col, "r--", lw=2, label="Collagen P∞")
        ax_perc.axhline(0.2593, color="gray", ls=":", alpha=0.7, lw=1,
                        label="p_c = 0.259")
        ax_perc.axvline(state.time, color="green", ls="-.", lw=1, alpha=0.6,
                        label=f"t = {state.time:.0f} s")
        ax_perc.set_xlabel("Time [s]")
        ax_perc.set_ylabel("Percolation order parameter P∞")
        ax_perc.set_title("Dual percolation dynamics")
        ax_perc.legend(fontsize=8)
        ax_perc.set_ylim(-0.05, 1.05)
        ax_perc.grid(True, alpha=0.3)

        fig.suptitle(
            f"Wound Healing Simulation — t = {state.time:.0f} s  "
            f"(invasion depth: {state.invasion_depth:.1f} µm)",
            fontsize=13,
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_state(self) -> None:
        """Append the current simulation state to the history list."""
        cell_pos = (
            np.array([c.position for c in self._cells], dtype=float)
            if self._cells
            else np.empty((0, 3), dtype=float)
        )
        invasion = (
            float(np.percentile(cell_pos[:, 0], 95))
            if len(cell_pos) > 0
            else 0.0
        )

        if self._network is not None:
            p_hyd = self._network.get_percolation_order_parameter()
        else:
            t_total = self.sp.n_steps * self.sp.dt
            t_now = self._step * self.sp.dt
            p_hyd = max(0.0, 1.0 - t_now / (t_total + 1e-15))

        self._history.append(
            SimulationState(
                time=float(self._step) * self.sp.dt,
                step=self._step,
                cell_positions=cell_pos.copy(),
                mmp_field=self._mmp_field.copy(),
                pdgf_field=self._pdgf_field.copy(),
                stiffness_field=self._stiffness_field.copy(),
                collagen_p_inf=self._collagen.get_percolation_order_parameter(),
                hydrogel_p_inf=p_hyd,
                invasion_depth=invasion,
                n_collagen_fibers=self._collagen.get_n_fibers(),
            )
        )
