"""
network_model.py — Module 1: HydrogelNetwork
============================================
Percolation Inversion Dynamics in Enzymatically Degrading Wound Hydrogels

Physical model
--------------
A hydrogel is represented as a 3-D random geometric graph (RGG):

  * **Nodes** are crosslink junctions placed according to a homogeneous
    Poisson point process with density ``rho_x`` (crosslinks µm⁻³) inside
    a cubic box of side ``box_size`` (µm).

  * **Edges** are polymer chains.  Two nodes are connected if their
    Euclidean distance is ≤ ``r_c`` (µm).  Each edge is assigned a
    *chain length* drawn from a lognormal distribution:

        L ~ LogNormal(µ_L, σ_L)

    with defaults chosen so the dispersity index Ð = exp(σ_L²) ≈ 1.8,
    which is typical of enzyme-catalysed ring-opening polymerisations.

  * **Bond types**

    - ``'covalent'`` — permanent backbone crosslinks; they can only be
      removed by MMP-mediated cleavage.
    - ``'ionic'`` — physical crosslinks with a finite lifetime modelled
      by Arrhenius kinetics:

          k_rupture(T) = (1/τ_ionic) * exp( -E_a/R * (1/T - 1/T_ref) )

      where T_ref = 310.15 K (body temperature).

Degradation
-----------
MMP (matrix metalloproteinase) enzymes cleave polymer chains.  Each edge *i*
carries a position-dependent cleavage rate:

    k_i(t) = k_base · [MMP](x_i, t) · f(L_i)

where ``x_i`` is the midpoint of the edge, ``[MMP]`` is the local MMP
concentration field (supplied externally, e.g. from a reaction-diffusion
solver), and

    f(L_i) = (L_i / L_mean)^alpha_access

is a chain-length accessibility factor: longer chains are more exposed to
enzyme attack (``alpha_access`` > 0, default 0.5).

In a discrete time step Δt the probability of cleavage for edge *i* is:

    p_cut,i = 1 − exp(−k_i · Δt)

Edges are removed stochastically at each step; the graph is updated in-place.

Percolation order parameter
---------------------------
P_∞(t) = |largest connected component| / N

is the conventional percolation order parameter.  When P_∞ → 0 the network
is below the percolation threshold — the gel has effectively dissolved.

Usage example
-------------
>>> from network_model import HydrogelParams, HydrogelNetwork
>>> import numpy as np
>>> params = HydrogelParams(box_size=50.0, rho_x=1.0, r_c=1.0)
>>> net = HydrogelNetwork(params, seed=42)
>>> print(net.get_percolation_order_parameter())
>>> # Simulate with a uniform MMP field of concentration 1.0
>>> mmp = np.ones((10, 10, 10))
>>> for _ in range(100):
...     net.degrade_step(mmp, dt=1.0)
>>> print(net.get_percolation_order_parameter())
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from .utils import compute_giant_component_fraction

__all__ = [
    "HydrogelParams",
    "HydrogelNetwork",
]

logger = logging.getLogger(__name__)

# Universal gas constant  [J mol⁻¹ K⁻¹]
_R_GAS: float = 8.314462618


# ---------------------------------------------------------------------------
# Physical parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class HydrogelParams:
    """
    All physical parameters governing the hydrogel network model.

    Parameters
    ----------
    box_size : float
        Side length of the cubic simulation domain [µm].  Default 50.0.
    rho_x : float
        Crosslink junction density [µm⁻³].  Controls the expected number of
        nodes N = rho_x * box_size³.  Default 1.0.
    r_c : float
        Chain cutoff radius [µm].  Two nodes within this distance are
        connected by an edge.  Default 5.0.
    mu_L : float
        Lognormal mean parameter (in ln-space) for the chain length
        distribution [µm].  Default ln(2.0) ≈ 0.693.
    sigma_L : float
        Lognormal standard-deviation parameter (in ln-space) for chain
        length.  Dispersity Ð = exp(σ_L²).  Default ~0.748, giving Ð ≈ 1.75.
    covalent_fraction : float
        Fraction of edges that are covalent (permanent backbone crosslinks).
        The remainder are ionic (labile physical crosslinks).  Default 0.7.
    k_base : float
        Base MMP cleavage rate [s⁻¹ · (concentration unit)⁻¹].  Default 0.01.
    alpha_access : float
        Exponent for the chain-length accessibility factor f(L) = (L/L_mean)^α.
        Default 0.5.
    tau_ionic : float
        Reference ionic bond lifetime [s] at temperature T_ref.  Default 3600.
    E_activation : float
        Arrhenius activation energy for ionic bond rupture [J mol⁻¹].
        Default 50 000.
    T_ref : float
        Reference temperature for Arrhenius model [K].  Default 310.15 (37 °C).
    T : float
        Simulation temperature [K].  Default 310.15 (37 °C).
    p_c : float
        Percolation threshold used as a starting guess in fitters.  Default
        0.2593, which is the bond percolation threshold for a simple cubic
        lattice — **not** appropriate for a random geometric graph (RGG).
        Empirical measurement via :py:meth:`HydrogelNetwork.measure_percolation_threshold`
        is preferred for RGG topologies.
    """

    box_size: float = 50.0
    rho_x: float = 1.0
    r_c: float = 1.0          # 1.0 µm → z≈4, p_c≈0.33 for RGG (was 5.0 → z≈524, unphysical)
    mu_L: float = float(np.log(2.0))        # lognormal location  [ln µm]
    sigma_L: float = float(np.sqrt(np.log(1.75)))  # Ð ≈ 1.75
    covalent_fraction: float = 0.70
    k_base: float = 0.0012    # s⁻¹ nM⁻¹; transition at ~step 500 with p_c≈0.547, [MMP]=1 nM
    alpha_access: float = 0.5
    tau_ionic: float = 3600.0
    E_activation: float = 50_000.0
    T_ref: float = 310.15
    T: float = 310.15
    # Default 0.2593 is the cubic-lattice bond percolation threshold; for a
    # random geometric graph use HydrogelNetwork.measure_percolation_threshold().
    p_c: float = 0.2593

    # ------------------------------------------------------------------ #
    # Derived quantities                                                   #
    # ------------------------------------------------------------------ #

    @property
    def L_mean(self) -> float:
        """Mean chain length of the lognormal distribution [µm]."""
        return math.exp(self.mu_L + 0.5 * self.sigma_L ** 2)

    @property
    def dispersity(self) -> float:
        """Dispersity index Ð = M_w / M_n = exp(σ_L²)."""
        return math.exp(self.sigma_L ** 2)

    @property
    def k_ionic_rupture(self) -> float:
        """
        Arrhenius rate of ionic bond rupture at temperature T [s⁻¹].

        k = (1/τ_ionic) · exp( -(E_a/R) · (1/T − 1/T_ref) )
        """
        exponent = -(self.E_activation / _R_GAS) * (1.0 / self.T - 1.0 / self.T_ref)
        return (1.0 / self.tau_ionic) * math.exp(exponent)

    def to_dict(self) -> dict:
        """Return JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HydrogelParams":
        """Reconstruct from a dictionary (e.g. loaded from JSON)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Main network class
# ---------------------------------------------------------------------------


class HydrogelNetwork:
    """
    3-D random geometric graph model of an enzymatically degrading hydrogel.

    The network is built once at construction time and then evolves via
    repeated calls to :py:meth:`degrade_step`.  All graph operations are
    delegated to a ``networkx.Graph``; spatial queries use a
    ``scipy.spatial.KDTree``.

    Parameters
    ----------
    params : HydrogelParams
        Physical parameters.  A default instance is created if omitted.
    seed : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    params : HydrogelParams
    graph : networkx.Graph
        Live graph.  Node attribute ``'pos'`` holds the 3-D position array.
        Edge attributes: ``'bond_type'`` (str), ``'chain_length'`` (float),
        ``'midpoint'`` (np.ndarray shape (3,)).
    time : float
        Accumulated simulation time [s].
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        params: Optional[HydrogelParams] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.params: HydrogelParams = params if params is not None else HydrogelParams()
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self.time: float = 0.0

        # Internal state
        self._positions: np.ndarray  # shape (N, 3)
        self.graph: nx.Graph

        self._build_network()

    # ------------------------------------------------------------------ #
    # Network construction helpers                                         #
    # ------------------------------------------------------------------ #

    def _build_network(self) -> None:
        """
        Initialise the random geometric graph.

        1. Sample node positions from a homogeneous Poisson point process
           (PPP) with intensity ``rho_x``.
        2. Build a KDTree and query pairs within cutoff ``r_c``.
        3. Assign lognormal chain lengths and bond types to edges.
        """
        p = self.params
        vol = p.box_size ** 3

        # ---- 1. Poisson point process for node positions ---- #
        n_expected = p.rho_x * vol
        n_nodes = int(self._rng.poisson(n_expected))
        if n_nodes < 2:
            warnings.warn(
                f"Only {n_nodes} node(s) generated. "
                "Consider increasing rho_x or box_size.",
                UserWarning,
                stacklevel=2,
            )
            n_nodes = max(n_nodes, 2)

        self._positions = self._rng.uniform(0.0, p.box_size, size=(n_nodes, 3))
        logger.debug("Generated %d nodes (expected %.1f).", n_nodes, n_expected)

        # ---- 2. Build graph and add nodes ---- #
        self.graph = nx.Graph()
        for idx, pos in enumerate(self._positions):
            self.graph.add_node(idx, pos=pos)

        # ---- 3. Spatial edge discovery with KDTree ---- #
        tree = KDTree(self._positions)
        pairs = tree.query_pairs(r=p.r_c, output_type="ndarray")  # shape (M, 2)
        logger.debug("Found %d candidate edges within r_c=%.2f µm.", len(pairs), p.r_c)

        if len(pairs) == 0:
            warnings.warn(
                "No edges found.  The network is disconnected.  "
                "Try increasing r_c or rho_x.",
                UserWarning,
                stacklevel=2,
            )
            return

        # ---- 4. Assign chain lengths and bond types ---- #
        n_edges = len(pairs)
        chain_lengths = self._rng.lognormal(
            mean=p.mu_L, sigma=p.sigma_L, size=n_edges
        )
        bond_mask = self._rng.random(n_edges) < p.covalent_fraction  # True → covalent

        for k, (i, j) in enumerate(pairs):
            i, j = int(i), int(j)
            mid = 0.5 * (self._positions[i] + self._positions[j])
            bond_type = "covalent" if bond_mask[k] else "ionic"
            self.graph.add_edge(
                i,
                j,
                bond_type=bond_type,
                chain_length=float(chain_lengths[k]),
                midpoint=mid,
            )

        logger.info(
            "Network built: %d nodes, %d edges "
            "(%.0f%% covalent, %.0f%% ionic).",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            100.0 * bond_mask.sum() / n_edges,
            100.0 * (~bond_mask).sum() / n_edges,
        )

    # ------------------------------------------------------------------ #
    # MMP field interpolation                                              #
    # ------------------------------------------------------------------ #

    def _interpolate_mmp(
        self,
        mmp_field: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Trilinearly interpolate the MMP concentration at arbitrary 3-D
        positions from a uniform rectilinear grid.

        Parameters
        ----------
        mmp_field : np.ndarray, shape (nx, ny, nz)
            MMP concentration grid spanning [0, box_size]³.
        positions : np.ndarray, shape (M, 3)
            Positions at which to evaluate [MMP].

        Returns
        -------
        np.ndarray, shape (M,)
            Interpolated MMP concentrations (floored at 0).
        """
        p = self.params
        grid_shape = np.array(mmp_field.shape, dtype=float)  # (nx, ny, nz)

        # Normalise positions to fractional grid coordinates in [0, nx-1] etc.
        frac = positions / p.box_size * (grid_shape - 1)
        frac = np.clip(frac, 0.0, grid_shape - 1 - 1e-9)

        i0 = frac[:, 0].astype(int)
        j0 = frac[:, 1].astype(int)
        k0 = frac[:, 2].astype(int)
        i1 = np.minimum(i0 + 1, int(grid_shape[0]) - 1)
        j1 = np.minimum(j0 + 1, int(grid_shape[1]) - 1)
        k1 = np.minimum(k0 + 1, int(grid_shape[2]) - 1)

        dx = frac[:, 0] - i0
        dy = frac[:, 1] - j0
        dz = frac[:, 2] - k0

        c000 = mmp_field[i0, j0, k0]
        c100 = mmp_field[i1, j0, k0]
        c010 = mmp_field[i0, j1, k0]
        c110 = mmp_field[i1, j1, k0]
        c001 = mmp_field[i0, j0, k1]
        c101 = mmp_field[i1, j0, k1]
        c011 = mmp_field[i0, j1, k1]
        c111 = mmp_field[i1, j1, k1]

        conc = (
            c000 * (1 - dx) * (1 - dy) * (1 - dz)
            + c100 * dx * (1 - dy) * (1 - dz)
            + c010 * (1 - dx) * dy * (1 - dz)
            + c110 * dx * dy * (1 - dz)
            + c001 * (1 - dx) * (1 - dy) * dz
            + c101 * dx * (1 - dy) * dz
            + c011 * (1 - dx) * dy * dz
            + c111 * dx * dy * dz
        )
        return np.maximum(conc, 0.0)

    # ------------------------------------------------------------------ #
    # Degradation step                                                     #
    # ------------------------------------------------------------------ #

    def degrade_step(
        self,
        mmp_field: np.ndarray,
        dt: float,
        T: Optional[float] = None,
    ) -> int:
        """
        Advance the degradation by one time step ``dt`` [s].

        For every surviving edge the cleavage probability is computed and
        the edge is removed stochastically.  Ionic bonds additionally
        undergo thermally driven rupture (Arrhenius).

        Algorithm
        ---------
        For each edge *e* with midpoint ``x_e`` and chain length ``L_e``:

            [MMP]_e  = interpolate(mmp_field, x_e)
            f_e      = (L_e / L_mean)^alpha_access
            k_MMP,e  = k_base · [MMP]_e · f_e
            k_e      = k_MMP,e   (covalent)
                     = k_MMP,e + k_ionic_rupture(T)   (ionic)
            p_cut,e  = 1 − exp(−k_e · dt)

        Parameters
        ----------
        mmp_field : np.ndarray, shape (nx, ny, nz)
            Current MMP concentration field over the simulation box.
            Units must be consistent with ``k_base``.
        dt : float
            Time increment [s].
        T : float or None
            Temperature for Arrhenius ionic rupture.  Overrides
            ``params.T`` if given.

        Returns
        -------
        int
            Number of edges removed in this step.
        """
        if self.graph.number_of_edges() == 0:
            self.time += dt
            return 0

        p = self.params
        temperature = T if T is not None else p.T
        k_ionic = (1.0 / p.tau_ionic) * math.exp(
            -(p.E_activation / _R_GAS) * (1.0 / temperature - 1.0 / p.T_ref)
        )
        L_mean = p.L_mean
        alpha = p.alpha_access
        k_base = p.k_base

        # Collect current edge data into arrays for vectorised computation
        edges = list(self.graph.edges(data=True))
        if not edges:
            self.time += dt
            return 0

        u_list = [e[0] for e in edges]
        v_list = [e[1] for e in edges]
        bond_types = np.array([e[2]["bond_type"] for e in edges])
        chain_lengths = np.array([e[2]["chain_length"] for e in edges], dtype=float)
        midpoints = np.array([e[2]["midpoint"] for e in edges], dtype=float)

        # Interpolate MMP field at edge midpoints
        mmp_vals = self._interpolate_mmp(mmp_field, midpoints)

        # Chain-length accessibility factor
        f_access = (chain_lengths / L_mean) ** alpha

        # MMP-mediated cleavage rate
        k_mmp = k_base * mmp_vals * f_access

        # Total rate: add ionic rupture for ionic bonds
        k_total = k_mmp.copy()
        ionic_mask = bond_types == "ionic"
        k_total[ionic_mask] += k_ionic

        # Cleavage probability in time step dt
        p_cut = 1.0 - np.exp(-k_total * dt)

        # Stochastic removal
        rand_vals = self._rng.random(len(edges))
        remove_mask = rand_vals < p_cut

        edges_to_remove = [
            (u_list[i], v_list[i]) for i in range(len(edges)) if remove_mask[i]
        ]
        self.graph.remove_edges_from(edges_to_remove)

        n_removed = int(remove_mask.sum())
        self.time += dt

        logger.debug(
            "t=%.2f s: removed %d/%d edges (%.1f%%).",
            self.time,
            n_removed,
            len(edges),
            100.0 * n_removed / len(edges) if edges else 0.0,
        )
        return n_removed

    # ------------------------------------------------------------------ #
    # Percolation / topology observables                                   #
    # ------------------------------------------------------------------ #

    def get_giant_component(self) -> nx.Graph:
        """
        Return the largest connected component as a ``networkx.Graph`` view.

        If the graph is empty (no nodes) a new empty ``Graph`` is returned.

        Returns
        -------
        networkx.Graph
            Subgraph view of the giant component.
        """
        if self.graph.number_of_nodes() == 0:
            return nx.Graph()
        components = list(nx.connected_components(self.graph))
        if not components:
            return nx.Graph()
        largest = max(components, key=len)
        return self.graph.subgraph(largest)

    def get_percolation_order_parameter(self) -> float:
        """
        Compute P_∞(t) = |giant component| / N.

        Returns
        -------
        float
            Percolation order parameter in [0, 1].
            Returns 0.0 if the network has no nodes.
        """
        n_total = self.graph.number_of_nodes()
        if n_total == 0:
            return 0.0
        gc = self.get_giant_component()
        return gc.number_of_nodes() / n_total

    def get_cluster_size_distribution(self) -> Dict[int, int]:
        """
        Compute the cluster-size distribution n_s(t).

        Returns
        -------
        dict
            Mapping ``{cluster_size: count}`` where *count* is the number
            of connected components of that size.  Sorted by cluster size.
        """
        distribution: Dict[int, int] = {}
        for component in nx.connected_components(self.graph):
            s = len(component)
            distribution[s] = distribution.get(s, 0) + 1
        return dict(sorted(distribution.items()))

    # ------------------------------------------------------------------ #
    # Spatial / structural accessors                                       #
    # ------------------------------------------------------------------ #

    def get_node_positions(self) -> np.ndarray:
        """
        Return all node positions.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Each row is the (x, y, z) coordinate [µm] of a node, ordered
            by node index.
        """
        return self._positions.copy()

    def get_active_edges(self) -> List[Tuple[int, int]]:
        """
        Return the currently surviving edges.

        Returns
        -------
        list of (int, int)
            Each tuple is a pair of node indices (i, j) with i < j.
        """
        return list(self.graph.edges())

    def compute_local_p(
        self,
        resolution: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a spatial map of the *local bond density* as a proxy for
        local stiffness / connectivity.

        The simulation box is divided into a uniform ``resolution³`` grid.
        For each voxel the local connectivity is estimated as:

            p_local(v) = n_active_edges(v) / n_possible_edges(v)

        where ``n_active_edges(v)`` is the number of surviving edges whose
        *midpoint* falls in voxel *v*, and ``n_possible_edges(v)`` is the
        number of node pairs within that voxel (i.e. the maximum possible
        given local density).

        If a voxel has fewer than 2 nodes the value is set to ``NaN``
        (undefined).

        Parameters
        ----------
        resolution : int
            Number of grid points along each spatial axis.  Total voxels =
            ``resolution³``.  Default 10.

        Returns
        -------
        grid_coords : np.ndarray, shape (resolution, resolution, resolution, 3)
            Centroid coordinates [µm] of each voxel.
        local_p_values : np.ndarray, shape (resolution, resolution, resolution)
            Local bond occupation fraction (proxy for local stiffness).
            ``NaN`` where undefined.
        """
        p = self.params
        edges = list(self.graph.edges(data=True))

        # Build voxel coordinate grid (centroids)
        edges_1d = np.linspace(0.0, p.box_size, resolution + 1)
        centres_1d = 0.5 * (edges_1d[:-1] + edges_1d[1:])
        gx, gy, gz = np.meshgrid(centres_1d, centres_1d, centres_1d, indexing="ij")
        grid_coords = np.stack([gx, gy, gz], axis=-1)  # (res, res, res, 3)

        voxel_size = p.box_size / resolution

        # Count nodes per voxel
        node_voxel_idx = np.floor(self._positions / voxel_size).astype(int)
        node_voxel_idx = np.clip(node_voxel_idx, 0, resolution - 1)
        node_counts = np.zeros((resolution, resolution, resolution), dtype=int)
        for vi, vj, vk in node_voxel_idx:
            node_counts[vi, vj, vk] += 1

        # Count active edges per voxel (by midpoint)
        active_edge_counts = np.zeros((resolution, resolution, resolution), dtype=int)
        for u, v, data in edges:
            mid = data["midpoint"]
            vi = int(min(math.floor(mid[0] / voxel_size), resolution - 1))
            vj = int(min(math.floor(mid[1] / voxel_size), resolution - 1))
            vk = int(min(math.floor(mid[2] / voxel_size), resolution - 1))
            active_edge_counts[vi, vj, vk] += 1

        # Possible edges per voxel = n*(n-1)/2
        possible_edges = node_counts * (node_counts - 1) // 2

        local_p_values = np.full((resolution, resolution, resolution), np.nan)
        valid = possible_edges > 0
        local_p_values[valid] = (
            active_edge_counts[valid].astype(float) / possible_edges[valid].astype(float)
        )

        return grid_coords, local_p_values

    def measure_percolation_threshold(
        self,
        n_p_points: int = 30,
        n_trials: int = 5,
        rng_seed: int = 0,
    ) -> float:
        """Estimate p_c empirically via bond dilution on this network topology.

        Progressively removes bonds at random and tracks P_inf(p). The
        threshold is the inflection point (steepest descent) of P_inf(p).
        This is network-specific and supersedes the lattice value 0.2593.

        Parameters
        ----------
        n_p_points : int
            Number of bond-fraction values to probe in [0, 1].
        n_trials : int
            Number of independent dilution trials (averaged for smoothing).
        rng_seed : int
            Seed for the random number generator.

        Returns
        -------
        float
            Estimated percolation threshold p_c.
        """
        rng = np.random.default_rng(rng_seed)
        all_edges = list(self.graph.edges())
        n_edges = len(all_edges)
        if n_edges == 0:
            return 0.5

        p_values = np.linspace(0.05, 0.95, n_p_points)
        P_inf_mean = np.zeros(n_p_points)

        for _ in range(n_trials):
            perm = rng.permutation(n_edges)
            for j, p in enumerate(p_values):
                n_keep = int(round(p * n_edges))
                keep_indices = perm[:n_keep]
                sub_edges = [all_edges[i] for i in keep_indices]
                nodes = list(self.graph.nodes())
                H = nx.Graph()
                H.add_nodes_from(nodes)
                H.add_edges_from(sub_edges)
                P_inf_mean[j] += compute_giant_component_fraction(H)

        P_inf_mean /= n_trials

        # p_c = inflection point (maximum of dP_inf/dp, i.e., steepest ascent)
        dP = np.gradient(P_inf_mean, p_values)
        p_c_idx = int(np.argmax(dP))
        return float(p_values[p_c_idx])

    def compute_susceptibility(self) -> float:
        """Percolation susceptibility χ = Σ s² n_s / N (non-spanning clusters).

        Diverges as χ ~ |p − p_c|^{-γ} with γ = 1.8 near p_c.  This is the
        theoretically correct diverging quantity for percolation EWS; it provides
        a physically grounded early warning signal that avoids the finite-size
        limitations of variance-based EWS.

        Returns
        -------
        float
            χ in units of nodes.  Returns 0.0 for empty graphs.
        """
        N = self.graph.number_of_nodes()
        if N == 0:
            return 0.0
        components = list(nx.connected_components(self.graph))
        if not components:
            return 0.0
        sizes = [len(c) for c in components]
        max_size = max(sizes)
        # Sum s² over all clusters EXCEPT the spanning (largest) component
        chi = sum(s * s for s in sizes if s < max_size) / N
        return float(chi)

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """
        Serialise the full network state to disk using ``pickle``.

        The saved object includes the ``HydrogelParams``, the live
        ``networkx.Graph`` (with all node/edge attributes), the node
        positions array, and the accumulated simulation time.

        Parameters
        ----------
        path : str or Path
            File path (e.g. ``'network_t100s.pkl'``).
        """
        path = Path(path)
        state = {
            "params": self.params.to_dict(),
            "time": self.time,
            "positions": self._positions,
            "graph_data": nx.node_link_data(self.graph, edges="links"),
        }
        with path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Network saved to '%s'.", path)

    @classmethod
    def load(cls, path: str | Path) -> "HydrogelNetwork":
        """
        Restore a network from a ``pickle`` file created by :py:meth:`save`.

        Parameters
        ----------
        path : str or Path
            File path to load.

        Returns
        -------
        HydrogelNetwork
            Fully reconstructed network instance.
        """
        path = Path(path)
        with path.open("rb") as fh:
            state = pickle.load(fh)

        params = HydrogelParams.from_dict(state["params"])
        net = cls.__new__(cls)
        net.params = params
        net._rng = np.random.default_rng()   # fresh RNG after reload
        net.time = float(state["time"])
        net._positions = state["positions"]
        net.graph = nx.node_link_graph(state["graph_data"], edges="links")

        # Restore numpy midpoint arrays (JSON round-trip converts to lists)
        for u, v, data in net.graph.edges(data=True):
            mid = data.get("midpoint")
            if mid is not None and not isinstance(mid, np.ndarray):
                net.graph[u][v]["midpoint"] = np.array(mid, dtype=float)

        logger.info("Network loaded from '%s' (t=%.2f s).", path, net.time)
        return net

    def to_json_dict(self) -> dict:
        """
        Export a JSON-compatible dictionary of the network state.

        Numpy arrays and graph structure are converted to plain Python
        objects.  Useful for lightweight logging or REST-API responses.
        Node-link format is used for the graph (same as ``networkx``
        ``node_link_data``).

        Returns
        -------
        dict
            JSON-serialisable representation of the network state.

        Notes
        -----
        The midpoint arrays stored on edges are converted to lists.  On
        re-import via :py:meth:`load` or direct reconstruction these are
        automatically converted back.
        """
        graph_dict = nx.node_link_data(self.graph, edges="links")

        # Convert numpy objects to plain Python for JSON compatibility
        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        for node in graph_dict.get("nodes", []):
            for k, v in list(node.items()):
                node[k] = _convert(v)
        for link in graph_dict.get("links", []):
            for k, v in list(link.items()):
                link[k] = _convert(v)

        return {
            "params": self.params.to_dict(),
            "time": self.time,
            "positions": self._positions.tolist(),
            "graph": graph_dict,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "HydrogelNetwork":
        """
        Reconstruct a ``HydrogelNetwork`` from a JSON-compatible dictionary
        produced by :py:meth:`to_json_dict`.

        Parameters
        ----------
        d : dict
            Dictionary as returned by ``to_json_dict()``.

        Returns
        -------
        HydrogelNetwork
        """
        params = HydrogelParams.from_dict(d["params"])
        net = cls.__new__(cls)
        net.params = params
        net._rng = np.random.default_rng()
        net.time = float(d["time"])
        net._positions = np.array(d["positions"], dtype=float)
        net.graph = nx.node_link_graph(d["graph"], edges="links")

        for u, v, data in net.graph.edges(data=True):
            mid = data.get("midpoint")
            if mid is not None and not isinstance(mid, np.ndarray):
                net.graph[u][v]["midpoint"] = np.array(mid, dtype=float)
        return net

    # ------------------------------------------------------------------ #
    # Elastic-network export (rigidity percolation)                        #
    # ------------------------------------------------------------------ #

    def to_elastic_arrays(self):
        """Export this RGG's current bond topology for the elastic solver
        in :mod:`gelrigidity.rigidity`.

        Returns
        -------
        pos : (N, 3) ndarray
            Node positions (µm).
        box_size : float
            Cube side length (µm), used only to normalise the energy
            density into a modulus; this network is an OPEN (non-periodic)
            specimen, so ``box_size`` here is a finite-volume approximation,
            not a periodic cell.
        bonds : (M, 2) int ndarray
            Currently PRESENT edges only (u, v) node-index pairs.
        rhat : (M, 3) ndarray
            Unit bond vectors (u -> v).
        rvec : (M, 3) ndarray
            Full bond vectors (u -> v), unwrapped (no periodic image
            convention needed — this is an open specimen).

        Notes
        -----
        Every edge currently in ``self.graph`` is treated as present
        (occupied); edges already removed by cleavage are absent from the
        graph and therefore automatically excluded. This lets
        ``rigidity.ElasticNetwork`` be evaluated on the *actual* simulated
        network at any point in its degradation/deposition trajectory, not
        just on an idealised bond-dilution ensemble.
        """
        edges = np.array(self.graph.edges(), dtype=np.int64)
        if edges.size == 0:
            return self._positions, self.params.box_size, edges.reshape(0, 2), \
                np.zeros((0, 3)), np.zeros((0, 3))
        rvec = self._positions[edges[:, 1]] - self._positions[edges[:, 0]]
        lengths = np.linalg.norm(rvec, axis=1)
        lengths = np.where(lengths > 0, lengths, 1.0)
        rhat = rvec / lengths[:, None]
        return self._positions, self.params.box_size, edges, rhat, rvec

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"HydrogelNetwork("
            f"N={self.graph.number_of_nodes()}, "
            f"E={self.graph.number_of_edges()}, "
            f"P_inf={self.get_percolation_order_parameter():.4f}, "
            f"t={self.time:.2f} s)"
        )

    def __getstate__(self) -> dict:
        """Support ``pickle.dumps(net)`` directly."""
        return {
            "params": self.params.to_dict(),
            "time": self.time,
            "positions": self._positions,
            "graph_data": nx.node_link_data(self.graph, edges="links"),
            "rng_state": self._rng.bit_generator.state,
        }

    def __setstate__(self, state: dict) -> None:
        """Support ``pickle.loads(data)`` directly."""
        self.params = HydrogelParams.from_dict(state["params"])
        self.time = float(state["time"])
        self._positions = state["positions"]
        self.graph = nx.node_link_graph(state["graph_data"], edges="links")
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]

        for u, v, data in self.graph.edges(data=True):
            mid = data.get("midpoint")
            if mid is not None and not isinstance(mid, np.ndarray):
                self.graph[u][v]["midpoint"] = np.array(mid, dtype=float)
