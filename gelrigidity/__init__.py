"""
gelrigidity: network-resolved rigidity percolation for degrading hydrogels that
are replaced by newly deposited matrix.

Modules
-------
rigidity    ElasticNetwork, periodic_poisson_rgg, fcc_lattice, neighbour_bonds.
            Linear-response shear modulus of a central-force spring network on a
            periodic random geometric graph, with an optional transverse
            stiffness ratio ``eps``. The modulus is computed from the bond
            network, not assumed.
dynamics    CoupledNetwork. Scaffold bonds are cleaved and matrix bonds are
            deposited on one shared node set; the scaffold, matrix and union
            moduli are recorded over time.
handoff     load_path_continuity_Q, rigidity_connectivity_lag,
            summarize_trajectory. The design scalar
            Q = min_t G_union(t) / G_target and the lag between loss of rigidity
            and loss of connectivity.
mean_field  affine_meanfield_modulus, meanfield_union_trajectory,
            rigidity_gap_bias. The affine reverse-gelation law of Akalp, Bryant
            and Vernerey (Soft Matter 12, 7505, 2016), evaluated on the same
            occupancy trajectory as the solver.

See README.md for the scope of the model and docs/METHODS.md for definitions.
"""

from .rigidity import ElasticNetwork, periodic_poisson_rgg, fcc_lattice, neighbour_bonds
from .dynamics import CoupledNetwork
from .handoff import load_path_continuity_Q, rigidity_connectivity_lag, summarize_trajectory
from .mean_field import affine_meanfield_modulus, meanfield_union_trajectory, rigidity_gap_bias

__all__ = [
    "ElasticNetwork", "periodic_poisson_rgg", "fcc_lattice", "neighbour_bonds",
    "CoupledNetwork",
    "load_path_continuity_Q", "rigidity_connectivity_lag", "summarize_trajectory",
    "affine_meanfield_modulus", "meanfield_union_trajectory", "rigidity_gap_bias",
]

__version__ = "0.2.0"
