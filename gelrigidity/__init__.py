"""
gelrigidity — dual-rigidity-percolation load-path-continuity model for
enzymatically degrading hydrogel wound scaffolds undergoing simultaneous
collagen/ECM deposition.

Public API
----------
network      : HydrogelNetwork / HydrogelParams — RGG scaffold generator
               (Poisson-point crosslinks, MMP-cleavage kinetics, ECM
               deposition kinetics).
rigidity     : ElasticNetwork, periodic_poisson_rgg, fcc_lattice,
               neighbour_bonds — non-affine central-force elastic solver
               that MEASURES the shear modulus G(p) and rigidity-percolation
               exponent f from the bond network (no closed-form/hardcoded
               constitutive law).
dynamics     : CoupledNetwork — couples scaffold degradation and ECM
               deposition on one shared periodic RGG substrate and computes
               G_scaffold(t), G_ecm(t), G_union(t).
handoff      : load_path_continuity_Q, rigidity_connectivity_lag,
               summarize_trajectory — the SINGLE, unified definition of the
               percolation-handoff design metric,
               Q = min_t G_union(t) / G_target.
mean_field   : affine_meanfield_modulus, meanfield_union_trajectory,
               rigidity_gap_bias — the classical mean-field/step-function
               reverse-gelation constitutive law (Akalp, Bryant & Vernerey,
               Soft Matter 12, 7505 (2016)), evaluated on the SAME measured
               occupancy trajectory as the rigidity-percolation solver, for
               direct quantitative baseline comparison.

See REPORT.md for the audit of the original repository and the physical
justification for each module, and the manuscript draft (paper/) for the
scientific narrative.
"""

from .rigidity import ElasticNetwork, periodic_poisson_rgg, fcc_lattice, neighbour_bonds
from .dynamics import CoupledNetwork
from .handoff import load_path_continuity_Q, rigidity_connectivity_lag, summarize_trajectory
from .network import HydrogelNetwork, HydrogelParams
from .mean_field import affine_meanfield_modulus, meanfield_union_trajectory, rigidity_gap_bias

__all__ = [
    "ElasticNetwork", "periodic_poisson_rgg", "fcc_lattice", "neighbour_bonds",
    "CoupledNetwork",
    "load_path_continuity_Q", "rigidity_connectivity_lag", "summarize_trajectory",
    "HydrogelNetwork", "HydrogelParams",
    "affine_meanfield_modulus", "meanfield_union_trajectory", "rigidity_gap_bias",
]

__version__ = "0.2.0"
