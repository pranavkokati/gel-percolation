"""
dynamics.py — Coupled degrading-scaffold + cell-deposited-ECM network dynamics.

Two bond populations live on ONE shared periodic random-geometric-graph (RGG)
node set — the same disordered topology used by ``network_model.py`` for the
hydrogel (homogeneous Poisson point process of crosslink nodes, edges within
cutoff ``r_c``), generated here under periodic boundary conditions so a single
realisation has no free surface. An FCC-lattice mode is retained
(``topology="fcc"``) purely as the ordered cross-check used in Phase 0 solver
validation against the published Chubynsky-Thorpe FCC bond-rigidity threshold;
production runs use ``topology="rgg"`` (the default), which is physically the
correct disordered network for an enzymatically degrading hydrogel.

  * scaffold bonds  — present at t=0, removed over time by spatially-resolved
                      MMP cleavage (REVERSE percolation).
  * ECM bonds       — absent at t=0, deposited over time by cells that secrete
                      matrix into their local neighbourhood (FORWARD percolation).

Because both populations share the node set, the *union* network is a single
elastic network whose shear modulus is solved directly by rigidity.ElasticNetwork
(no assumed constitutive law, no hand-drawn collagen sigmoid). At every recorded
time we measure, for scaffold-only / ECM-only / union:
    - connectivity P_inf  (largest-cluster fraction)
    - rigidity  (does a spanning STRESS-BEARING cluster exist; measured G)

This is the object the original repository lacked: the modulus and both
percolation channels are emergent from the same evolving network, so the
scaffold->ECM load handoff can be judged mechanically rather than by connectivity.

Degradation kinetics
--------------------
Each scaffold bond has a cleavage hazard per step
    p_cut = 1 - exp(-k_eff * dt),   k_eff = k_base * [MMP](x) * (L/L_mean)**alpha
where [MMP](x) is the local protease concentration (uniform field or cell-sourced),
L is the bond's chain length (lognormal, dispersity ~1.75), alpha>0 makes longer
chains cleave faster (more scissile bonds). This is the same reverse-percolation
kinetics used by Abete et al., PRL 93, 228301 (2004), but resolved on a 3D
elastic network so mechanics (not just gel fraction) is tracked.

Deposition kinetics
-------------------
Cells sit at random node positions. Each step, a candidate ECM bond within a
cell's secretion radius is deposited with probability p_dep = 1 - exp(-k_dep*dt),
biased toward bonds near cells. ECM bonds carry stiffness k_ecm relative to
scaffold (k_scaf = 1). This replaces the hard-coded logistic collagen curve.
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from .rigidity import fcc_lattice, neighbour_bonds, ElasticNetwork, periodic_poisson_rgg


def largest_component_fraction(N, bonds_sub):
    if len(bonds_sub) == 0:
        return 0.0
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(bonds_sub)
    gc = max((len(c) for c in nx.connected_components(G)), default=0)
    return gc / N


class CoupledNetwork:
    """Coupled scaffold-degradation + ECM-deposition on a shared periodic
    random geometric graph (default) or FCC lattice (Phase-0 cross-check)."""

    def __init__(self, L=8, a=1.0, r_cut=0.75,
                 k_scaffold=1.0, k_ecm=1.0,
                 seed=0, topology="rgg", rho_x=1.0, box_size=None):
        """
        topology : {"rgg", "fcc"}
            "rgg" (default): periodic Poisson random geometric graph, the
            physical hydrogel topology (matches network_model.HydrogelNetwork).
            Node density is ``rho_x`` and box edge is ``box_size`` (defaults
            to ``L`` if not given, so existing FCC-era call sites that only
            set ``L`` still produce a comparably-sized system).
            "fcc": ordered FCC lattice, retained only as the Phase-0
            validation cross-check against literature FCC thresholds.
        """
        self.rng = np.random.default_rng(seed)
        self.topology = topology
        if topology == "fcc":
            self.pos, self.box = fcc_lattice(L, a)
            self.bonds, self.rhat, self.rvec = neighbour_bonds(self.pos, self.box, r_cut)
        elif topology == "rgg":
            box = box_size if box_size is not None else float(L)
            self.pos, self.box, self.bonds, self.rhat, self.rvec = periodic_poisson_rgg(
                rho_x=rho_x, box_size=box, r_c=r_cut, seed=seed)
        else:
            raise ValueError(f"unknown topology {topology!r}; expected 'rgg' or 'fcc'")
        self.net = ElasticNetwork(self.pos, self.box, self.bonds, self.rhat, self.rvec)
        self.N = self.net.N
        self.M = self.net.M
        self.k_scaffold = k_scaffold
        self.k_ecm = k_ecm

        # chain length per bond (lognormal, dispersity ~1.75 -> sigma~0.748)
        self.L_chain = self.rng.lognormal(mean=np.log(1.0), sigma=0.748, size=self.M)
        self.L_mean = self.L_chain.mean()

        # bond midpoints (for cell-local MMP / deposition)
        a_idx, b_idx = self.bonds[:, 0], self.bonds[:, 1]
        mid = 0.5 * (self.pos[a_idx] + self.pos[b_idx])
        self.bond_mid = mid

        # scaffold occupancy: start with a chosen initial fraction present
        self.scaffold = None
        self.ecm = None

    # --------------------------------------------------------------------- #
    def seed_scaffold(self, p0=1.0):
        """Initial scaffold bond occupancy (fraction p0 present, well above p_r)."""
        self.scaffold = self.rng.random(self.M) < p0

    def seed_cells(self, n_cells=20, secretion_radius=2.0):
        """Place cells at random node positions; precompute per-cell bond neighbourhoods."""
        cell_nodes = self.rng.choice(self.N, size=n_cells, replace=False)
        self.cell_pos = self.pos[cell_nodes]
        self.secretion_radius = secretion_radius
        # distance from each candidate bond midpoint to nearest cell (min image)
        d = np.full(self.M, np.inf)
        for c in self.cell_pos:
            dr = self.bond_mid - c
            dr -= self.box * np.round(dr / self.box)
            dd = np.linalg.norm(dr, axis=1)
            d = np.minimum(d, dd)
        self.dist_to_cell = d
        # ECM starts empty
        self.ecm = np.zeros(self.M, bool)

    # --------------------------------------------------------------------- #
    def degrade_step(self, mmp_level, k_base, dt, alpha=0.75, force_bias=None, beta_f=0.0):
        """One MMP cleavage step on scaffold bonds (uniform MMP field).

        force_bias : optional (M,) array of per-bond force magnitudes from the
            most recent elastic-stress refresh (see ``refresh_stress``). When
            given, the cleavage hazard is modulated by a Bell/catch-slip-style
            exponential factor ``exp(beta_f * (f_i - f_median))``, following
            the experimentally measured force-accelerated proteolysis of
            loaded collagen by MMP-1 (Adhikari, Chai & Dunn, J. Am. Chem.
            Soc. 133(6), 1686-1689 (2011): ~100-fold rate increase under
            ~10 pN load). beta_f=0
            recovers the original force-independent (memoryless) kinetics.
        """
        active = self.scaffold
        k_eff = k_base * mmp_level * (self.L_chain / self.L_mean) ** alpha
        if force_bias is not None and beta_f != 0.0:
            f_med = np.median(force_bias[active]) if active.any() else 0.0
            k_eff = k_eff * np.exp(beta_f * (force_bias - f_med))
        p_cut = 1.0 - np.exp(-k_eff * dt)
        cut = (self.rng.random(self.M) < p_cut) & active
        self.scaffold = active & ~cut

    def refresh_stress(self, shear=("x", "z")):
        """Re-solve the elastic problem on the CURRENT scaffold and cache the
        per-bond force field. This is the costly step that is only performed
        every ``stress_update_every`` timesteps in ``run(..., mode='stress')``
        -- the finite cadence between refreshes is the physical relaxation
        timescale that makes bond removal genuinely rate-dependent (fast
        degradation removes many bonds per refresh on stale stress
        information; slow degradation stays close to the quasistatic,
        stress-tracking removal order). Returns the (M,) force array (0 for
        absent bonds) and caches it as ``self._last_force``.
        """
        kvec = np.where(self.scaffold, self.k_scaffold, 0.0)
        self._last_force = self.net.bond_forces(self.scaffold, kvec=kvec, shear=shear)
        return self._last_force

    def deposit_step(self, k_dep, dt):
        """One ECM deposition step: cells lay down bonds near themselves."""
        # deposition hazard decays with distance from nearest cell (exp kernel)
        within = self.dist_to_cell < self.secretion_radius
        kernel = np.exp(-self.dist_to_cell / (0.5 * self.secretion_radius))
        p_dep = (1.0 - np.exp(-k_dep * dt)) * kernel
        new = (self.rng.random(self.M) < p_dep) & within & (~self.ecm)
        self.ecm = self.ecm | new

    # --------------------------------------------------------------------- #
    def union_occ_and_k(self):
        """Union bond mask and per-bond stiffness (scaffold+ECM; ECM overrides k)."""
        occ = self.scaffold | self.ecm
        kvec = np.zeros(self.M)
        kvec[self.scaffold] = self.k_scaffold
        # ECM present: add its stiffness (parallel springs where both present)
        kvec[self.ecm] += self.k_ecm
        return occ, kvec

    def measure(self, solve_rigidity=True):
        """Measure connectivity + rigidity for scaffold, ECM, union at current state."""
        occ_u, kvec = self.union_occ_and_k()
        out = {
            "p_scaffold": self.scaffold.mean(),
            "p_ecm": self.ecm.mean(),
            "p_union": occ_u.mean(),
            "Pinf_scaffold": largest_component_fraction(self.N, self.bonds[self.scaffold]),
            "Pinf_ecm": largest_component_fraction(self.N, self.bonds[self.ecm]),
            "Pinf_union": largest_component_fraction(self.N, self.bonds[occ_u]),
            "z_union": self.net.mean_coordination(occ_u),
        }
        if solve_rigidity:
            out["G_scaffold"] = self.net.shear_modulus(
                self.scaffold, kvec=np.where(self.scaffold, self.k_scaffold, 0.0))
            out["G_ecm"] = self.net.shear_modulus(
                self.ecm, kvec=np.where(self.ecm, self.k_ecm, 0.0))
            out["G_union"] = self.net.shear_modulus(occ_u, kvec=kvec)
        return out

    def run(self, n_steps=400, dt=1.0, record_every=5,
            mmp_level=1.0, k_base=0.01, alpha=0.75,
            k_dep=0.004, deposition_delay=0, solve_rigidity=True,
            beta_f=0.0, stress_update_every=1):
        """Run coupled degradation + deposition, recording time series.

        beta_f : float
            Force-sensitivity of the cleavage hazard (see ``degrade_step``).
            beta_f=0 (default) reproduces the original memoryless kinetics,
            for which the surviving-bond ensemble at any accumulated
            occupancy fraction p is statistically independent of the rate
            k_base (Poisson thinning is history-independent) -- i.e. no
            genuine quench-rate physics is possible in that limit. beta_f>0
            makes bond removal preferentially strip the most heavily loaded
            bonds *as resolved at the last stress refresh*, which is
            rate-dependent: a fast quench (large k_base relative to the
            stress-refresh cadence) removes many bonds per refresh on stale
            (pre-degradation) stress information, while a slow quench stays
            close to the quasistatic, always-current-stress removal order.
            This rate-dependence of *which* bonds fail is the mechanism that
            can produce a Kibble-Zurek-type freeze-out lag between the
            driving rate and the network's mechanical response.
        stress_update_every : int
            Re-solve the elastic stress field (costly) every this many
            steps; between refreshes, degrade_step reuses the cached
            force field. Only used when beta_f != 0.
        """
        rec = {kk: [] for kk in
               ["t", "p_scaffold", "p_ecm", "p_union",
                "Pinf_scaffold", "Pinf_ecm", "Pinf_union", "z_union",
                "G_scaffold", "G_ecm", "G_union"]}
        force_bias = None
        if beta_f != 0.0:
            force_bias = self.refresh_stress()
        for step in range(n_steps + 1):
            if step > 0:
                if beta_f != 0.0 and (step - 1) % stress_update_every == 0:
                    force_bias = self.refresh_stress()
                self.degrade_step(mmp_level, k_base, dt, alpha,
                                   force_bias=force_bias, beta_f=beta_f)
                if step >= deposition_delay:
                    self.deposit_step(k_dep, dt)
            if step % record_every == 0:
                m = self.measure(solve_rigidity=solve_rigidity)
                rec["t"].append(step * dt)
                for kk, vv in m.items():
                    rec[kk].append(vv)
        return {kk: np.asarray(vv) for kk, vv in rec.items()}
