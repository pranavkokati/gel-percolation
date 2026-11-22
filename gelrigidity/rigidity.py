"""
rigidity.py — Elastic-network shear-modulus solver and rigidity percolation.

The shear modulus is computed from the actual bond network by a non-affine
linear-response elastic solve. No constitutive law is assumed, so any fitted
rigidity-percolation exponent is an output of the fit, not an input.

Physics
-------
A central-force spring network on a periodic random geometric graph (RGG) —
a disordered network of crosslinks (homogeneous Poisson point process, edges
wherever two nodes fall within a cutoff distance r_c), generated under periodic
boundary conditions so that a single realisation has no free surface. An FCC-lattice
generator (``fcc_lattice``) is retained purely as an independent, ordered-
topology cross-check against the published FCC bond-rigidity threshold
(Chubynsky & Thorpe 2007); it is NOT the physical model of the hydrogel and
is not used for the production measurements below.

Each present bond is a Hookean spring of unit stiffness acting along the
bond direction:

    E = (1/2) * sum_bonds k * ( n_hat . (u_a - u_b) + delta_affine )**2

Under an imposed simple shear (deformation gradient F = I + gamma * e_x (x) e_z),
the affine bond extension is delta_affine = gamma * r_z * n_x. We minimise the
elastic energy over the non-affine displacement field u' (periodic), then read
off the shear modulus from the residual energy density:

    E_min = (1/2) * G * gamma**2 * V   =>   G = 2 * E_min / (gamma**2 * V)

Because the response is linear we set gamma = 1. A fully-connected lattice gives a
finite G; a sparse network relaxes the affine strain at near-zero cost (G -> 0).
The rigidity threshold p_r (where G first becomes non-zero) lies ABOVE the
connectivity threshold p_c: a network can be geometrically connected yet
mechanically floppy. That separation is the physical core of this project.

References for the method (not implementation copied from):
  - S. Feng, P. N. Sen, PRL 52, 216 (1984) — central-force rigidity percolation.
  - M. Thorpe, J. Non-Cryst. Solids 57, 355 (1983) — floppy modes / constraint counting.
  - Head, Levine, MacKintosh, PRE 68, 061907 (2003) — non-affine response of networks.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import identity as sp_identity, diags
from scipy.sparse.linalg import lsmr, splu
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
#  Lattice construction
# --------------------------------------------------------------------------- #
def fcc_lattice(L: int, a: float = 1.0):
    """Face-centred-cubic lattice, L conventional cells per side.

    Returns
    -------
    pos : (N, 3) float array of node positions, N = 4 * L**3
    box : float, cubic box edge length (periodic)
    """
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.0, 0.5, 0.5]]) * a
    cells = np.array([[i, j, k]
                      for i in range(L) for j in range(L) for k in range(L)],
                     dtype=float) * a
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    box = L * a
    return pos, box


def periodic_poisson_rgg(rho_x: float, box_size: float, r_c: float, seed=None):
    """Periodic 3-D Poisson random geometric graph — the production topology.

    Nodes are a homogeneous Poisson point process of intensity ``rho_x``;
    edges join any two nodes within ``r_c``. The cube is periodic
    (minimum-image convention), so a single realisation has no free surface.

    Parameters
    ----------
    rho_x : float
        Node number density (nodes / volume).
    box_size : float
        Cubic periodic cell edge length.
    r_c : float
        Bond cutoff radius.
    seed : int or None
        RNG seed.

    Returns
    -------
    pos  : (N, 3) float array of node positions
    box  : float, the periodic box edge (== box_size)
    bonds: (M, 2) int array of node-index pairs
    rhat : (M, 3) float array of unit bond vectors (minimum image, a->b)
    rvec : (M, 3) float array of bond vectors (minimum image, a->b)
    """
    rng = np.random.default_rng(seed)
    n_expected = rho_x * box_size ** 3
    n_nodes = int(rng.poisson(n_expected))
    n_nodes = max(n_nodes, 2)
    pos = rng.uniform(0.0, box_size, size=(n_nodes, 3))
    bonds, rhat, rvec = neighbour_bonds(pos, box_size, r_c)
    return pos, box_size, bonds, rhat, rvec


def neighbour_bonds(pos: np.ndarray, box: float, r_cut: float):
    """All unique bonds within r_cut under periodic minimum-image convention.

    Returns
    -------
    bonds : (M, 2) int array of node-index pairs
    rhat  : (M, 3) float array of unit bond vectors (minimum image, a->b)
    rvec  : (M, 3) float array of bond vectors (minimum image, a->b)
    """
    tree = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(r_cut, output_type="ndarray")
    a, b = pairs[:, 0], pairs[:, 1]
    dr = pos[a] - pos[b]
    dr -= box * np.round(dr / box)          # minimum image
    L = np.linalg.norm(dr, axis=1)
    rhat = dr / L[:, None]
    return pairs, rhat, dr


# --------------------------------------------------------------------------- #
#  Elastic solver
# --------------------------------------------------------------------------- #
class ElasticNetwork:
    """Central-force elastic network on a fixed periodic node set.

    Parameters
    ----------
    pos   : (N,3) node positions
    box   : periodic box edge
    bonds : (M,2) candidate bond list (node index pairs)
    rhat  : (M,3) unit bond vectors
    rvec  : (M,3) bond vectors (minimum image)
    """

    def __init__(self, pos, box, bonds, rhat, rvec):
        self.pos = np.asarray(pos, float)
        self.N = len(pos)
        self.box = float(box)
        self.V = self.box ** 3
        self.bonds = np.asarray(bonds, int)
        self.rhat = np.asarray(rhat, float)
        self.rvec = np.asarray(rvec, float)
        self.M = len(self.bonds)

    # --- bond-projection matrix B (present bonds only) -------------------- #
    def _projection(self, occ, kvec):
        """Sparse (m_occ, 3N) matrix B with sqrt(k)*n_hat / -sqrt(k)*n_hat rows,
        so that (B u) = sqrt(k) * n_hat . (u_a - u_b) per present bond."""
        idx = np.flatnonzero(occ)
        m = len(idx)
        a = self.bonds[idx, 0]
        b = self.bonds[idx, 1]
        nh = self.rhat[idx] * np.sqrt(kvec[idx])[:, None]
        rows = np.repeat(np.arange(m), 6)
        cols = np.column_stack([3 * a, 3 * a + 1, 3 * a + 2,
                                3 * b, 3 * b + 1, 3 * b + 2]).ravel()
        vals = np.column_stack([nh, -nh]).ravel()
        B = coo_matrix((vals, (rows, cols)), shape=(m, 3 * self.N)).tocsr()
        return B, idx

    def _projection_eps(self, occ, kvec, eps, di, gi):
        """Augmented operator for transverse stiffness ``eps`` in (0, 1].

        Bond energy  (k/2) * [ (1-eps) (n.d)^2 + eps |d|^2 ],  d = du + delta_aff,
        i.e. longitudinal stiffness k and transverse stiffness eps*k.  eps = 0
        is the central-force network; eps = 1 is the isotropic zero-rest-length
        (Gaussian / phantom) spring.  Physically eps ~ (bond tension)/(k_par *
        bond length): pre-stress or entropic transverse stiffness.
        Returns (B, delta) so that E = 1/2 |delta + B u|^2.
        """
        idx = np.flatnonzero(occ)
        m = len(idx)
        a = self.bonds[idx, 0]
        b = self.bonds[idx, 1]
        k = kvec[idx]
        nh = self.rhat[idx]
        rg = self.rvec[idx, gi]
        blocks, deltas = [], []
        N3 = 3 * self.N
        if eps < 1.0:
            wc = np.sqrt(k * (1.0 - eps))
            nw = nh * wc[:, None]
            rows = np.repeat(np.arange(m), 6)
            cols = np.column_stack([3 * a, 3 * a + 1, 3 * a + 2,
                                    3 * b, 3 * b + 1, 3 * b + 2]).ravel()
            vals = np.column_stack([nw, -nw]).ravel()
            blocks.append(coo_matrix((vals, (rows, cols)), shape=(m, N3)).tocsr())
            deltas.append(wc * rg * nh[:, di])
        wt = np.sqrt(k * eps)
        for al in range(3):
            rows = np.repeat(np.arange(m), 2)
            cols = np.column_stack([3 * a + al, 3 * b + al]).ravel()
            vals = np.column_stack([wt, -wt]).ravel()
            blocks.append(coo_matrix((vals, (rows, cols)), shape=(m, N3)).tocsr())
            deltas.append(wt * rg if al == di else np.zeros(m))
        from scipy.sparse import vstack
        return vstack(blocks).tocsr(), np.concatenate(deltas), idx

    def _solve_eps(self, B, delta, method="direct", atol=1e-10, btol=1e-10):
        if method == "lsmr":
            return lsmr(B, -delta, atol=atol, btol=btol, maxiter=50000)[0]
        K = (B.T @ B).tocsc()
        rhs = -(B.T @ delta)
        reg = 1e-12 * K.diagonal().max()
        lu = splu(K + reg * sp_identity(K.shape[0], format="csc"))
        u = lu.solve(rhs)
        return u

    def shear_modulus(self, occ, shear=("x", "z"), kvec=None,
                      atol=1e-10, btol=1e-10, return_field=False,
                      eps=0.0, method="direct"):
        """Measure the linear-response shear modulus G for occupancy `occ`.

        occ   : (M,) boolean mask of present bonds
        shear : tuple naming the shear plane (drive, gradient); default xz.
        kvec  : (M,) per-bond stiffness (default all ones on present bonds)
        eps   : transverse/longitudinal stiffness ratio in [0, 1].  eps = 0
                (default) is the purely central-force solver
                (the original code path).  See ``_projection_eps``.
        method: solver for eps > 0, ``"direct"`` (sparse LU on normal
                equations, default) or ``"lsmr"``.
        Returns G (float).  G ~ 0 (below numerical floor) means floppy.
        """
        occ = np.asarray(occ, bool)
        if kvec is None:
            kvec = np.ones(self.M)
        drive, grad = shear
        di = {"x": 0, "y": 1, "z": 2}[drive]
        gi = {"x": 0, "y": 1, "z": 2}[grad]
        if eps < 0.0 or eps > 1.0:
            raise ValueError("eps must lie in [0, 1]")
        if eps > 0.0:
            if not occ.any():
                return 0.0
            B, delta, _ = self._projection_eps(occ, kvec, float(eps), di, gi)
            u = self._solve_eps(B, delta, method, atol, btol)
            resid = delta + B.dot(u)
            G = 2.0 * 0.5 * float(resid @ resid) / self.V
            if return_field:
                return G, u.reshape(-1, 3)
            return G

        B, idx = self._projection(occ, kvec)
        m = len(idx)
        if m == 0:
            return 0.0

        # affine extension per present bond, gamma = 1:
        #   delta_affine = gamma * r_grad * n_drive, weighted by sqrt(k)
        delta_a = (self.rvec[idx, gi] * self.rhat[idx, di]
                   * np.sqrt(kvec[idx]))

        # minimise (1/2)||delta_a + B u||^2  ->  solve B u = -delta_a (least sq)
        sol = lsmr(B, -delta_a, atol=atol, btol=btol, maxiter=20000)
        u = sol[0]
        resid = delta_a + B.dot(u)
        E_min = 0.5 * float(resid @ resid)
        G = 2.0 * E_min / self.V
        if return_field:
            return G, u.reshape(-1, 3)
        return G

    def bond_forces(self, occ, kvec=None, shear=("x", "z"), atol=1e-10, btol=1e-10):
        """Per-bond force magnitude under a fixed macroscopic affine shear strain.

        Solves the same linear-response problem as :meth:`shear_modulus` (a
        small affine shear ``gamma=1`` is applied and the non-affine
        relaxation field ``u`` is solved for), then returns the *residual*
        bond force ``|delta_affine + (B u)_bond|`` for every bond -- zero for
        absent bonds. This is the mechanical tension/compression felt by
        each occupied bond under load, and is the physical quantity that
        force-accelerated proteolysis (e.g. MMP-1 cleaving loaded collagen
        ~100x faster than unloaded collagen; Adhikari, Chai & Dunn, J. Am.
        Chem. Soc. 133(6), 1686-1689 (2011)) couples to.

        Returns
        -------
        forces : (M,) float array, 0 for unoccupied bonds.
        """
        occ = np.asarray(occ, bool)
        if kvec is None:
            kvec = np.ones(self.M)
        drive, grad = shear
        di = {"x": 0, "y": 1, "z": 2}[drive]
        gi = {"x": 0, "y": 1, "z": 2}[grad]

        B, idx = self._projection(occ, kvec)
        m = len(idx)
        forces = np.zeros(self.M)
        if m == 0:
            return forces
        delta_a = (self.rvec[idx, gi] * self.rhat[idx, di] * np.sqrt(kvec[idx]))
        sol = lsmr(B, -delta_a, atol=atol, btol=btol, maxiter=20000)
        u = sol[0]
        resid = delta_a + B.dot(u)
        forces[idx] = np.abs(resid)
        return forces

    def mean_coordination(self, occ):
        """Mean number of present bonds per node (z)."""
        occ = np.asarray(occ, bool)
        deg = np.zeros(self.N)
        np.add.at(deg, self.bonds[occ, 0], 1)
        np.add.at(deg, self.bonds[occ, 1], 1)
        return deg.mean()


    def conductance_modulus(self, occ, kvec=None, grad="z", reg=1e-12):
        """Independent scalar (random-resistor-network) reference for the
        eps = 1 limit.  For isotropic zero-rest-length springs the three
        displacement components decouple, and the sheared component solves
        the scalar problem  min_phi sum_b k_b (phi_a - phi_b + r_grad,b)^2,
        i.e. the effective conductivity of the bond network under a unit
        uniform field.  Returns G = (1/V) min_phi sum_b k_b (...)^2.
        Used only for validation of the vector solver at eps = 1.
        """
        occ = np.asarray(occ, bool)
        if kvec is None:
            kvec = np.ones(self.M)
        gi = {"x": 0, "y": 1, "z": 2}[grad]
        idx = np.flatnonzero(occ)
        if len(idx) == 0:
            return 0.0
        a, b = self.bonds[idx, 0], self.bonds[idx, 1]
        w = np.sqrt(kvec[idx])
        m = len(idx)
        rows = np.repeat(np.arange(m), 2)
        cols = np.column_stack([a, b]).ravel()
        vals = np.column_stack([w, -w]).ravel()
        A = coo_matrix((vals, (rows, cols)), shape=(m, self.N)).tocsr()
        d = w * self.rvec[idx, gi]
        K = (A.T @ A).tocsc()
        lu = splu(K + reg * K.diagonal().max() * sp_identity(self.N, format="csc"))
        phi = lu.solve(-(A.T @ d))
        r = d + A @ phi
        return float(r @ r) / self.V
