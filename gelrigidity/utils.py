"""Shared utilities for the gel-percolation simulation framework."""

import logging
import os
import pickle
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Configure and return the gel_percolation logger.

    Args:
        level:    Logging level string ('DEBUG', 'INFO', 'WARNING', etc.).
        log_file: Optional path to a file handler; output is tee'd there as well.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger("gel_percolation")
    logger.setLevel(getattr(logging, level.upper()))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def set_random_seed(seed: int) -> None:
    """Set the global random seeds for both numpy and the stdlib random module.

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)
    random.seed(seed)


class ProgressTimer:
    """Context manager that logs the wall-clock elapsed time on exit.

    Usage::

        with ProgressTimer("my-step", logger=my_logger):
            do_work()

    If *logger* is None the elapsed time is printed to stdout.

    Args:
        label:  Human-readable label included in the log / print message.
        logger: Optional :class:`logging.Logger`.  When provided the message is
                emitted at INFO level; otherwise it is sent to ``print``.
    """

    def __init__(self, label: str, logger: logging.Logger = None) -> None:
        self.label = label
        self.logger = logger
        self._t0: float = 0.0

    def __enter__(self) -> "ProgressTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.perf_counter() - self._t0
        msg = f"{self.label} completed in {elapsed:.4f}s"
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)
        # Do not suppress exceptions.
        return False


def interpolate_field(
    field: np.ndarray,
    positions: np.ndarray,
    grid_coords: np.ndarray,
) -> np.ndarray:
    """Trilinear interpolation of a 3D scalar field at arbitrary positions.

    Args:
        field:       3-D array of shape ``(Nx, Ny, Nz)``.
        positions:   Array of shape ``(M, 3)`` containing the query points
                     ``(x, y, z)`` in the same coordinate system as
                     *grid_coords*.
        grid_coords: Tuple/list ``(x_grid, y_grid, z_grid)`` where each
                     element is a strictly increasing 1-D array that spans the
                     corresponding axis of *field*.

    Returns:
        1-D array of shape ``(M,)`` with the interpolated values.
    """
    from scipy.interpolate import RegularGridInterpolator

    x, y, z = grid_coords
    interp = RegularGridInterpolator(
        (x, y, z),
        field,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return interp(positions)


def gradient_field_3d(field: np.ndarray, dx: float) -> np.ndarray:
    """Central-difference gradient of a 3-D scalar field.

    Uses :func:`numpy.gradient`, which applies second-order central differences
    on interior points and first-order differences at the boundaries.

    Args:
        field: 3-D array of shape ``(Nx, Ny, Nz)``.
        dx:    Uniform grid spacing (same in all three directions).

    Returns:
        Array of shape ``(Nx, Ny, Nz, 3)`` where the last axis holds
        ``(dF/dx, dF/dy, dF/dz)`` at every voxel.
    """
    grad = np.empty(field.shape + (3,), dtype=float)
    grad[..., 0] = np.gradient(field, dx, axis=0)
    grad[..., 1] = np.gradient(field, dx, axis=1)
    grad[..., 2] = np.gradient(field, dx, axis=2)
    return grad


def compute_giant_component_fraction(graph: nx.Graph) -> float:
    """Return the fraction of nodes that belong to the largest connected component.

    Args:
        graph: An undirected :class:`networkx.Graph`.

    Returns:
        ``|GCC| / |V|``.  Returns 0.0 for an empty graph.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    largest = max(nx.connected_components(graph), key=len)
    return len(largest) / n


def poisson_point_process_3d(
    density: float,
    box_size: float,
    seed: int = None,
) -> np.ndarray:
    """Draw points from a homogeneous Poisson point process in 3-D.

    The number of points is sampled from a Poisson distribution with mean
    ``density * box_size**3``; the points themselves are drawn uniformly
    from the cube ``[0, box_size]^3``.

    Args:
        density:  Number density of points (points per unit volume).
        box_size: Edge length of the cubic domain.
        seed:     Optional integer seed for reproducibility.

    Returns:
        Float array of shape ``(N, 3)`` where *N* is Poisson-drawn.
    """
    rng = np.random.default_rng(seed)
    expected_n = density * box_size ** 3
    n = int(rng.poisson(expected_n))
    return rng.uniform(0.0, box_size, size=(n, 3))


def lognormal_chain_lengths(
    n: int,
    mu_L: float,
    sigma_L: float,
    seed: int = None,
) -> np.ndarray:
    """Sample polymer chain lengths from a log-normal distribution.

    The parameters *mu_L* and *sigma_L* are the mean and standard deviation of
    the underlying normal distribution (i.e. the parameters passed to
    ``np.random.lognormal``).

    To convert from desired linear-space moments to log-space parameters::

        sigma_L = sqrt(log(1 + (L_std / L_mean) ** 2))
        mu_L    = log(L_mean) - sigma_L ** 2 / 2

    Args:
        n:       Number of samples.
        mu_L:    Mean of the underlying normal (log-space mean).
        sigma_L: Standard deviation of the underlying normal (log-space sigma).
        seed:    Optional integer seed.

    Returns:
        1-D float array of shape ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    return rng.lognormal(mu_L, sigma_L, size=n)


def build_random_geometric_graph_3d(
    positions: np.ndarray,
    r_c: float,
) -> nx.Graph:
    """Construct a 3-D random geometric graph efficiently using a KD-tree.

    Two nodes are connected by an edge if and only if their Euclidean distance
    is at most *r_c*.

    Args:
        positions: Float array of shape ``(N, 3)`` giving node coordinates.
        r_c:       Connection radius (cutoff distance).

    Returns:
        :class:`networkx.Graph` with

        * node attribute ``"pos"`` — ``(3,)`` position array for each node;
        * edge attribute ``"distance"`` — Euclidean distance between the two
          endpoints.
    """
    n = len(positions)
    tree = KDTree(positions)
    # query_pairs returns all pairs (i, j) with i < j and dist(i,j) <= r_c.
    pairs = tree.query_pairs(r_c, output_type="ndarray")

    G = nx.Graph()
    G.add_nodes_from(range(n))
    nx.set_node_attributes(G, {i: positions[i] for i in range(n)}, "pos")

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        diffs = positions[i_idx] - positions[j_idx]
        distances = np.linalg.norm(diffs, axis=1)
        for k, (i, j) in enumerate(pairs):
            G.add_edge(int(i), int(j), distance=float(distances[k]))

    return G


def save_checkpoint(state: dict, filepath: str) -> None:
    """Serialise *state* to *filepath* using :mod:`pickle`.

    Parent directories are created automatically if they do not exist.

    Args:
        state:    Arbitrary Python dictionary to serialise.
        filepath: Destination file path (string or path-like).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(filepath: str) -> dict:
    """Deserialise a checkpoint previously written by :func:`save_checkpoint`.

    Args:
        filepath: Path to the pickle file.

    Returns:
        The deserialised state dictionary.
    """
    with open(filepath, "rb") as fh:
        return pickle.load(fh)


def ensure_dir(path: str) -> str:
    """Create *path* and all intermediate directories if they do not exist.

    Equivalent to ``mkdir -p`` but idempotent.

    Args:
        path: Directory path to create.

    Returns:
        The same *path* string (for convenient chaining).
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
