# gelrigidity: Dual-Rigidity-Percolation Load-Path-Continuity Model for Enzymatically Degrading Hydrogel Wound Scaffolds

**A finite-rate, network-resolved rigidity-percolation framework quantifying whether a degrading hydrogel scaffold hands off mechanical load to a growing ECM network fast enough to prevent wound re-opening.**

---

## What this is

A wound-healing hydrogel scaffold is progressively cleaved by matrix metalloproteinases (MMPs) while fibroblasts simultaneously deposit new collagen extracellular matrix (ECM) within it. The scaffold's ability to bear mechanical load is not the same thing as its connectivity: a network can remain a single geometrically spanning cluster while carrying essentially zero shear stress, because rigidity percolation (the threshold at which a spanning set of bonds first constrains all non-trivial deformations) sits at a strictly higher bond-occupation fraction than connectivity percolation. This package:

1. Builds the scaffold and ECM as **periodic random geometric graphs (RGG)** — homogeneous Poisson node placement with edges wherever two nodes fall within a cutoff radius — matching the disordered topology of a real crosslinked hydrogel (no ordered lattice in the production model).
2. **Measures** (never assumes) the shear modulus G(p) of a given bond-occupation network via a non-affine linear-response elastic solve, so the rigidity-percolation exponent and threshold are outputs, not hardcoded inputs.
3. Runs **coupled scaffold-degradation + ECM-deposition dynamics** on a single shared network substrate, tracking the union network's shear modulus G_union(t) through the handoff window.
4. Defines a single, dimensionally consistent, formulation-independent design metric — **Q = min_t G_union(t) / G_target** — quantifying whether the composite network sustains the physiologically required load-bearing modulus at every instant of the remodeling window.
5. Investigates whether the *rate* at which bonds are removed (not merely the total fraction removed) sets a finite-rate, Kibble-Zurek-type freeze-out scale for the mechanical response, when cleavage kinetics are coupled back to the locally resolved bond stress (force-accelerated proteolysis).

---

## Repository provenance and audit

This repository began life as a public codebase making several specific, checkable claims (universal critical scaling of G'(ω,t), a percolation-susceptibility early-warning signal, a topological lead indicator, and a formulation-independent "percolation handoff" design metric). A line-by-line pre-publication audit found that several of the original implementation's central results were **circular or synthetic**:

- The rigidity-percolation exponent `f` was hardcoded into a closed-form constitutive law and then "recovered" by fitting the same law to its own output.
- The collagen/ECM percolation curve was a hand-drawn logistic sigmoid, never actually driven by the agent-based cell-invasion model it was said to summarize.
- The early-warning Kendall-tau signal was produced by an Ornstein-Uhlenbeck noise term engineered to diverge at the percolation threshold, not derived from the network dynamics.
- Two different, dimensionally inconsistent definitions of the "handoff quality" Q coexisted in the code.

**Full findings, every measured number, and the literature cross-checks are in [`REPORT.md`](REPORT.md).** The original flagged modules are retained, unmodified except for an explicit deprecation banner, under [`legacy/`](legacy/) for provenance and audit-trail purposes only — **they are not imported by any current code, test, or figure**. The current package, `gelrigidity/`, replaces every flagged component with a measured (not closed-form) elastic solver, network-driven (not synthetic) ECM growth, and one unified Q definition.

---

## Package layout

```
gelrigidity/            Public package — the only code path used by tests/figures/manuscript
    network.py           HydrogelNetwork / HydrogelParams — RGG scaffold + MMP-cleavage kinetics generator
    rigidity.py           ElasticNetwork, periodic_poisson_rgg, fcc_lattice, neighbour_bonds
                          non-affine central-force elastic solver; MEASURES G(p) and the
                          rigidity-percolation exponent from the bond network directly
    dynamics.py           CoupledNetwork — couples scaffold degradation + ECM deposition on one
                          shared periodic RGG substrate; optional force-coupled ("stress-guided")
                          cleavage kinetics for the finite-rate / Kibble-Zurek study
    handoff.py            load_path_continuity_Q, rigidity_connectivity_lag, summarize_trajectory
                          the single, unified Q = min_t G_union(t)/G_target definition
    utils.py              shared numerical utilities

tests/                  pytest suite for gelrigidity/ (26 tests; see below)
scripts/                Figure-generation and analysis entry points for the current package
figures/                Publication figures generated from RGG results (fig0, fig_divergence,
                        fig_designmap, fig_validation, plus the Kibble-Zurek scaling figure)
paper/                  Manuscript draft (title/abstract/intro/methods/results/discussion)
legacy/                 Original flagged modules + their tests/scripts/figures, deprecated,
                        retained for audit provenance only — NOT part of the public API
REPORT.md               Full audit trail: original-repo findings, literature validation,
                        methodology for every measured number in this repository
CITATION.cff            Machine-readable citation metadata
environment.yml          Conda environment specification
setup.py / requirements.txt   Package metadata and pinned dependencies
```

---

## Installation

```bash
git clone https://github.com/pranavkokati/gel-percolation
cd gel-percolation
conda env create -f environment.yml
conda activate gelrigidity
pip install -e .
pytest tests/ -v
```

**Core dependencies:** Python >= 3.10, numpy, scipy, matplotlib, networkx, pandas, joblib, PyYAML, tqdm. (`ripser`/`persim`/`gudhi`/`mesa`/`seaborn` are optional, legacy-only dependencies — see `requirements.txt`.)

---

## Quick start

```python
from gelrigidity.rigidity import periodic_poisson_rgg, ElasticNetwork
from gelrigidity.dynamics import CoupledNetwork
from gelrigidity.handoff import summarize_trajectory

import numpy as np

# Measure G(p) directly on a periodic RGG (no closed-form constitutive law)
pos, box, bonds, rhat, rvec = periodic_poisson_rgg(rho_x=1.0, box_size=12.0, r_c=1.5, seed=0)
net = ElasticNetwork(pos, box, bonds, rhat, rvec)
occ = np.ones(net.M, dtype=bool)   # boolean bond-occupancy mask, e.g. a random dilution
G = net.shear_modulus(occ)

# Coupled scaffold-degradation + ECM-deposition dynamics on the RGG topology
cn = CoupledNetwork(topology="rgg", rho_x=1.0, box_size=12.0, r_cut=1.5, seed=0)
cn.seed_scaffold(p0=1.0)
cn.seed_cells(n_cells=20, secretion_radius=2.0)
rec = cn.run(n_steps=300, k_base=0.012, k_dep=0.010)

# The single, unified handoff design metric
summary = summarize_trajectory(rec, G_target=0.1)
print(summary["Q"], summary["safe"])
```

---

## Central physical results (RGG topology; see REPORT.md §9 / NOVELTY.md / paper/ for full detail)

- **Rigidity threshold sits strictly above the connectivity threshold** on the disordered RGG (p_r ≈ 0.40-0.45 across box sizes 8-20 [0.4042, 0.4503, 0.4432, 0.4163], vs. connectivity p_c ≈ 0.11-0.12), giving a "floppy but connected" window with zero shear modulus. This is the physical mechanism underlying every downstream result.
- **The rigidity-percolation exponent is measured, not assumed**: the best-resolved fit (box_size=12, 8 seeds, N≈1720) gives p_r=0.4503±0.0059, f=1.325±0.058 (R²=0.9994, critical-region fit p≤0.72), consistent with the Feng-Thorpe-Garboczi effective-medium value f≈1 within the finite-size/fit-window bias documented in REPORT.md; the exponent is not yet asymptotic across box sizes (range 1.29-1.76 over box 8-20) and this spread is reported honestly rather than rounded to a single value.
- **The scaffold loses rigidity 165-185 timesteps before it loses connectivity** in coupled degradation/deposition runs, reproduced on both the ordered FCC cross-check lattice (tau_gap=165) and the disordered production RGG topology (tau_gap=185) — confirming the effect is a property of rigidity percolation itself, not an artifact of lattice order.
- **The Q design map** over (MMP degradation rate x ECM deposition rate), swept on the RGG topology (4x4 grid, 3 seeds/cell, 48 runs), identifies the safe/unsafe boundary for load-path continuity directly from measured mechanics: 69% of swept formulations are safe (Q≥1), with a monotone gradient rising with ECM deposition rate and falling with MMP degradation rate.
- **Direct, matched-condition test against the tissue-engineering field's own mean-field model** (Akalp, Bryant & Vernerey, *Soft Matter* 2016) shows the mean-field prediction tracks the network-measured Q within ~1.5x in safe/near-critical regimes, but over-predicts Q by **up to 172x** in the failure regime (fast degradation, slow deposition) — exactly where a design tool matters most.

---

## Running the analysis

```bash
# Full test suite
pytest tests/ -v

# Regenerate all current-package figures from the checkpointed result arrays
# under results/ (fast: seconds, no hardcoded numbers -- every panel is
# recomputed from the saved raw arrays each run)
python scripts/generate_figures.py

# Regenerate one or more figures only
python scripts/generate_figures.py --only fig_rgg_thresholds_exponent fig_rgg_divergence

# Regenerate from scratch (re-runs the underlying rigidity/dynamics scans;
# slow -- the box=20 finite-size point alone takes ~15-20 min single-core).
# fig_meanfield_comparison has no from-scratch path yet: it is checkpoint-only
# (see scripts/generate_figures.py module docstring) and raises with --recompute.
python scripts/generate_figures.py --recompute
```

---

## Citation

If this framework contributes to your research, please cite this repository (see [`CITATION.cff`](CITATION.cff)) and the manuscript draft in `paper/` once published.

```bibtex
@software{kokati_gelrigidity,
  author  = {Kokati, Pranav},
  title   = {gelrigidity: a dual-rigidity-percolation load-path-continuity
             model for enzymatically degrading hydrogel wound scaffolds},
  year    = {2026},
  url     = {https://github.com/pranavkokati/gel-percolation},
  version = {0.2.0}
}
```

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.
