# Gel-Percolation: Percolation Inversion Dynamics in Enzymatically Degrading Wound Hydrogels

**A Topological Early Warning Framework for Predicting Fibroblast Invasion Windows**

---

## Abstract

Enzymatic degradation of wound-healing hydrogels is not a smooth, monotonic process — it is a **critical phase transition** (gel→sol inverse percolation) with universal power-law dynamics. This computational framework models that transition at the network level and makes five falsifiable predictions: (1) bulk rheology G'(ω,t) obeys critical scaling near the gel point; (2) critical slowing down produces measurable early warning signals in G'(t) before mechanical failure; (3) H₁ persistent homology loop count peaks *before* G' variance, providing an earlier rheological warning than classical EWS; (4) stiffness gradient magnitude is maximised precisely at the percolation threshold, creating the optimal window for fibroblast invasion; and (5) a "percolation handoff" metric Q quantifies whether collagen ECM percolates fast enough to support wound healing before the scaffold mechanically fails — the first such formulation-independent design target derived from first principles.

---

## Novel Scientific Claims

1. **Inverse percolation is a critical phenomenon**: G'(t) ~ |p(t) − p_c|^f·ω^Δ, with measurable critical exponents from bulk rheology
2. **Critical slowing down precedes gel–sol transition**: AR1 autocorrelation and variance of G' diverge with a computable lead time of minutes to hours
3. **H₁ topological persistence peaks before G' variance**: topological loops dissolve before bulk percolation shifts — topology is an *earlier* warning signal
4. **Stiffness gradient diverges at p_c**: optimal fibroblast seeding time is precisely the percolation edge, not the intact or failed scaffold
5. **The percolation handoff metric Q is a design target**: Q = dP∞_col/dt − dP∞_hyd/dt at t* predicts wound healing success/failure from first principles

---

## Architecture

```
                      ┌─────────────────────────────────┐
                      │     run_simulation.py            │
                      │  (CLI: single / sweep / validate)│
                      └────────────┬────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│  Module 1       │    │  Module 2           │    │  Module 3        │
│  network_model  │───▶│  mechanical_props   │───▶│  early_warning   │
│                 │    │                     │    │                  │
│ Stochastic RGG  │    │ p(t) → G'(ω,t)     │    │ AR1, Var, H₁ TDA │
│ Poisson nodes   │    │ Critical scaling    │    │ EWS lead times   │
│ MMP degradation │    │ Affine network      │    │ Persistence diagm│
└─────────────────┘    └─────────────────────┘    └──────────────────┘
          │                        │
          └──────────┐             │
                     ▼             ▼
          ┌─────────────────────────────────────┐
          │  Module 4: cell_invasion            │
          │                                     │
          │  Fibroblast agents (durotaxis +     │
          │  chemotaxis + MMP feedback)         │
          │  CollagenNetwork (growing P∞_col)   │
          │  MMP/PDGF diffusion solvers         │
          └────────────────┬────────────────────┘
                           │
                           ▼
          ┌─────────────────────────────────────┐
          │  Module 5: percolation_analysis     │
          │                                     │
          │  DualPercolationTracker             │
          │  Handoff quality Q                  │
          │  Parameter space sweep              │
          │  Critical exponent fitting          │
          └─────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/pranavkokati/gel-percolation
cd gel-percolation
pip install -r requirements.txt
pip install -e .
```

**Required:** Python ≥ 3.9, numpy, scipy, matplotlib, networkx, pandas, ripser

---

## Quick Start

```python
from src.network_model import HydrogelNetwork, HydrogelParams
from src.cell_invasion import WoundHealingSimulation, CellParams, SimParams
from src.percolation_analysis import DualPercolationTracker

params = HydrogelParams(box_size=50.0, rho_x=1.0, r_c=5.0)
net = HydrogelNetwork(params, seed=42)
sim = WoundHealingSimulation(sim_params=SimParams(n_steps=600), hydrogel_network=net)
sim.initialize()
history = sim.run()
print(f"Invasion depth: {sim.get_invasion_depth():.1f} µm")
```

---

## Running Simulations

```bash
# Single run with default parameters (30 min wound healing simulation)
python run_simulation.py --mode single

# Single run with custom parameters
python run_simulation.py --mode single --rho-x 1.5 --n-cells 30 --output results/high_density/

# 2D parameter sweep (rho_x vs k_base grid)
python run_simulation.py --mode sweep --output results/sweep/

# Validation suite against analytical limits
python run_simulation.py --mode validate

# Quick demo (100 steps, ~5 seconds)
python run_simulation.py --mode demo
```

All results are saved to `results/` including:
- `times.npy`, `p_inf_hydrogel.npy`, `p_inf_collagen.npy`, `invasion_depth.npy`
- `snapshot_final.png` — 4-panel simulation snapshot
- `dual_percolation.png` — P∞ crossover with Q annotation
- `report.txt` — human-readable summary

---

## Module Reference

**Module 1 — `src/network_model.py`**  
Builds a 3D random geometric graph (Poisson node placement, KDTree edges) and drives enzymatic degradation. Each edge is cleaved per-timestep with probability k_base · [MMP](x,t) · (L/L_mean)^α. Tracks P∞(t) and the cluster-size distribution n_s(t).

**Module 2 — `src/mechanical_properties.py`**  
Maps the local bond fraction p(x,t) to mechanical observables. Near p_c uses critical scaling G' ~ |p−p_c|^f·ω^Δ (f=2.1 for alginate, 3.75 for PEG); far above p_c uses affine rubber-elastic theory G'=ρ_chain·k_B·T. Computes spatially-resolved stiffness gradient ∇E_eff that drives durotaxis.

**Module 3 — `src/early_warning.py`** *(most novel)*  
Detects critical slowing down via rolling AR1 autocorrelation, rolling variance, and — uniquely — H₁ persistent homology loop count via Ripser. Quantifies the lead time by which topological loops dissolve before bulk G' variance diverges.

**Module 4 — `src/cell_invasion.py`**  
Agent-based fibroblast model with standard durotaxis (stiffness-guided migration toward the intact scaffold), MMP secretion feedback (high on stiff substrate), collagen deposition, and MMP/PDGF reaction-diffusion solvers. Fibroblasts follow the stiffness gradient toward the intact scaffold (standard durotaxis); MMP secretion accelerates the degradation front, creating a dynamic gradient that cells continue to chase. Builds a competing collagen percolation network as cells invade.

**Module 5 — `src/percolation_analysis.py`**  
Tracks both P∞_hydrogel and P∞_collagen simultaneously. Computes handoff time t* and quality Q. Includes 2D parameter space sweeper and critical exponent fitting.

---

## Theoretical Background

### Percolation theory
A random graph undergoes a percolation phase transition at a critical bond fraction p_c. Below p_c the network is disconnected (sol phase); above p_c a giant connected component spans the system (gel phase). The order parameter P∞ ~ |p−p_c|^β (β=0.418 for 3D). This is universally sharp — the gel point is not a gradual weakening but a critical point.

### Inverse percolation (the novel direction)
The literature studies gelation (sol→gel, p increasing). This project studies the *reverse*: enzymatic degradation drives p decreasing from 1 toward p_c. The same critical exponents apply but have never been measured in this direction for wound hydrogels.

### Critical slowing down and early warning signals
Near any continuous phase transition, the system's relaxation time diverges (critical slowing down). This manifests as increasing temporal autocorrelation (AR1 → 1) and variance before the transition. These "early warning signals" are well-established in ecology and climate science — this project is the first application to degrading biomaterials.

### Persistent homology
The H₁ Betti number counts topological 1-cycles (closed loops) in the network. As p decreases toward p_c, mesoscale pores emerge before bulk connectivity is lost — the H₁ count peaks at an earlier time than G' variance. This provides a topological early warning advantage measurable in principle from network images (cryo-SEM, confocal).

### The percolation handoff metric Q
Q = dP∞_collagen/dt|_{t*} − dP∞_hydrogel/dt|_{t*}

where t* is the crossover time when both order parameters are near their respective p_c values. Q > 0 means the collagen network percolates faster than the scaffold fails — a smooth mechanical load transfer and successful wound healing. Q < 0 predicts re-opening. Q is computable from the model and, in principle, from real-time rheology.

---

## Experimental Validation Protocol

**Rheometer (primary instrument):**

1. *Inverse percolation scaling*: Prepare hydrogels at 3 crosslink densities. Expose to collagenase. Run oscillatory time sweeps at ω = 0.1, 1, 10 rad/s simultaneously. At the gel point, fit G'(ω,t*) ~ ω^Δ — compare Δ_fit to model prediction.

2. *EWS detection*: Post-process G'(t) from the same experiments. Compute rolling AR1 and variance. Measure lead time to gel-sol transition. Compare to model.

3. *Topology validation* (if cryo-SEM available): Extract network topology from images at controlled degradation timepoints. Compute H₁ persistence directly on imaged network. Compare to simulation.

**UV-Vis (supporting):** Hydroxyproline colorimetric assay at 560 nm for collagen quantification; fluorogenic MMP-cleavable peptide release kinetics.

---

## Dependencies

| Package | Version | Use |
|---------|---------|-----|
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.10 | PDE solvers, fitting, interpolation |
| networkx | ≥3.1 | Graph construction and percolation |
| matplotlib | ≥3.7 | All visualisation |
| pandas | ≥2.0 | Parameter sweep results |
| ripser | ≥0.6.4 | Persistent homology (TDA) |
| joblib | ≥1.3 | Parallel parameter sweeps |
| PyYAML | ≥6.0 | Configuration files |
| tqdm | ≥4.65 | Progress bars |

---

## Citation

If this framework contributes to your research, please cite:

```bibtex
@software{kokati2025gelpercolation,
  author  = {Kokati, Pranav},
  title   = {Gel-Percolation: Percolation Inversion Dynamics in
             Enzymatically Degrading Wound Hydrogels},
  year    = {2025},
  url     = {https://github.com/pranavkokati/gel-percolation},
  version = {0.1.0}
}
```

---

## License

MIT License. See `LICENSE` for details.
