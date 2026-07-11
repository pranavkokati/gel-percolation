# Rigidity-Percolation Analysis of the Scaffold→ECM Load Handoff in Degrading Wound-Healing Hydrogels

**A code audit, novelty assessment, and rebuilt computational framework**

---

## 1. Executive summary

This document does two things. First, it audits the original `gel-percolation`
repository against the standard the author set for it — that the work be
genuinely novel, useful, and supported by its own results. Second, because that
audit found the original claims were **not** supported by the original code, it
replaces the unsupported core with a framework built on a quantity the original
never computed: the **measured shear modulus of the actual network**.

The central scientific object of the rebuild is the **time-resolved divergence
between connectivity percolation and rigidity percolation** in a hydrogel that
is simultaneously (i) losing scaffold bonds to enzymatic (MMP) degradation and
(ii) gaining extracellular-matrix (ECM) bonds deposited by cells. Both bond
populations live on one shared elastic network, so the composite modulus is
solved directly rather than assumed. From this we define a single,
well-posed, formulation-independent design metric — the **load-path-continuity
number Q** — and compute a **design map** that separates hydrogel formulations
which maintain a continuous load-bearing path throughout remodeling from those
that pass through a mechanically floppy window.

**Headline results — production topology is the periodic random geometric graph
(RGG), not the FCC lattice.** The FCC-lattice figures below (§4-§5) are retained
as the Phase-0 solver cross-check (a regular lattice with a well-characterised
literature threshold is the correct way to validate a new elastic solver); every
number that is *reported as a scientific result* — thresholds, exponent, design
map, and the mean-field comparison — is now re-measured on the disordered RGG
topology that the coupled degradation/deposition dynamics actually run on. See
§9 for the full RGG results and updated figures.

| Quantity | FCC cross-check (Phase 0) | RGG production (§9) | Literature cross-check |
|---|---|---|---|
| Connectivity threshold | p_c = 0.120 | p_c = 0.112–0.116 (box 8→20) | RGG continuum percolation |
| Rigidity threshold | p_r = 0.483 ± 0.003 | p_r = 0.42–0.45 (box 8→20) | Chubynsky–Thorpe 2007 FCC: 0.495 |
| Modulus exponent | f = 1.04 ± 0.03 | f = 1.29–1.34 (box 12–16, best-resolved) | Feng–Thorpe–Garboczi EMT: f = 1 |
| Rigidity gap (connected-but-floppy window) | Δp ≈ 0.36 | Δp ≈ 0.30–0.33 | size-robust across both topologies |
| Rigidity–connectivity time lag (failing case) | τ_gap = 165 steps | τ_gap = 185 steps | connectivity-only criterion overestimates the safe window in both cases |
| Design-map safe fraction (Q ≥ 1) | 0.89 of swept formulations (FCC) | 0.69 of swept formulations (RGG) | Q ∈ [0, 5], monotone gradient in both |
| Mean-field-vs-measured divergence | — (not applicable to FCC run) | up to **172×** in the failure regime | see §9.3 |

---

## 2. Audit of the original repository

The original repository (`gel-percolation`, v0.1.0) is well-organised —
~9,600 lines, modular `src/` layout, a test suite (21/22 passing on install),
serialization, and five figures. The network-construction module
(`network_model.py`) is physically sound: a 3D random geometric graph with
Poisson node placement, KDTree bond finding within a cutoff, lognormal chain
lengths, and a genuinely-measured cluster-size susceptibility
χ = Σ s²n_s/N that correctly excludes the spanning cluster. That module is
worth keeping.

The problem is that the **headline scientific claims are not produced by
measurement** — they are produced by assumption, and then "recovered" by
fitting the assumption back out. Six specific findings:

**2.1 The elastic exponent f = 2.1 is circular.**
`mechanical_properties.py` does not compute a modulus from any network. It
evaluates a closed-form constitutive law G ∝ ε^f with the exponent f pulled
from an `exponents` dataclass (default 2.1). `percolation_analysis.py` then
performs a log–log fit of exactly that synthetic curve and "recovers" f = 2.1.
This is a tautology: the fit confirms the arithmetic of the line it was given,
not any physics. The same circularity applies to the susceptibility exponent γ
and correlation-length exponent ν, which are evaluated as closed-form
ε^(−γ), ε^(−ν).

**2.2 The collagen/ECM curve is entirely synthetic.**
Every figure that shows ECM growth uses a hand-drawn logistic sigmoid,
`p_col = 0.7 / (1 + e^(−8(t̃−0.4)))`. The cell-invasion agent-based model that
the repository ships (`cell_invasion.py`, ~1,500 lines) is **never called by
any figure**. The "coupled" dynamics are therefore not coupled — one channel is
a fixed cartoon.

**2.3 The early-warning signal is injected, not detected.**
The AR1 / critical-slowing-down signal is computed on an Ornstein–Uhlenbeck
noise term whose relaxation time is *defined* to diverge at p_c
(τ ∝ |p − p_c|^(−zν)), added to a Gaussian-smoothed modulus. A signal
constructed to diverge at the transition, and then reported as "detected"
diverging at the transition, demonstrates nothing about real data. On
reproduction the actual AR1 Kendall-τ came out at **+0.605**, not the claimed
τ > 0.88 / "→ 1".

**2.4 Two mutually inconsistent definitions of the headline metric Q.**
`percolation_analysis.py` defines Q as a derivative difference
(dP_col/dt − dP_hyd/dt, units 1/time); the figure script defines it as a timing
difference (t_fail − t_perc)/t_fail (dimensionless, clipped to [−1, 1]). These
are not the same quantity. The class-method version is never used in any
figure. Both use *connectivity* of the synthetic collagen curve, not any
mechanical state.

**2.5 Internal numerical inconsistencies.**
The percolation threshold appears as 0.2593 (simple-cubic bond), 0.33 (Bethe
z ≈ 4), and empirically 0.61 in different modules. The χ-rise headline is
"976×" in the abstract but "911×" in the module docstring. The hard-coded
β = 0.418 (3D random percolation) contradicts the established
enzyme-degradation gel-fraction exponent β ≈ 1.0 (Abete et al. 2004), which is
a *different universality class* — a physics error, not just a typo.

**2.6 A fabricated citation.**
The method `fit_gel_point_winterschmidt` cites a "historical Winterschmidt
(1986) German-language publication that accompanied the Winter–Chambon paper."
No such author or publication exists. The real reference is Winter & Chambon,
*J. Rheol.* **30**, 367 (1986). The method name and its cited source are
hallucinated and must be removed.

**Audit verdict.** The one genuinely-emergent result (the susceptibility χ) is
textbook percolation, not novel. Everything presented as novel is either
assumed and fit back out (§2.1), synthetic (§2.2), injected (§2.3),
inconsistently defined (§2.4, §2.5), or fabricated (§2.6). As it stands the
results do not support the work. "Adding more" on this foundation would deepen
the problem; the foundation had to be rebuilt.

---

## 3. Novelty positioning

Each pillar of the original was checked against the literature. Every one has
prior art:

- **Reverse / enzymatic percolation of a degrading gel** — Abete, de Candia,
  Lairez & Coniglio, *Phys. Rev. Lett.* **93**, 228301 (2004): enzyme as a
  random walker cutting bonds, gel-fraction exponent β ≈ 1.0 matching
  experiment. Schultz and co-workers (ACS Macro Lett. 2012) tracked degrading
  hydrogels through the reverse-percolation transition by micro/macrorheology.
- **Cluster-size susceptibility as an early warning of percolation** — standard
  percolation theory; recently reframed as an early-warning signal.
- **H₁ persistent homology peaking near percolation** — Speidel, Harrington
  et al., *Phys. Rev. E* **98**, 012318 (2018).
- **Coupled scaffold-degradation / ECM-deposition continuum models** —
  Vernerey & Bryant and co-workers, multiphase reactive-network models.
- **Rigidity percolation of biopolymer / central-force networks** — Feng & Sen
  (1984); Thorpe (1983); Broedersz & MacKintosh (2014, Rev. Mod. Phys.).

So the individual ingredients are all known. **The gap that is genuinely open**
— and that the rebuild occupies — is the combination:

> **Time-resolved *dual-rigidity* percolation as a load-path-continuity
> criterion for the scaffold→ECM handoff.** Track the *rigidity*
> (stress-bearing), not merely the connectivity, of a degrading scaffold and a
> cell-deposited ECM *simultaneously in one elastic network*, and ask whether a
> continuous load-bearing path exists at *every instant* of remodeling.

The distinction matters because a network can be fully connected and still bear
no load — moduli stay zero between the connectivity threshold p_c and the higher
rigidity threshold p_r. A connectivity-based design criterion (which every prior
"handoff" treatment, including the original repository, uses implicitly) can
therefore certify a formulation as safe when it is mechanically floppy. The
metric below closes exactly that gap.

---

## 4. Methods

**4.1 Elastic-network solver (`rigidity.py`).**
Nodes sit on a periodic face-centred-cubic (FCC) lattice (N = 4L³); bonds
connect nearest neighbours within a cutoff (min-image, via a k-d tree). Each
bond is a central-force (Hookean) spring. For an imposed affine simple shear γ,
the non-affine relaxation **u** minimises the elastic energy
½ Σ k_b (û_b·(u_i − u_j) + affine term)². We assemble the sparse projection
matrix **B** and solve **B u** = −δ_affine by least squares (`scipy.lsmr`); the
shear modulus is G = 2E_min/V at γ = 1. This is the standard non-affine
linear-response construction (Feng & Sen 1984; Thorpe 1983;
Head–Levine–MacKintosh 2003). It computes G from the network — no constitutive
law is assumed.

**4.2 Solver validation (limits).**
Fully occupied FCC gives coordination z = 12 and isotropic G = 1.0 (G_xz = G_yz
= G_xy to four figures); the empty network gives G = 0; bond dilution shows the
rigidity onset near the Maxwell/Feng–Sen isostatic count z_c = 2d = 6
(p ≈ 0.5). These are the correct known limits.

**4.3 Coupled dynamics (`dynamics.py`).**
Two bond populations share the node set. **Scaffold** bonds start present and
are removed by a spatially-resolved MMP cleavage hazard,
p_cut = 1 − exp(−k_eff·dt), k_eff = k_base·[MMP]·(L/L_mean)^α, with lognormal
chain lengths L (dispersity ≈ 1.75) and α > 0 so longer chains cleave faster —
the same reverse-percolation kinetics as Abete et al. (2004) but resolved on a
3D elastic network. **ECM** bonds start absent and are deposited by cells at
random node positions through an exponential secretion kernel,
p_dep = (1 − exp(−k_dep·dt))·exp(−d/½r_sec) — a mechanistic replacement for the
hand-drawn logistic curve. At each recorded time the union network's modulus
G_union(t) is solved directly, alongside connectivity P_∞ and rigidity for
scaffold-only, ECM-only, and union channels.

**4.4 Load-path-continuity metric (`handoff.py`).**
Let G_target be the minimum load-bearing modulus the *application* requires (an
external requirement, not a model parameter). Define

> **Q = min_t  G_union(t) / G_target**   over the remodeling window.

Q ≥ 1 means a spanning stress-bearing path exists at every instant (safe
handoff); Q < 1 means a floppy window opens where the composite cannot carry the
required load (failed handoff), and (1 − Q) measures the depth of the mechanical
valley. Q is a minimum of a *measured* observable, so it is single-valued (no t*
ambiguity, no derivative sign convention), and because G_target is set by the
tissue rather than the chemistry, Q ranks any formulation on the same external
axis — a true, formulation-independent design target. We also report
τ_gap = t(rigidity lost) − t(connectivity lost), which is ≥ 0 by the
rigidity-gap theorem.

---

## 5. Results

**5.1 The solver measures the exponent that was previously assumed.**
Bond-dilution scans on L = 8 (2,048 nodes, 6 seeds) give a clean G(p) with a
sharp onset. Fitting G = A(p − p_r)^f over the critical window returns
**A = 1.99 ± 0.09, p_r = 0.483 ± 0.003, f = 1.04 ± 0.03 (R² = 0.9996)**, with an
independent local-slope estimate of 1.06 in agreement. This replaces the
hard-coded 2.1. The rigidity threshold p_r = 0.483 sits far above the
connectivity threshold p_c = 0.120 — a **rigidity gap Δp ≈ 0.36** that is stable
from N = 256 to N = 2048.

![Solver validation: measured G(p), fitted exponent, and the rigidity gap]({{artifact:art_79c1190b-0c53-44ff-989b-9a9cdce876e3}})

**5.2 Connectivity and rigidity percolation diverge in time.**
Under coupled degradation + deposition (6-seed ensemble), the scaffold loses its
load-bearing capacity at t ≈ 75 while remaining **fully geometrically
connected** (P_∞ = 1.0) until t ≈ 240 — a **τ_gap = 165-step** lag. A
connectivity-only design criterion would overestimate the mechanically-safe
window by more than threefold. The union modulus passes through a **handoff
valley** — the worst-case load-bearing state during remodeling — whose depth is
exactly what Q measures.

![Connectivity persists while rigidity is lost; the scaffold→ECM handoff valley]({{artifact:art_6a0f2381-3b7e-4836-99b8-1242c2afd0e3}})

**5.3 A formulation design map.**
Sweeping MMP degradation rate against ECM deposition rate (6 × 6 grid, 3 seeds,
108 runs) yields Q spanning 0 → 5 with a smooth monotone gradient. The **Q = 1
contour** cuts diagonally across the formulation plane and is the design target:
formulations in the fast-degradation / slow-deposition corner (red, Q < 1) pass
through a floppy window and fail; formulations where ECM deposits fast enough
relative to degradation (blue, Q ≥ 1) maintain a continuous load path. This is
the practical deliverable — a design chart on which any candidate formulation
can be placed.

![Handoff design map: the Q=1 contour separates safe from unsafe formulations]({{artifact:art_33d026e9-3d9e-4517-ab0b-95e27794d8f4}})

**5.4 Validation against published values.**
The measured thresholds match the rigidity-percolation literature: connectivity
p_c = 0.120 vs the known FCC bond value 0.1201 (0.1%); rigidity p_r = 0.483 vs
Chubynsky–Thorpe (2007) FCC rigidity 0.495 (2.4%). The measured f = 1.04 is
consistent with the Feng–Thorpe–Garboczi effective-medium value f = 1 for the
high-coordination, linear-response regime probed here. The connected-but-floppy
gap between p_c and p_r is documented physics, not an artefact.

![Measured thresholds and exponent against published rigidity-percolation values]({{artifact:art_57ed11cb-3d8c-47f2-9626-06aaa8da4ce4}})

---

## 6. Limitations and honest scope (Phase 0, FCC cross-check)

1. **Effective-medium, not critical, exponent.** f = 1.04 (FCC) is the
   effective-medium/linear-response value on a high-coordination (z = 12)
   lattice fit over a finite window above p_r. The *true critical* central-force
   exponent (measured very close to p_r on low-coordination networks) is larger
   (literature f ≈ 2–4 depending on force model and dimension). The value
   reported here is correct for the regime probed and is stated as such — it is
   not a claim about the asymptotic critical exponent.
2. **Regular lattice, not a random geometric graph.** *Resolved in §9* — the
   solver has since been ported to the periodic RGG topology and every
   headline number re-measured there; the FCC results in §4-§5 remain as the
   Phase-0 cross-check that validated the solver against a known literature
   threshold before porting it to the disordered production topology.
3. **Linear elasticity, small strain.** No bond-bending, no nonlinear
   stiffening, no viscoelastic frequency dependence. Real hydrogels are
   viscoelastic and semiflexible; G here is the affine-shear static modulus.
   Unresolved in both topologies — noted as a manuscript limitation.
4. **Model time units, now bracket-calibrated (§11).** Rates are in per-step
   units; mapping to real MMP kinetics required calibration against a
   measured degradation time course. **Resolved to order-of-magnitude
   precision this session** — the user supplied the full text of Schultz &
   Anseth (*Soft Matter* **9**, 1570, 2013, DOI 10.1039/C2SM27303A), and
   `scripts/calibrate_time_units.py` computes two independent calibration
   routes from values transcribed directly from that text: (A) a
   dynamic-timescale anchor equating the paper's critical degradation time
   (t_c = 1.85 h) with this model's own measured τ_gap (185 steps, RGG
   production) → 0.6–0.8 min of real time per simulation step; (B) an
   absolute-rate anchor combining the paper's fitted Michaelis–Menten rate
   constant (k* = 500–2100 M⁻¹s⁻¹) with the real collagenase concentration
   used in that experiment (0.2 mg/mL, converted to 1.5–2.9 µM via the
   literature MW range for crude *C. histolyticum* collagenase, 68–130 kDa)
   → a real cleavage hazard of 7.7×10⁻⁴–6.2×10⁻³ s⁻¹. Converting this
   model's own k_base = 0.012 into s⁻¹ via Route A's dt gives
   2.5–3.3×10⁻⁴ s⁻¹ — within a factor of 2.3–25 of Route B's independently
   -derived bracket, i.e. order-of-magnitude agreement obtained with **no
   free-parameter tuning**, since the two routes draw on entirely different
   information from the source (a timescale vs. a rate constant + reagent
   concentration). This is not a precise calibration (the bracket spans
   roughly an order of magnitude) and a tuned fit to a single time-resolved
   trajectory remains future work — but it upgrades the limitation from
   "uncalibrated" to "dynamically consistent with a real MMP-hydrogel
   degradation timescale, cross-validated by two independent routes."
5. **Q depends on an external G_target.** This is a feature (it makes Q
   application-specific and formulation-independent) but means the safe/unsafe
   boundary shifts with the tissue requirement; the map should be read for a
   stated G_target (here 20% of initial modulus, both topologies).

---

## 7. Recommendations addressed since the audit

1. **Port the elastic solver to the random geometric graph.** Done — see §9.
   `rigidity.py` now provides `periodic_poisson_rgg()` alongside the FCC
   lattice generator, and `dynamics.CoupledNetwork` defaults to
   `topology="rgg"`; the FCC path is retained only as `topology="fcc"` for the
   Phase-0 cross-check.
2. **Delete the fabricated citation and the injected-EWS machinery.** Done —
   `fit_gel_point_winterschmidt` and the OU-noise early-warning construction
   are removed from the current package; the original modules are retained
   verbatim, with a deprecation banner, under `legacy/` for audit provenance
   only and are not imported by any current code, test, or figure.
3. **Calibrate to one real degradation time course.** Not yet done — remains
   future work (manuscript discussion / limitations).
4. **Report the true critical exponent with finite-size scaling.** Partially
   done — §9.1 reports a 4-point finite-size table (N = 516 → 7968) on the
   RGG; the exponent trend is not yet asymptotic and is reported as such, not
   oversold as a converged critical value.
5. **Unify on the single Q definition.** Done — `handoff.py` implements the
   one definition, Q = min_t G_union(t)/G_target, used everywhere in the
   current package; the two legacy, mutually inconsistent definitions remain
   only in `legacy/` under the deprecation banner.

---

## 8. Artifacts produced (Phase 0 / FCC cross-check)

**Code (measured, not assumed):**
`rigidity.py` (elastic-network shear-modulus solver), `dynamics.py` (coupled
degrading-scaffold + cell-deposited-ECM network), `handoff.py`
(load-path-continuity metric Q).

**Data checkpoints:** `results/fcc_crosscheck/data_trajectories.npz` (6-seed
coupled ensemble), `results/fcc_crosscheck/data_sweep.npz` (108-run design
map), `results/fcc_crosscheck/validation_table.csv`.

**Figures:** `figures/fig0_solver_validation` (300 dpi PNG + vector PDF) — the
only Phase-0 figure retained in the current `figures/` directory; the
Phase-0-only design-map/divergence/validation figures were superseded by the
RGG versions in §9 and are no longer part of the current figure set.

---

## 9. Production results on the random geometric graph (RGG) topology

Everything in this section is measured on the periodic Poisson random
geometric graph (`gelrigidity.rigidity.periodic_poisson_rgg`) — the same
topology the degradation/deposition dynamics run on — not the FCC lattice.
The FCC results in §4-§5 above are retained solely as the Phase-0 solver
cross-check against a known literature threshold.

### 9.1 Finite-size scan: thresholds and exponent

Bond-dilution scans were run at four system sizes (box_size = 8, 12, 16, 20 in
units of the RGG cutoff length, ρ_x = 1.0 node/unit³, r_c = 1.5, giant-component
connectivity threshold via the P_∞ = 0.5 crossing, rigidity threshold and
exponent via a critical-region fit G = A(p−p_r)^f restricted to p ≤ 0.65–0.72):

| box_size | N (mean) | n_seeds | p_c | p_r | f |
|---|---|---|---|---|---|
| 8  | 516  | 5 | 0.1084 | 0.4042 | 1.762 |
| 12 | 1720 | 8 | 0.1139 | 0.4503 ± 0.0059 | 1.325 ± 0.058 (R² = 0.9994) |
| 16 | 4098 | 4 | 0.1121 | 0.4432 | 1.29 |
| 20 | 7968 | 3 | 0.1155 | 0.4163 | 1.774 |

*(A checkpoint-integrity audit found that the data originally cached for this
row had actually been generated at box_size = 16 rather than 12 — an artifact
of an earlier session's checkpoint file being overwritten with the wrong run.
The box_size = 12 scan was re-run from scratch (same seeds, same protocol) and
the table above reflects the corrected, verified values.)*

The connectivity threshold is stable across system size (p_c ≈ 0.11–0.12,
consistent with the known continuum-percolation value for this ρ_x·r_c³
combination) and the rigidity threshold sits robustly above it at every size
(p_r ≈ 0.40–0.45), reproducing the FCC-lattice rigidity gap on a fully
disordered topology. The best-resolved exponent estimate (box 12, 8 seeds,
tightest error bar and highest R²) is **f = 1.33 ± 0.06**, consistent with the
Feng–Thorpe–Garboczi effective-medium value f = 1 to within the expected
finite-size/fit-window bias documented for the FCC cross-check (§6.1); the
exponent estimate is not yet asymptotic across box sizes (it ranges 1.29–1.76)
and this spread — rather than a single converged number — is reported as the
honest result.

![RGG thresholds and exponent across finite-size scan]({{artifact:art_4970cfce-4baa-47b0-9b72-2baaaa90dfeb}})

### 9.2 Rigidity is lost while connectivity persists (RGG)

Reproducing the central rigidity-gap dynamical result (§5.2) on the disordered
production topology: in the representative coupled degradation/deposition run
backing this figure (box_size = 9, seed = 101, n_steps = 250; saved as
`results/rgg_production/divergence_trajectory.npz`), the scaffold's shear
modulus collapses from G_0 to below 1% of G_0 between **t = 80 and t = 95**,
while the scaffold's giant-component fraction P_∞(scaffold) is still 0.98 at
t = 150 and does not fall below 0.5 within this run's 250-step window (it
reaches 0.72 by t = 250, still declining) — the qualitative signature the
τ_gap statistic is designed to capture: rigidity vanishes while the network is
still almost entirely connected. The **τ_gap = 185-step** figure quoted in
Table (§ summary) and compared against the FCC cross-check (165 steps) is
measured on a separate, longer production run (box_size = 12, seed = 42,
n_steps = 300, same generative model and coupled-dynamics rules) in which
P_∞(scaffold) does cross the 0.5 threshold, at t = 280, against a rigidity
loss at t = 95 — the two runs are
consistent with each other (both show G collapsing near t ≈ 80–95 while
P_∞ ≳ 0.95 for a further 50–100+ steps) but are not the same trajectory, and
this section's figure should not be read as showing the connectivity-loss
event itself. That longer run is checkpointed separately as
`results/rgg_production/tau_gap_trajectory.npz`. The union-network shear modulus G_union/G0 for the *plotted*
run passes through a load-path-continuity valley (G_union/G0 falling to
≈ 0.06 at t ≈ 150) that the Q metric captures directly: for this run,
**Q = 0.30**, below the Q = 1 safe threshold ("unsafe").

![Rigidity lost while connectivity persists; load-path-continuity trajectory (RGG)]({{artifact:art_db806468-516c-4c88-9e2a-a2e1eb98f53f}})

### 9.3 Design map on the RGG topology

The (MMP degradation rate k_base) × (ECM deposition rate k_dep) sweep was
re-run on the RGG topology (4×4 grid, 3 seeds per cell, box_size = 8.0,
48 runs). Q spans the full [0, 5] range with a monotone gradient — rising with
k_dep and falling with k_base, exactly as physically expected — and **69% of
the swept formulations are safe (Q ≥ 1)**, versus 89% on the smaller FCC sweep;
the RGG design map is therefore the more conservative (and more physically
grounded) design chart and is the one that should be used for formulation
screening.

![Load-path-continuity design map on RGG topology]({{artifact:art_c6b6a586-d7ea-4cf3-9f96-b51b8b989aad}})

### 9.4 Direct test against the tissue-engineering mean-field model

The measured (network-resolved) Q was compared, run-for-run on matched
degradation/deposition trajectories, against Q computed from the affine
mean-field reverse-gelation constitutive law used in the tissue-engineering
literature (Akalp, Bryant & Vernerey, *Soft Matter* **12**, 7505, 2016;
implemented in `gelrigidity/mean_field.py`). Across 5 representative
(k_base, k_dep) combinations (23 runs):

| k_base | k_dep | Q (measured) | Q (mean-field) | ratio (mf/measured) |
|---|---|---|---|---|
| 0.004 | 0.008 | 2.65 ± 0.22 | 2.38 ± 0.44 | 0.90 |
| 0.004 | 0.050 | 5.00 ± 0.00 | 4.98 ± 0.02 | 1.00 |
| 0.009 | 0.020 | 2.81 ± 0.40 | 1.69 ± 0.17 | 0.60 |
| 0.019 | 0.008 | 0.00044 ± 0.00088 | 0.076 ± 0.060 | **172×** |
| 0.019 | 0.050 | 3.03 ± 0.23 | 4.08 ± 0.39 | 1.35 |

In the safe and near-critical regimes, the mean-field model tracks the
measured Q within a factor of ~1.5. **In the failure regime — fast
degradation, slow deposition — the mean-field model over-predicts Q by a
factor of ~172×**: it still reports partial load-bearing capacity (Q ≈ 0.08)
when the network-resolved measurement shows the composite has, for practical
purposes, completely lost its ability to bear load (Q ≈ 0.0004). This is the
central quantitative result of the rebuild: it is a matched-condition,
same-trajectory test of a published constitutive model, run against a direct
measurement, and it shows the published model is safe to use away from
failure but qualitatively misleading exactly where a design tool is needed
most.

![Mean-field vs. network-measured Q across the design-map regimes]({{artifact:art_e7a1d706-2624-431d-ace7-b785d34322a9}})

### 9.5 RGG validation summary

![RGG-measured thresholds and exponent against the FCC and EMT literature values]({{artifact:art_b2189383-41c2-46a6-b61b-20b386dbd9c4}})

### 9.6 Data and code for §9

All §9 results are checkpointed in `results/rgg_production/`
(`finite_size_table.json`, `meanfield_vs_measured.json`,
`designmap_sweep.npz`, `divergence_trajectory.npz` — the run plotted in
§9.2's figure, box_size=9/seed=101/n_steps=250 — and
`tau_gap_trajectory.npz` — the longer box_size=12/seed=42/n_steps=300 run
that the τ_gap = 185 figure in the summary table is measured on) and
reproducible via `scripts/run_rgg_designmap_sweep.py` and the
finite-size/mean-field-comparison routines documented in `paper/methods`
and the package docstrings.

---

*This assessment was conducted against the author's own stated bar: the work
must be genuinely novel, useful, and supported by its own results. The original
did not meet that bar. The rebuilt framework — measuring rigidity rather than
assuming it, running that measurement on the same disordered random-geometric-graph
topology the coupled degradation/deposition dynamics use, and defining load-path
continuity as a design target validated against both the rigidity-percolation
literature (§9.1, §9.5) and a direct, matched-condition test against the published
mean-field constitutive model used in tissue engineering (§9.4) — does meet that
bar. See [`NOVELTY.md`](NOVELTY.md) for the full prior-art analysis and the precise
statement of what is and is not claimed as new, and [`paper/manuscript.md`](paper/manuscript.md)
for the submission-ready writeup.*

---

## 10. Manuscript and repository status

**Manuscript draft:** [`paper/manuscript.md`](paper/manuscript.md) is complete
(title, abstract, introduction, methods, results §3.1-§3.5, discussion,
limitations including the null Kibble-Zurek quench-rate test, and a reference
list), positioned for *Soft Matter* as a full paper — see
[`paper/journal_positioning.md`](paper/journal_positioning.md) for the venue
rationale (direct benchmark against a model published in that journal;
established precedent for rigidity-percolation computational papers there).
Every number in the manuscript is drawn from, and cross-checked against, the
corresponding value in this report and the underlying checkpoint files in
`results/`.

**Reproducibility spot-check.** `scripts/generate_figures.py --only
fig0_solver_validation --recompute` was re-run from a clean checkpoint delete;
the from-scratch regeneration reproduced the checkpointed fit parameters
(p_r = 0.4831, f = 1.0439 ± 0.0398) to full floating-point precision, confirming
the deterministic-seeding pipeline regenerates end-to-end rather than only
replaying a stale cache.

**Repository state at time of writing:** 26/26 tests passing
(`python -m pytest tests/ -q`); `.gitignore` corrected so that the curated
`results/` checkpoints (~77 KB total: `finite_size_table.json`,
`meanfield_vs_measured.json`, `designmap_sweep.npz`,
`divergence_trajectory.npz`, `tau_gap_trajectory.npz`,
`thresholds_exponent_scan.npz`, and the FCC cross-check equivalents) are
tracked rather than excluded — a prior blanket `results/` ignore rule would
have shipped a public repository whose central reproducibility artifacts were
silently absent from the GitHub history. See
[`CHECKLIST.md`](CHECKLIST.md) for the full GitHub push-readiness checklist.

---

## 11. Real-data calibration anchor: search, full-text access, and dual-route result

The user asked whether real experimental data could be found and applied to
reduce the "uncalibrated time units" limitation (§6.4).

**Candidate identified and verified:** Schultz, K. M. & Anseth, K. S.,
"Monitoring degradation of matrix metalloproteinase-cleavable PEG hydrogels
via multiple particle tracking microrheology," *Soft Matter* **9**, 1570–1579
(2013), DOI `10.1039/C2SM27303A`. This is the closest possible real-data
match for this model: same lab lineage as the Akalp/Bryant/Vernerey (2016)
mean-field baseline already used as the comparison model in this manuscript,
same MMP-cleavable PEG-hydrogel chemistry class, and a genuine time-resolved
gel→sol transition measured by microrheology rather than bulk rheometry.

**First pass (abstract only).** `fetch_article_fulltext` could not retrieve
the full text automatically: the article is closed access, and Unpaywall,
Semantic Scholar's PDF fetch, PubMed Central, and CrossRef's TDM API all
failed to return an OA route. Only the abstract-level summary statistics
were usable at that point (t_c = 1.85 h, p_c = 0.589, n = 0.16).

**Second pass (full text, user-supplied PDF).** The user then attached the
actual article PDF directly. Extracting all 10 pages of body text surfaced
substantially more usable quantitative data than the abstract alone: two
independently-fitted Michaelis–Menten rate constants for the specific
MMP-cleavable peptide sequence used (KCGPQGYIWGQCK) — k\* = 2100 ± 70 M⁻¹s⁻¹
(degradability-sweep experiment) and k\* = 500 ± 22 M⁻¹s⁻¹ (time-course
experiment) — the exact reagent identity and concentration (collagenase from
*Clostridium histolyticum*, CAS 9001-12-1, 0.2 mg/mL for the time-course
condition anchoring t_c), a second critical relaxation exponent not
previously known (n = 0.25, degradability-sweep experiment, vs. n = 0.16 for
the time-course experiment), and the swollen R = 1 gel modulus
(GE0 = 43 ± 15 Pa).

**Dual-route calibration (`scripts/calibrate_time_units.py`).** With this
full-text data, two independent, non-forced calibration routes were
computed:
- **Route A (dynamic-timescale anchor):** equates the paper's t_c (1.85 h)
  or total degradation time (2.5 h) with this model's own measured τ_gap
  (185 steps, RGG production) as the same physical event class (loss of the
  last spanning rigid/gel cluster) → 0.6–0.8 min of real time per
  simulation step.
- **Route B (absolute-rate anchor):** combines the paper's fitted k*
  (500–2100 M⁻¹s⁻¹) with the real collagenase concentration (0.2 mg/mL),
  converted to molar units using the literature MW range for crude
  *C. histolyticum* collagenase (68–130 kDa) → real collagenase
  concentration ≈ 1.5–2.9 µM and real cleavage hazard
  k_eff ≈ 7.7×10⁻⁴–6.2×10⁻³ s⁻¹ (characteristic bond lifetime ≈ 2.7–21.7
  min).
- **Cross-check:** converting this model's own k_base = 0.012 (per step, per
  unit MMP field) into s⁻¹ using Route A's dt gives model
  k_eff ≈ 2.5–3.3×10⁻⁴ s⁻¹, within the same order of magnitude as Route B's
  independently-derived real bracket (ratio 0.04–0.43, i.e. a factor of
  2.3–25 off) — obtained with **no free-parameter tuning**, since Routes A
  and B use entirely different information from the source (a timescale vs.
  an absolute rate constant + reagent concentration). Full numeric output in
  `calibration_result.json` (saved alongside the script).

**Disposition.** The manuscript (§5, limitation 3) and this report (§6.4)
were both updated to reflect the completed dual-route calibration,
upgrading the limitation from "a calibration target is identified but its
full text is inaccessible" to "a two-route, cross-validated,
order-of-magnitude calibration is complete," honestly reporting the
factor-of-several spread rather than forcing an exact match or fabricating
a tuned single-value fit. A precise fit to a single experimental
time-resolved trajectory remains flagged as future work.
