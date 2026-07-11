# Novelty and Prior-Art Memo

**Subject:** dual-rigidity-percolation load-path-continuity model for the scaffold→ECM
mechanical handoff in enzymatically degrading hydrogel wound scaffolds.
**Scope:** this memo documents the literature search underlying the novelty claim in
`REPORT.md` §3 and the manuscript (`paper/manuscript.md`) introduction, with the specific
prior-art items checked against each pillar of the model, and states precisely what is
and is not new about this work.

---

## 1. The claim, stated precisely

The novel contribution is **not** any single ingredient below — each of those has
prior art, cited in full. The novel contribution is the specific combination:

> A network-resolved elastic solver (not a closed-form constitutive law) that
> **measures** the shear modulus of a randomly geometric-graph hydrogel network
> undergoing **simultaneous, kinetically independent** bond removal (MMP scaffold
> degradation) and bond addition (cell-deposited ECM) on **one shared node set**,
> from which a single design scalar — **Q = min_t G_union(t)/G_target** — is
> computed directly from the measured mechanics, and which is shown, by direct
> head-to-head computation against the standard mean-field/affine reverse-gelation
> constitutive law used in the tissue-engineering literature, to diverge from that
> mean-field prediction by up to two orders of magnitude in the regime that matters
> most (near-failure formulations).

Each clause below is checked against the closest prior art found.

---

## 2. Prior art, pillar by pillar

### 2.1 Rigidity percolation as a distinct, higher threshold than connectivity percolation

This is **established physics**, not new. Feng & Sen (1984) first showed elastic
(central-force) percolation is a distinct universality class from scalar connectivity
percolation; Thorpe (1983) developed the constraint-counting (rigidity) theory;
Jacobs & Thorpe (1996) gave the 2D generic rigidity algorithm; Chubynsky & Thorpe
(2007, *Phys. Rev. E* **76**, 041135) computed the FCC-lattice rigidity threshold this
work validates against (0.495, vs. our measured 0.483-0.443 depending on topology and
system size). Effective-medium behavior for the modulus exponent near the isostatic
point is standard (Feng-Thorpe-Garboczi theory, and the review by Broedersz &
MacKintosh, *Rev. Mod. Phys.* **86**, 995 (2014)).

**What is new here:** applying this well-established distinction as a *design
criterion timed against a competing bond-addition process*, rather than as a static
material property of a single network.

### 2.2 Non-affine elastic-network solvers on random geometric graphs / off-lattice networks

Also established. Off-lattice, non-affine linear-response solvers for measuring the
shear modulus of disordered spring networks are used throughout the biopolymer-network
literature (Head, Levine & MacKintosh 2003; Broedersz et al. 2011, *Nat. Phys.* **7**,
983; the "Mikado model" for 2D fiber networks). Methodologically adjacent and recent:
a November 2024 preprint (arXiv:2411.14159, "Predicting rigidity and connectivity
percolation in disordered particulate networks using graph neural networks", later
published) trains graph neural networks to predict rigidity/connectivity percolation
class directly from off-lattice spring-network graphs. Rigidity there is determined
from the viscoelastic shear response of the network at low frequency and identified
with a non-zero storage modulus — methodologically the same family of measurement as
the solver in `rigidity.py`, but that work predicts percolation class with an ML
surrogate on static (non-evolving) networks; it does not model a time-dependent
degrading/depositing scaffold or define a handoff design metric.

Random-geometric-graph continuum percolation itself (nodes placed by a Poisson process,
edges within a cutoff radius) is a standard construction (Chattterjee & Grimaldi,
*Phys. Rev. E* **92**, 032121 (2015), for rod systems; used generically in continuum
percolation theory).

**What is new here:** porting the measured (not closed-form) elastic solver to the
same periodic-Poisson RGG topology used for the dynamical scaffold/ECM model, so that
the threshold, exponent, and time-resolved modulus are all measured on the *same*
disordered network that the degradation/deposition kinetics run on — eliminating the
lattice-vs-disorder mismatch that the original repository's audit (§2, `REPORT.md`)
flagged as a limitation of the FCC-only validation.

### 2.3 Reverse percolation / enzymatic degradation of gels

Established. Abete, de Candia, Lairez & Coniglio (*Phys. Rev. Lett.* **93**, 228301,
2004) modeled an enzyme as a random walker cutting bonds and measured a gel-fraction
exponent β ≈ 1.0 that matches experiment; this is a *different* universality class
from the standard 3D random-percolation β ≈ 0.418 that the original audited repository
had hard-coded (a physics error flagged in `REPORT.md` §2.5). Vernerey and coworkers
have modeled coupled scaffold-degradation/tissue-deposition systems with continuum,
multiphase reactive-network approaches (Akalp, Bryant & Vernerey, *Soft Matter*
**12**, 7505 (2016); Vernerey and coworkers' topical review "The role of percolation
in hydrogel-based tissue engineering and bioprinting", *Curr. Opin. Biomed. Eng.*
(2020), doi:10.1016/j.cobme.2020.01.005, confirmed by CrossRef lookup during this
project's literature search). The Soft Matter 2016 paper is the mean-field/affine
reverse-gelation model directly implemented as the baseline in
`gelrigidity/mean_field.py`, so the quantitative comparison in this work is a
head-to-head test against that specific published model, not a straw-man.

**What is new here:** the mean-field model in the tissue-engineering literature is
a *continuum, homogenized* constitutive law — it has no notion of connectivity vs.
rigidity distinction and cannot represent a floppy-but-connected window. This work
computes the *same physical quantity* (the load-bearing modulus during the handoff)
two ways — mean-field and network-resolved — on the *same* occupancy trajectory, and
shows the two predictions diverge by ~172× specifically in the regime where the
scaffold degrades faster than the ECM can deposit (the regime that matters clinically:
failing formulations). This quantitative, matched-condition divergence between the
literature's own mean-field model and a network-resolved measurement, in the
regime where it is most consequential, has not been previously reported.

### 2.4 Cluster-size susceptibility and topological (persistent-homology) early-warning signals for percolation transitions

Established as physics; suggested as an early-warning framework in prior work,
including the project's own earlier phase. Speidel, Harrington and coworkers
(*Phys. Rev. E* **98**, 012318, 2018) studied H₁ persistent homology near percolation
transitions. This pillar was **descoped** from the current package (retained only in
`legacy/` with a deprecation banner) because the original implementation's realization
of it was diagnosed as circular/injected (an Ornstein-Uhlenbeck noise term whose
relaxation time was defined to diverge at p_c, per `REPORT.md` §2.3) and the
current package makes no early-warning claim that isn't independently re-derived.

### 2.5 Tensegrity / mixed rigid-floppy random networks (closest recent methodological analogue)

A December 2022 arXiv preprint ("Rigidity percolation in a random tensegrity via
analytic graph theory", arXiv:2212.04004), later published in PNAS (Nordstrom et al.,
*PNAS* **120**, e2302536120, 2023), analyzes rigidity percolation in random
tensegrities where compression- and tension-only elements ("struts" and "cables") are
added at random to a regular backbone, finding that adding cable-like elements
qualitatively changes the character of the rigidity transition, including avalanche-like
collective loss/gain of floppy modes. This is the most closely related recent
theoretical result on mixed bond-type rigidity percolation, but it is a static,
single-network analytical/graph-theoretic study of a fixed random backbone plus added
cables — it does not address two independently kinetically-evolving bond populations
sharing one node set, nor any biological degradation/deposition application, nor a
design metric.

---

## 3. Explicit statement of the novelty gap

No prior work identified — across the rigidity-percolation physics literature, the
reverse-gelation/enzymatic-degradation literature, the tissue-engineering
mean-field-modeling literature, or the recent off-lattice/ML rigidity-percolation
literature — computes a **time-resolved, network-measured, dual-population
(degrading scaffold + depositing ECM) rigidity-percolation handoff metric**, or
reports a **direct, quantitative, same-trajectory comparison** between that
measurement and the mean-field constitutive law the tissue-engineering field
currently uses for the same physical question. That comparison, and the resulting
design map over (MMP degradation rate × ECM deposition rate), is the specific,
falsifiable, genuinely open contribution of this work.

## 4. What is explicitly NOT claimed as novel

- The existence of a rigidity gap above the connectivity threshold (Feng-Sen,
  Thorpe — 1980s).
- Non-affine elastic-network solvers themselves (Head-Levine-MacKintosh 2003 and
  the broader biopolymer-network-mechanics literature).
- Random geometric graphs as a hydrogel topology model (standard in the polymer
  physics / continuum percolation literature).
- Enzymatic/reverse percolation of gels (Abete et al. 2004; Schultz et al.).
- Mean-field / affine reverse-gelation modeling of tissue-engineering scaffolds
  (Akalp, Bryant & Vernerey 2016 — this is the baseline being tested against, not
  a claim of this work).

## 5. Search methodology and coverage

Searches were run against CrossRef (targeted DOI verification of the suspected
Vernerey/Akalp Soft Matter baseline and related Acta Biomaterialia candidates),
arXiv, and general web search across the rigidity-percolation, tissue-engineering
percolation-modeling, and off-lattice/ML network-mechanics literatures. Full-text
of the two most methodologically load-bearing papers (the Vernerey group review and
a directly relevant biomechanics modeling paper) was fetched and read in full during
the earlier audit phase of this project (see `articles/` and the citations embedded
in `REPORT.md` §3 and §7). This memo's coverage is a snapshot as of the search dates
recorded in the project session log and should be re-verified against CrossRef/arXiv
immediately before submission, since new preprints in this active area (e.g. the
Nov-2024 GNN rigidity-percolation preprint found during this search) continue to
appear.
