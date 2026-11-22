# Methods summary

**Network.** Nodes are a Poisson point process in a periodic cube; bonds join nodes closer than a cutoff `r_c` (the tests use density 1 and `r_c` 1.5). The FCC lattice is used only to validate the solver.

**Modulus.** Bonds are linear springs. For an occupancy mask the code applies a small shear strain, relaxes node positions to mechanical equilibrium (non-affine), and returns the shear modulus. `eps` adds a transverse stiffness equal to `eps` times the longitudinal one. `G` below numerical floor means floppy.

**Gap.** The rigidity onset `p_on` is the first bond occupation at which G exceeds a fraction `tau` of its value at full occupancy (interpolated). The gap is `p_on(eps) - p_on(eps = 1)`, where the `eps = 1` network is rigid at the connectivity threshold. Bootstrap intervals resample seeds.

**Coupled dynamics.** Scaffold bonds are cleaved with a per-step hazard `k_base`; matrix bonds are deposited at rate `k_dep` on the same nodes. The union network carries both.

**Q.** `Q = min_t G_union(t) / (f * G_union(0))`, `f = 0.2`. Because the minimum includes `t = 0`, Q has a ceiling of `1/f` = 5 (reached when G never falls below its initial value). This is a property of the definition, not a clip applied in code. Q is floored at 1e-6 in the `Qm_f`/`Qf_f` columns, and log10 ratios use a floor of 1e-3; the safe/unsafe call does not depend on either floor. Twenty of the 192 `eps = 0` runs sit at the log floor.

**Mean-field comparison.** The affine law is evaluated on the same occupancy trajectories, scaffold with `G0 = 1` and matrix with `G0 = k_ecm = 2`, `p_c = 0.1126`. A run is misclassified when the two models disagree on `Q >= 1`.

**Intervals.** Agreement intervals are 95 % percentile bootstraps over runs (2000 resamples). Runs within a (k_base, k_dep) cell share parameters, so run-level resampling treats them as exchangeable; intervals are therefore indicative.

**Q-target sensitivity.** `results/sensitivity/qtarget_sensitivity.csv` repeats the misclassification rate for target fractions 0.05 to 0.40.
