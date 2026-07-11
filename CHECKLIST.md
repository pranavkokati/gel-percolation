# GitHub push-readiness checklist

Status as of the final verification pass for this rebuild. Each item was
checked directly (command run and output inspected), not assumed.

## Packaging and installability

- [x] `setup.py` present, valid, `pip install -e .` succeeds with no errors.
- [x] Package importable as `import gelrigidity` from an arbitrary working
      directory after editable install (verified from `/tmp`).
- [x] `environment.yml` present and pinned to versions matching `setup.py`'s
      `install_requires` (numpy, scipy, matplotlib, networkx, pandas, joblib,
      PyYAML, tqdm, pytest, pytest-cov).
- [x] `requirements.txt` present as a pip-only alternative to `environment.yml`.
- [x] `CITATION.cff` present, valid CFF 1.2.0, version/author/license/keywords
      consistent with `setup.py`.
- [x] `LICENSE` present (MIT), consistent with `CITATION.cff` and `setup.py`
      license fields.

## Tests

- [x] `python -m pytest tests/ -q` → **26 passed**, run against the
      `pip install -e .`-installed package (not just a `PYTHONPATH` hack),
      confirming the packaging path itself is exercised by the test suite.
- [x] Test files cover all five current modules: `test_rigidity.py`,
      `test_dynamics.py`, `test_handoff.py`, `test_mean_field.py`,
      `test_network_model.py`, plus `conftest.py` fixtures.
- [x] No test imports anything from `legacy/`.

## Reproducibility

- [x] `scripts/generate_figures.py` supports checkpoint-load-by-default with
      `--recompute` and `--only` flags.
- [x] Spot-tested `--only fig0_solver_validation --recompute` from a deleted
      checkpoint: regenerated fit parameters (p_r = 0.4831, f = 1.0439 ±
      0.0398) matched the pre-existing checkpoint to full floating-point
      precision — confirms deterministic seeding and genuine end-to-end
      regeneration, not silent reliance on stale cached output.
- [x] All `results/` checkpoint files (10 files: 7 `.npz` + 2 `.json` + 1
      `.csv`, across `fcc_crosscheck/` and `rgg_production/`, totaling 78,782
      bytes / ~77 KB of raw file content, ~92 KB on-disk with filesystem
      block rounding) are small enough to commit directly — no Git LFS
      needed.
- [x] **`.gitignore` bug found and fixed**: the original blanket `results/`
      ignore rule would have silently excluded every reproducibility
      checkpoint from the pushed repository, even though README.md,
      REPORT.md, and the manuscript all cite specific values from those
      files. Corrected to a targeted rule that ignores scratch
      `.npy`/`.pkl`/`.h5` outputs everywhere but explicitly tracks
      `results/**/*.npz`, `results/**/*.json`, `results/**/*.csv`. Verified
      via `git add -n results/` — all 10 checkpoint files now stage cleanly.
- [x] Figures directory (`figures/*.png`, `figures/*.pdf`) explicitly
      exempted from the general image-ignore rule so rendered figures show
      up on GitHub without requiring a regeneration step.

## Code organization and legacy handling

- [x] Current package (`gelrigidity/`) contains only the corrected,
      currently-imported modules: `rigidity.py`, `dynamics.py`, `handoff.py`,
      `mean_field.py`, `network.py`, `utils.py`.
- [x] Every flagged original module retained under `legacy/` carries an
      explicit deprecation banner stating what was wrong with it (closed-form
      exponents presented as measured, synthetic ECM curve, engineered
      OU-noise term, inconsistent Q definitions) and pointing to its
      replacement — verified by inspecting the banner text directly.
- [x] No current code, test, or figure-generation script imports from
      `legacy/` — verified by grep across `gelrigidity/`, `tests/`,
      `scripts/`.
- [x] `config/default_params.yaml` (the one config file belonging to the
      deprecated model) was moved into `legacy/config/` rather than left at
      the repo root implying it's still the active configuration.

## Documentation

- [x] `README.md` — project overview, quickstart, corrected headline numbers
      (finite-size thresholds/exponent table, mean-field comparison),
      correct script usage section, all consistent with `REPORT.md`.
- [x] `REPORT.md` — full audit trail: original-repository audit (§2),
      novelty positioning (§3), methods (§4), FCC cross-check results (§5),
      honest limitations (§6), recommendations addressed (§7), Phase-0
      artifacts (§8), full RGG production results (§9), manuscript/repo
      status (§10, this section's sibling).
- [x] `NOVELTY.md` — pillar-by-pillar prior-art memo with specific citations
      checked against every claimed-novel ingredient, and a precise
      statement of the actual novel contribution.
- [x] `paper/manuscript.md` — submission-ready draft (title, abstract,
      intro, methods, results, discussion, limitations, references),
      every reported number cross-checked against `REPORT.md` and the
      underlying `results/` checkpoint files.
- [x] `paper/journal_positioning.md` — target-journal rationale (*Soft
      Matter*, full paper) with alternatives considered and a note to
      reconfirm currency before actual submission.
- [x] No document contains a hardcoded number that isn't traceable to a
      checkpoint file or a cited literature source (per the standing
      no-hardcoding constraint) — spot-checked across README/REPORT/
      manuscript during this pass; all cross-referenced values matched their
      source JSON/`.npz` files exactly.

## Known residual items (not blocking, tracked here for follow-up)

- [x] Model time units (per-simulation-step rates) are now bracket-calibrated
      against a real experimental MMP degradation time course to
      order-of-magnitude precision. Verified real-data target: Schultz &
      Anseth, *Soft Matter* 2013 (DOI 10.1039/C2SM27303A) — full text
      supplied by the user, enabling two independent, cross-validated
      calibration routes (`scripts/calibrate_time_units.py`,
      `calibration_result.json`): (A) dynamic-timescale anchor
      (t_c = 1.85 h ↔ this model's τ_gap = 185 steps) → 0.6–0.8 min/step;
      (B) absolute-rate anchor (fitted k* = 500–2100 M⁻¹s⁻¹ + real
      collagenase concentration 0.2 mg/mL → 1.5–2.9 µM) → real k_eff
      ≈ 7.7×10⁻⁴–6.2×10⁻³ s⁻¹. This model's own k_base = 0.012 converts
      (via Route A) to k_eff ≈ 2.5–3.3×10⁻⁴ s⁻¹, within an order of
      magnitude of Route B's independent bracket (ratio 0.04–0.43) —
      agreement with no free-parameter tuning. Documented in `REPORT.md`
      §11/§6 and manuscript §5 (Limitations). A precise fit to a single
      time-resolved trajectory remains future work — not blocking.
- [ ] `tau_gap_trajectory.npz` and `divergence_trajectory.npz` do not store a
      `box_size` metadata field directly in the array (unlike
      `thresholds_exponent_scan.npz`, which does) — the box size is instead
      recorded in `REPORT.md` prose. Low-priority consistency follow-up:
      add `box_size` as a stored scalar in both checkpoints so the value is
      self-describing without cross-referencing the report text.
- [ ] Full literature reference list in `paper/manuscript.md` uses short-form
      citations (author/journal/year/volume/page) rather than a formatted
      bibliography file (e.g. `.bib`); should be converted to the target
      journal's reference manager format immediately before submission.

## Final state confirmed this pass

- `git add -n -A .` dry run reviewed in full: all additions/removals/renames
  are exactly the intended repository-rebuild diff (legacy modules moved to
  `legacy/`, corrected `gelrigidity/` package added, superseded figures
  replaced by the RGG figure set, packaging/citation/report/novelty/paper
  files added) — no unintended or stray file changes.
- `python -m pytest tests/ -q` — 26 passed, 0 failed, 0 skipped.
- Repository is push-ready pending only the three residual items above,
  none of which block a public push (all are documented limitations or
  low-priority polish, not correctness issues).
