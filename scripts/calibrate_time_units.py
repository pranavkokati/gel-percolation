"""
scripts/calibrate_time_units.py

Converts this model's dimensionless per-simulation-step degradation rate
(k_base, k_dep) into real physical units (seconds, nM) using data extracted
from the full text of:

    Schultz, K. M. & Anseth, K. S. "Monitoring degradation of matrix
    metalloproteinases-cleavable PEG hydrogels via multiple particle
    tracking microrheology." Soft Matter 9, 1570-1579 (2013).
    DOI: 10.1039/C2SM27303A

All literal values below are transcribed directly from that paper's text
(abstract, Results, Fig. 6/7 captions) or from the cited Sigma-Aldrich
product data sheet for the exact reagent used (Collagenase from
Clostridium histolyticum, CAS 9001-12-1). No calibration number here is
fabricated or rounded to a "nice" value; every number is either quoted
from the source or is a straightforward derived quantity with the
derivation shown.

Two INDEPENDENT calibration routes are computed and cross-checked:

  Route A (dynamic-timescale anchor): the paper's measured critical
  degradation time t_c is identified with this model's measured tau_gap
  (rigidity-loss-to-connectivity-loss window, in simulation steps) as the
  same physical event class (loss of the last sample-spanning rigid/gel
  cluster). This directly fixes seconds-per-step, with no additional
  chemistry assumption.

  Route B (absolute rate-constant anchor): the paper's own fitted
  Michaelis-Menten rate constant k* = kcat/Km (M^-1 s^-1) for this exact
  MMP-peptide sequence is combined with the real collagenase
  concentration used in that experiment (0.2 mg/mL), converted to a molar
  concentration using the literature molecular-weight range for crude
  Clostridium histolyticum collagenase (68-130 kDa, Sigma-Aldrich product
  data), to give a real cleavage hazard k_eff = k*[MMP] in s^-1. This is
  compared against the model's own per-step cleavage hazard once Route A's
  seconds-per-step is applied.

Agreement between the two independent routes (within an order of
magnitude) is the actual calibration evidence reported in the manuscript;
it is NOT forced to agree by construction.
"""

import numpy as np
import json

# ---------------------------------------------------------------------
# 1. Literal values transcribed from Schultz & Anseth (2013), full text
# ---------------------------------------------------------------------
PAPER_DOI = "10.1039/C2SM27303A"

# Time-course experiment (R = 0.55, 0.2 mg/mL collagenase, 37 C):
TC_HOURS = 1.85                 # critical degradation time (last spanning cluster)
N_RELAX_TIMECOURSE = 0.16       # critical relaxation exponent, time-course
TOTAL_DEGRADATION_H = 2.5       # total time to full degradation, same condition
COLLAGENASE_MGML_TIMECOURSE = 0.2  # mg/mL used for the time-course experiment

# Degradability-sweep experiment (R = 0.85, 1; varying non-degradable fraction p):
PC_DEGRADABILITY = 0.589        # critical extent of degradability (time-cure superposition)
N_RELAX_DEGRADABILITY = 0.25    # critical relaxation exponent, degradability sweep

# Michaelis-Menten fits (k* = kcat/Km) for the MMP-cleavable peptide
# KCGPQGYIWGQCK, fit to the two independent experiments above:
K_STAR_DEGRADABILITY_M1S1 = 2100.0   # M^-1 s^-1, +/- 70, fit to Fig. 7 (varying p)
K_STAR_TIMECOURSE_M1S1 = 500.0       # M^-1 s^-1, +/- 22, fit to time-course data
GE0_SWOLLEN_PA = 43.0                # Pa, +/- 15, measured swollen R=1 gel modulus

# Collagenase molecular weight range (Sigma-Aldrich product data for
# Collagenase from Clostridium histolyticum, CAS 9001-12-1 -- the exact
# reagent named in the paper's Experimental section): crude culture
# filtrate contains >=7 proteases spanning 68-130 kDa.
MW_COLLAGENASE_LOW_GMOL = 68e3
MW_COLLAGENASE_HIGH_GMOL = 130e3

# ---------------------------------------------------------------------
# 2. This model's own measured quantities (already reported in REPORT.md)
# ---------------------------------------------------------------------
TAU_GAP_RGG_STEPS = 185     # steps between rigidity loss and connectivity loss (RGG production)
K_BASE_PRODUCTION = 0.012   # per-step, per-field-unit ("nM") degradation rate used in dynamics runs
MMP_FIELD_DEFAULT = 1.0     # uniform [MMP] field value used in direct-degradation production runs


def mgml_to_nM(mgml, mw_gmol):
    """mg/mL == g/L; convert to nM via molar mass."""
    conc_M = mgml / mw_gmol
    return conc_M * 1e9


def route_a_dt_per_step():
    """Anchor: paper's t_c <-> model's tau_gap, same physical event class."""
    dt_s_using_tc = (TC_HOURS * 3600) / TAU_GAP_RGG_STEPS
    dt_s_using_total = (TOTAL_DEGRADATION_H * 3600) / TAU_GAP_RGG_STEPS
    return dt_s_using_tc, dt_s_using_total


def route_b_k_eff_bracket():
    """Anchor: paper's absolute k* x real collagenase concentration."""
    mmp_nM_low = mgml_to_nM(COLLAGENASE_MGML_TIMECOURSE, MW_COLLAGENASE_HIGH_GMOL)  # heavier MW -> lower nM
    mmp_nM_high = mgml_to_nM(COLLAGENASE_MGML_TIMECOURSE, MW_COLLAGENASE_LOW_GMOL)  # lighter MW -> higher nM

    k_star_lo_nM1s1 = K_STAR_TIMECOURSE_M1S1 / 1e9
    k_star_hi_nM1s1 = K_STAR_DEGRADABILITY_M1S1 / 1e9

    k_eff_bracket = np.array([
        k_star_lo_nM1s1 * mmp_nM_low,
        k_star_lo_nM1s1 * mmp_nM_high,
        k_star_hi_nM1s1 * mmp_nM_low,
        k_star_hi_nM1s1 * mmp_nM_high,
    ])
    return mmp_nM_low, mmp_nM_high, k_eff_bracket


def cross_check(dt_s_bracket, k_eff_bracket):
    """Model's per-second hazard at MMP_FIELD_DEFAULT=1, using Route A's dt,
    compared against Route B's real k_eff bracket."""
    model_k_eff_per_s = np.array([K_BASE_PRODUCTION / dt for dt in dt_s_bracket])
    ratio_lo = model_k_eff_per_s.min() / k_eff_bracket.max()
    ratio_hi = model_k_eff_per_s.max() / k_eff_bracket.min()
    return model_k_eff_per_s, (ratio_lo, ratio_hi)


if __name__ == "__main__":
    dt_tc, dt_total = route_a_dt_per_step()
    mmp_lo, mmp_hi, k_eff_bracket = route_b_k_eff_bracket()
    model_k_eff, ratio = cross_check((dt_tc, dt_total), k_eff_bracket)

    result = {
        "source_doi": PAPER_DOI,
        "route_a_seconds_per_step": {"using_tc": dt_tc, "using_total_degradation_time": dt_total},
        "route_a_minutes_per_step": {"using_tc": dt_tc / 60, "using_total_degradation_time": dt_total / 60},
        "route_b_real_MMP_concentration_nM_bracket": [mmp_lo, mmp_hi],
        "route_b_k_eff_s-1_bracket": [float(k_eff_bracket.min()), float(k_eff_bracket.max())],
        "route_b_characteristic_bond_lifetime_min_bracket": [
            1 / k_eff_bracket.max() / 60, 1 / k_eff_bracket.min() / 60,
        ],
        "model_k_eff_per_s_at_MMP_field=1_using_route_a_dt": model_k_eff.tolist(),
        "cross_check_ratio_model_over_real_bracket": ratio,
        "tau_gap_steps_used": TAU_GAP_RGG_STEPS,
        "k_base_production_used": K_BASE_PRODUCTION,
    }
    print(json.dumps(result, indent=2))
    with open("calibration_result.json", "w") as f:
        json.dump(result, f, indent=2)
