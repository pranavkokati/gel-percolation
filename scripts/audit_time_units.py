"""Audit of the time-unit calibration (replaces Route A of calibrate_time_units.py).
Finding: Route A set (t_c of Schultz & Anseth 2013, the experimental GEL time) equal to tau_gap (model: the LAG between loss of rigidity
and loss of connectivity).  These are different quantities, so Route A is withdrawn.  Route B (k_eff = k* [E], an absolute per-bond hazard)
fixes seconds per model step independently: dt = k_base / k_eff.  Inputs are the values already transcribed in calibrate_time_units.py.
Prediction check (not a fit): in pure first-order degradation p(t) = exp(-k_base t), the model loses rigidity at p_r and connectivity at p_c.
Compared with the measured gel time (1.85 h) via the Route-B dt; the enzyme is crude collagenase, so [E] (hence k_eff) is an upper bound on active enzyme."""
import json, numpy as np
from scripts.calibrate_time_units import *
_,_,kb=route_b_k_eff_bracket(); kmin,kmax=kb.min(),kb.max()
dt_hi,dt_lo=K_BASE_PRODUCTION/kmin,K_BASE_PRODUCTION/kmax   # s per step
out=dict(route_B_dt_s_per_step=[dt_lo,dt_hi],route_A_dt_s_per_step=TC_HOURS*3600/TAU_GAP_RGG_STEPS)
for name,p in [('rigidity_loss_pr_0.41',0.41),('connectivity_loss_pc_0.113',0.113)]:
    steps=np.log(1/p)/K_BASE_PRODUCTION; out[name]=dict(steps=steps,minutes=[steps*dt_lo/60,steps*dt_hi/60])
out['tau_gap_pure_degradation_steps']=out['connectivity_loss_pc_0.113']['steps']-out['rigidity_loss_pr_0.41']['steps']
out['tau_gap_minutes']=[out['tau_gap_pure_degradation_steps']*dt_lo/60,out['tau_gap_pure_degradation_steps']*dt_hi/60]
out['experimental_gel_time_min']=TC_HOURS*60
out['note']='Route B dt is an order-of-magnitude bracket (2 x 2 corners of k* and enzyme MW). Experimental gel time is a different gel chemistry (PEG-peptide, flexible), so the comparison tests order of magnitude only.'
json.dump(out,open('results/calibration/time_unit_audit.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
