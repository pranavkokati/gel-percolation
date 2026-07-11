"""
run_rgg_designmap_sweep.py -- production design-map sweep on RGG topology.

Sweeps MMP degradation rate (k_base) x ECM deposition rate (k_dep), computing
the load-path-continuity metric Q for each combination, on the periodic
random-geometric-graph topology (the physical hydrogel topology; see
gelrigidity/dynamics.py). Replaces the earlier FCC-lattice design-map sweep
(results/fcc_crosscheck/data_sweep.npz), which is retained only as the
ordered-lattice cross-check.
"""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
from joblib import Parallel, delayed
from gelrigidity.dynamics import CoupledNetwork
from gelrigidity.handoff import load_path_continuity_Q, rigidity_connectivity_lag

k_base_vals = np.array([0.004, 0.009, 0.019, 0.032])
k_dep_vals  = np.array([0.008, 0.020, 0.050, 0.080])
SEEDS = [11, 22, 33]
G_TARGET_FRAC = 0.2
K_ECM = 2.0

RHO_X = 1.0
BOX_SIZE = 8.0     # bounded-cost production size (rho_x*box^3 ~ 512 nodes)
R_CUT = 1.5
N_STEPS = 150
RECORD_EVERY = 15
N_CELLS = 20
SEC_R = 2.5

def one(kb, kd, seed):
    cn = CoupledNetwork(topology="rgg", rho_x=RHO_X, box_size=BOX_SIZE, r_cut=R_CUT,
                         k_scaffold=1.0, k_ecm=K_ECM, seed=seed)
    cn.seed_scaffold(1.0)
    cn.seed_cells(n_cells=N_CELLS, secretion_radius=SEC_R)
    rec = cn.run(n_steps=N_STEPS, dt=1.0, record_every=RECORD_EVERY, mmp_level=1.0,
                 k_base=kb, k_dep=kd, solve_rigidity=True)
    G0 = rec['G_union'][0]
    q = load_path_continuity_Q(rec['t'], rec['G_union'], G_target=G_TARGET_FRAC*G0)
    lag = rigidity_connectivity_lag(rec['t'], rec['Pinf_scaffold'], rec['G_scaffold'])
    return (kb, kd, seed, q['Q'], q['t_valley'], lag['tau_gap'])

jobs = [(kb, kd, s) for kb in k_base_vals for kd in k_dep_vals for s in SEEDS]
t0 = time.time()
print(f"RGG design-map sweep: N_jobs={len(jobs)}, rho_x={RHO_X}, box_size={BOX_SIZE}, r_cut={R_CUT}, n_steps={N_STEPS}", flush=True)

res = Parallel(n_jobs=8, backend="threading")(delayed(one)(kb, kd, s) for kb, kd, s in jobs)
res = np.array(res)
print(f"ALL DONE wall={time.time()-t0:.1f}s", flush=True)

Qgrid = np.zeros((len(k_base_vals), len(k_dep_vals)))
taugrid = np.full_like(Qgrid, np.nan)
for i, kb in enumerate(k_base_vals):
    for j, kd in enumerate(k_dep_vals):
        m = (np.isclose(res[:, 0], kb)) & (np.isclose(res[:, 1], kd))
        Qgrid[i, j] = np.nanmean(res[m, 3])
        taugrid[i, j] = np.nanmean(res[m, 5])

np.savez("/tmp/rgg_data_sweep.npz", k_base=k_base_vals, k_dep=k_dep_vals,
         Q=Qgrid, tau_gap=taugrid, G_target_frac=G_TARGET_FRAC, k_ecm=K_ECM,
         rho_x=RHO_X, box_size=BOX_SIZE, r_cut=R_CUT, raw=res)

print("Q_min", Qgrid.min(), "Q_max", Qgrid.max(), "safe_frac", (Qgrid >= 1).mean())
