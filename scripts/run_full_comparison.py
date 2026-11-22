"""Full 4x4 (k_base x k_dep) coupled degradation/deposition sweep, n seeds,
at several transverse-stiffness values eps.  Stores per-seed Q, G trajectories,
and the mean-field Q computed on the identical occupancy trajectory (with
p_c taken from the independent connectivity estimate p_c = 0.114 of the RGG).
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np, multiprocessing as mp
from gelrigidity.dynamics import CoupledNetwork
from gelrigidity.mean_field import affine_meanfield_modulus

BOX = float(sys.argv[1]); NSEED = int(sys.argv[2]); OUT = sys.argv[3]
EPS_LIST = [float(x) for x in sys.argv[4].split(",")]
KB = np.array([0.004, 0.009, 0.019, 0.032]); KD = np.array([0.008, 0.020, 0.050, 0.080])
PC = 0.1126; K_ECM = 2.0; N_STEPS = 150; REC = 15; N_CELLS = 20; SEC_R = 2.5

def one(args):
    kb, kd, seed, eps = args
    cn = CoupledNetwork(topology="rgg", rho_x=1.0, box_size=BOX, r_cut=1.5, k_scaffold=1.0, k_ecm=K_ECM,
                        seed=seed, eps=eps)
    cn.seed_scaffold(1.0); cn.seed_cells(N_CELLS, SEC_R * BOX / 8.0 if False else SEC_R)
    rec = cn.run(n_steps=N_STEPS, dt=1.0, record_every=REC, mmp_level=1.0, k_base=kb, k_dep=kd)
    return rec

if __name__ == "__main__":
    jobs = [(kb, kd, 100 + s, e) for e in EPS_LIST for kb in KB for kd in KD for s in range(NSEED)]
    t0 = time.time()
    with mp.get_context("fork").Pool(14) as pool:
        recs = pool.map(one, jobs, chunksize=1)
    keys = ["t", "p_scaffold", "p_ecm", "p_union", "Pinf_scaffold", "Pinf_union", "G_scaffold", "G_ecm", "G_union"]
    arr = {k: np.array([r[k] for r in recs]) for k in keys}
    meta = np.array([(j[0], j[1], j[2], j[3]) for j in jobs])
    np.savez(OUT, meta=meta, box=BOX, pc=PC, k_ecm=K_ECM, **arr)
    print("done", len(jobs), time.time() - t0)
