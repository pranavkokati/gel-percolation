"""G(p, eps) sweep: transverse-stiffness crossover. Nested bond dilution
(common random numbers across p and eps) on periodic Poisson RGG."""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import multiprocessing as mp
from gelrigidity.rigidity import periodic_poisson_rgg, ElasticNetwork

BOX = float(sys.argv[1]); NSEED = int(sys.argv[2]); OUT = sys.argv[3]
PS = np.round(np.concatenate([np.linspace(0.06, 0.20, 8), np.linspace(0.22, 0.60, 20), [0.65, 0.75, 0.9, 1.0]]), 4)
EPS = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])

def one(seed):
    pos, box, bonds, rhat, rvec = periodic_poisson_rgg(1.0, BOX, 1.5, seed=1000 + seed)
    net = ElasticNetwork(pos, box, bonds, rhat, rvec)
    u = np.random.default_rng(seed).random(net.M)
    G = np.zeros((len(PS), len(EPS)))
    for i, p in enumerate(PS):
        occ = u < p
        for j, e in enumerate(EPS):
            G[i, j] = net.shear_modulus(occ, eps=e)
    return G, net.N, net.M

t0 = time.time()
with mp.get_context('fork').Pool(min(NSEED, 14)) as pool:
    res = pool.map(one, range(NSEED))
G = np.array([r[0] for r in res])
np.savez(OUT, G=G, ps=PS, eps=EPS, N=res[0][1], box=BOX)
print("done", BOX, NSEED, time.time() - t0, G.shape)
