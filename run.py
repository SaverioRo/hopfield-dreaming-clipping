"""
Learning-and-dreaming algorithm (Alg. 1 of the paper) with clipping.

Each instance draws P = alpha*N random patterns and starts from J = 0, then repeats T cycles of
  - L learning steps (L distinct patterns, drawn without replacement within the cycle),
  - D dreaming steps,
clipping the couplings to [-A, A] after every update. The recognition rate rho
(2% criterion) is measured during the run and averaged over the instances.

Special cases (see README for the exact commands of each figure):
  Hebb rule      T=1, L=P, D=0
  dreaming       T=1, L=P, D>0, measured after the dreaming steps (--measure dream)
  daydreaming    L=D=1, tau_l=tau_d
Use a very large A (e.g. 1e7) for the unclipped model.

Output: one .npz file with
  matrix : rho averaged over instances, one value per measurement
  cycles : the cycles at which rho was measured (or `dreams`, the dreaming steps, with --measure dream)
"""
import argparse, time
import numpy as np
import hopfield as hf

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--N", type=int, default=200, help="number of neurons")
p.add_argument("--alpha", type=float, default=0.8, help="load alpha = P/N")
p.add_argument("--L", type=int, default=None, help="learning steps per cycle (default: P)")
p.add_argument("--D", type=int, default=0, help="dreaming steps per cycle")
p.add_argument("--T", type=int, default=1, help="number of cycles")
p.add_argument("--tau_l", type=float, default=1.0, help="inverse learning rate")
p.add_argument("--tau_d", type=float, default=10.0, help="inverse dreaming rate")
p.add_argument("--A", type=float, default=0.4, help="clipping threshold (1e7 = unclipped)")
p.add_argument("--instances", type=int, default=50, help="independent sets of random patterns")
p.add_argument("--measure", choices=["cycle", "dream"], default="cycle",
               help="measure rho after cycles, or after dreaming steps")
p.add_argument("--every", type=int, default=1, help="measure every k cycles (or dreaming steps)")
p.add_argument("--logspaced", type=int, default=0,
               help="instead of --every, measure at this many log-spaced cycles")
p.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
a = p.parse_args()

N = a.N
P = round(a.alpha * N)
L = P if a.L is None else a.L
if L > P:
    raise SystemExit("L cannot exceed P")
if a.seed is not None:
    np.random.seed(a.seed)
    hf.seed_numba(a.seed + 1)

# measurement schedule
if a.measure == "cycle":
    if a.logspaced > 0:
        checks = np.unique(np.round(np.logspace(0, np.log10(a.T), a.logspaced)).astype(int))
    else:
        checks = np.arange(a.every, a.T + 1, a.every)
else:
    checks = np.arange(a.every, a.T * a.D + 1, a.every)          # global dreaming-step index
checkset = set(checks.tolist())
rho = np.zeros((a.instances, len(checks)))

t0 = time.time()
for n in range(a.instances):
    X = np.random.choice(np.array([-1, 1], dtype=np.float32), size=(P, N))
    J = np.zeros((N, N), dtype=np.float32)
    k = 0
    dstep = 0
    for c in range(1, a.T + 1):
        hf.learn(J, X[np.random.choice(P, L, replace=False)], a.A, a.tau_l)
        for _ in range(a.D):
            hf.dream(J, a.A, a.tau_d)
            dstep += 1
            if a.measure == "dream" and dstep in checkset:
                rho[n, k] = hf.recognition_rate(J, X); k += 1
        if a.measure == "cycle" and c in checkset:
            rho[n, k] = hf.recognition_rate(J, X); k += 1
    print(f"instance {n+1}/{a.instances} done ({time.time()-t0:.0f} s)", flush=True)

out = (f"Recognition2percentGeneral_N_{N}_load_{P/N:.3f}_Dreams_{a.D}_Batches_{L}_taod_{a.tau_d:.2f}"
       f"_taol_{a.tau_l:.2f}_Cycles_{a.T}_Ninstances_{a.instances}_clippedA{a.A:.2f}.npz")
np.savez_compressed(out, matrix=rho.mean(0), **{"cycles" if a.measure == "cycle" else "dreams": checks})
print("saved", out)
