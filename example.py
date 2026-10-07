"""
Minimal example (about one minute on a laptop).

N = 100 neurons store P = 50 random patterns (alpha = 0.5) with the Hebb rule,
then the network dreams. We follow the recognition rate rho during dreaming,
with clipping (A = 0.4) and without clipping (A = 1e7). This is a small-N
version of Fig. 2 of the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import hopfield as hf

N, alpha = 100, 0.5
tau_l, tau_d = 1.0, 100.0
D, every = 6000, 100            # dreaming steps, measure rho every 100 steps
instances = 5

np.random.seed(0)
hf.seed_numba(1)
P = round(alpha * N)
steps = np.arange(0, D + 1, every)

for A, label in [(1e7, "unclipped"), (0.4, "clipped, A = 0.4")]:
    rho = np.zeros((instances, len(steps)))
    for n in range(instances):
        X = np.random.choice(np.array([-1, 1], dtype=np.float32), size=(P, N))
        J = hf.learn(np.zeros((N, N), dtype=np.float32), X, A, tau_l)   # Hebb rule
        rho[n, 0] = hf.recognition_rate(J, X)
        for d in range(1, D + 1):
            hf.dream(J, A, tau_d)
            if d % every == 0:
                rho[n, d // every] = hf.recognition_rate(J, X)
    plt.plot(steps / (tau_d * N), rho.mean(0), label=label)
    print(f"{label}: rho before dreaming = {rho[:, 0].mean():.3f}, best = {rho.mean(0).max():.3f}")

plt.xlabel(r"$d/(\tau_d N)$")
plt.ylabel(r"$\rho$")
plt.title(rf"$N={N}$, $\alpha={alpha}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("example.png", dpi=150)
print("saved example.png")
