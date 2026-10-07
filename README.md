# Dreaming in a Hopfield model with bounded synaptic strength

Code for

> E. Marinari, S. Rossi, F. Zamponi,
> *Dreaming improves memorization in a Hopfield model with bounded synaptic strength*,
> submitted to Neural Networks (2026).

The code simulates a Hopfield network of $N$ binary neurons that stores $P=\alpha N$
random patterns. It combines three ingredients:

- Hebbian **learning**, $J_{ij} \leftarrow J_{ij} + \xi_i\xi_j/(\tau_l\sqrt N)$;
- **dreaming** (unlearning), $J_{ij} \leftarrow J_{ij} - s^*_i s^*_j/(\tau_d\sqrt N)$, where $s^*$ is the
  fixed point reached from a random configuration;
- **clipping**, where every coupling is kept in $[-A, A]$ after each update.

A pattern counts as retrieved if the zero-temperature dynamics started from it stops at a
fixed point that differs from it in fewer than 2% of the neurons. The recognition rate
$\rho$ is the number of retrieved patterns divided by $N$.

## Installation

Python ≥ 3.9 is required. The notebook renders text with LaTeX (`text.usetex`), so it also
needs a LaTeX installation; set `"text.usetex": False` in its first cell to do without.

```bash
pip install -r requirements.txt
```

## Files

| file | content |
|---|---|
| `hopfield.py` | core routines: `learn`, `dream`, `relax` (asynchronous dynamics), `recognition_rate` |
| `run.py` | command-line launcher for the learning-and-dreaming algorithm (Alg. 1 of the paper); writes one `.npz` file |
| `paper_runs.sh` | prints the `run.py` commands that produce the data of Figs. 1–5 |
| `example.py` | small example (a few seconds on a laptop): Hebb rule followed by dreaming, with and without clipping |
| `PlotFigures.ipynb` | makes Figs. 1–5 of the paper from the files in `data/` |
| `data/` | simulation results used in the figures, one `.npz` file per run (see below) |
| `figures/` | Figs. 1–5 as produced by the notebook |

## Quick start

```bash
python3 example.py
```

This produces `example.png`, a small-$N$ version of Fig. 2: $\rho$ as a function of the number
of dreaming steps, with and without clipping.

## Running simulations

`run.py` runs Alg. 1. For each instance it draws $P$ patterns, sets $J=0$, and then repeats $T$ cycles.
Each cycle has $L$ learning steps, on $L$ distinct patterns drawn at random, followed by $D$
dreaming steps. The main options are listed below; `python3 run.py -h` lists them all.

| option | meaning | default |
|---|---|---|
| `--N`, `--alpha` | number of neurons, load $\alpha=P/N$ | 200, 0.8 |
| `--L`, `--D`, `--T` | learning steps and dreaming steps per cycle, number of cycles | $P$, 0, 1 |
| `--tau_l`, `--tau_d` | inverse learning and dreaming rates | 1, 10 |
| `--A` | clipping threshold (`1e7` = unclipped) | 0.4 |
| `--measure` | measure $\rho$ after cycles (`cycle`) or after dreaming steps (`dream`) | `cycle` |
| `--every`, `--logspaced` | measure every *k* cycles or dreaming steps, or at *n* log-spaced cycles | 1, off |
| `--instances`, `--seed` | number of independent instances, random seed | 50, random |

Special cases:

- **Hebb rule:** `--T 1 --D 0`, with the default $L=P$.
- **Hebb rule followed by dreaming:** `--T 1 --D <d> --measure dream`.
- **Daydreaming** (Serricchio et al. 2025): `--L 1 --D 1` with `tau_l = tau_d`.

The output file stores `matrix`, the recognition rate averaged over instances at each
measurement, and `cycles` (or `dreams` with `--measure dream`), the cycles or dreaming steps
at which $\rho$ was measured. For example:

```bash
python3 run.py --N 200 --alpha 1.2 --A 0.4 --tau_d 100 --D 6000 --measure dream --every 100 --instances 10
```

This runs the Hebb rule followed by dreaming, with clipping. $\rho$ goes from about 0.05 to a
maximum of about 0.15 near $d \approx 1700$, which is the large-$\alpha$ curve of Fig. 2(b).

## Reproducing the figures

```bash
./paper_runs.sh fig2                                 # list the commands for Fig. 2
./paper_runs.sh fig2 | xargs -P 8 -I{} sh -c "{}"    # run them, 8 at a time
```

| figure | content | parameters |
|---|---|---|
| 1 | Hebb rule, unclipped and clipped | $N=100\ldots1600$, $A=0.4$ |
| 2 | Hebb rule followed by dreaming, $\rho$ vs dreaming steps | $N=200$, $\tau_d=100$; unclipped $D=\tau_d N$, clipped $D=0.3\,\tau_d N$ |
| 3 | maximum of $\rho$ over the dreaming steps vs $\alpha$ | $N=100\ldots1600$, $\tau_d=100$; unclipped $D=\tau_d N$, clipped $D=0.2\,\tau_d N$ |
| 4 | learning and dreaming, unclipped, in the $(L,R)$ plane with $R=D\tau_l/(L\tau_d)$ | $N=200$, $\alpha=0.8$, $\tau_l=1$, $\tau_d=10$, $L=5\ldots100$, $R=2^{-5}\ldots2$; panel (b): $L=80$, $R=0.6\ldots1.4$ |
| 5 | learning and dreaming, clipped, same grid as Fig. 4 | as Fig. 4, with $A=0.4$ and $T=50$ |

Every run writes its result to a separate file. The files used in the paper are in `data/`,
one folder per type of run:

| folder | figure |
|---|---|
| `UnclippedHebb`, `ClippedHebb` | 1, 3(b) |
| `UnclippedDreaming`, `ClippedDreaming` | 2 |
| `UnclippedDreaming_undersample`, `ClippedDreaming_undersample` | 3 |
| `UnclippedMixed` | 4 |
| `ClippedMixed` | 5 |

`PlotFigures.ipynb` reads them and writes the figures to `figures/`.
The supplementary figures use the same launcher with different parameters.

The heavy routines are compiled with [numba](https://numba.pydata.org). The dynamics updates
the local fields after each spin flip instead of recomputing them. This gives the same
trajectories as the direct implementation, up to floating-point rounding, and runs several
times faster.

## Citation

If you use this code, please cite the paper above.

## License

MIT, see `LICENSE`.
