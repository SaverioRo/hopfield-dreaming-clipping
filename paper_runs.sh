#!/bin/bash
# Prints the run.py commands that generate the data of Figs. 1-5 of the paper,
# one command per line. Each command writes one .npz file.
#
#   ./paper_runs.sh fig2                                  # show the commands
#   ./paper_runs.sh fig2 | bash                           # run them one after the other
#   ./paper_runs.sh fig2 | xargs -P 8 -I{} sh -c "{}"     # run 8 at a time
#
# Some runs (Figs. 3 and 4 at large N or many cycles) take days: on a cluster,
# send each line as a separate job.

case "$1" in

fig1)   # Hebb rule, unclipped and clipped (A = 0.4)
  for N in 100 200 400 800 1600; do
    for a in 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.25 0.3 0.4; do
      for A in 1e7 0.4; do
        echo "python3 run.py --N $N --alpha $a --A $A --tau_l 1 --tau_d 1 --instances 50"
      done
    done
  done ;;

fig2)   # Hebb rule followed by dreaming, rho after every dreaming step, N = 200
  for a in 0.02 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 1.0; do
    echo "python3 run.py --N 200 --alpha $a --A 1e7 --tau_d 100 --D 20000 --measure dream --every 1 --instances 50"
  done
  for a in 0.02 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.4; do
    echo "python3 run.py --N 200 --alpha $a --A 0.4 --tau_d 100 --D 6000 --measure dream --every 1 --instances 50"
  done ;;

fig3)   # as Fig. 2 for several N, rho measured every N dreaming steps; the figure uses its maximum over d
  for N in 100 200 400 800 1600; do
    for a in 0.02 0.1 0.2 0.3 0.4 0.5 0.6 0.65 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.95 1.0 1.1 1.2; do
      echo "python3 run.py --N $N --alpha $a --A 1e7 --tau_d 100 --D $((100*N)) --measure dream --every $N --instances 50"
    done
    for a in 0.02 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2; do
      echo "python3 run.py --N $N --alpha $a --A 0.4 --tau_d 100 --D $((20*N)) --measure dream --every $N --instances 50"
    done
  done ;;

fig4)   # learning and dreaming, unclipped, tau_l = 1, tau_d = 10, J = 0 at start
        # grid: L = 5..100, R = D tau_l/(L tau_d) = 2^(1-k/2), k = 0..12
        # number of cycles = S(R)/L, with S(R) the patterns needed to reach the stationary state
  python3 -c '
import numpy as np
for R in 2**(1 - 0.5*np.arange(13)):
    S = 2e4 if R > 1.3 else 4e5 if R > 0.8 else 2e5 if R > 0.45 else 1e5
    for L in range(5, 101, 5):
        D, T = int(np.round(10*R*L)), int(np.ceil(S/L))
        print(f"python3 run.py --N 200 --alpha 0.8 --A 1e7 --tau_l 1 --tau_d 10 --L {L} --D {D} --T {T} --logspaced 100 --instances 50")
for R in [0.6, 0.8, 1.0, 1.2, 1.4]:               # panel (b): L = 80 across the band R ~ 1
    print(f"python3 run.py --N 200 --alpha 0.8 --A 1e7 --tau_l 1 --tau_d 10 --L 80 --D {int(round(800*R))} --T 10000 --logspaced 5000 --instances 40")
' ;;

fig5)   # learning and dreaming, clipped (A = 0.4), same (L, R) grid as Fig. 4, 50 cycles
  python3 -c '
import numpy as np
for R in 2**(1 - 0.5*np.arange(13)):
    for L in range(5, 101, 5):
        print(f"python3 run.py --N 200 --alpha 0.8 --A 0.4 --tau_l 1 --tau_d 10 --L {L} --D {int(np.round(10*R*L))} --T 50 --instances 50")
' ;;

*)
  echo "usage: $0 fig1|fig2|fig3|fig4|fig5" >&2; exit 1 ;;
esac
