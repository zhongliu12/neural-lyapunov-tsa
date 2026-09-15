# 02 — Lipschitz-bounded verification

The second framework. It exists because verification in
[framework 01](../01-neural-ode-lyapunov/) was too slow to be usable at realistic system sizes.

The Lyapunov conditions there are discharged by an SMT solver, which scales badly as the state
dimension grows. Here the network is trained with a bounded Lipschitz constant, and that bound is
used to certify the Lyapunov conditions directly by turning a search problem into a bound check.

Verification becomes fast and scales. The price is that a tighter Lipschitz bound forces a smoother
Lyapunov function, which cannot hug the true stability boundary as closely, so the certified region
of attraction contracts. Recovering that region is what
[framework 03](../03-gp-active-learning/) is for.

## How to run

Open `Lyapunov_Lip_3MG.ipynb` in Google Colab and run all cells.

## Results

**The notebook is committed with its outputs intact.** Verification timings, the certified region
under different Lipschitz bounds, and the comparison against the solver-based approach are all
visible by opening the notebook here on GitHub.
