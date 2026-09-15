# neural-lyapunov-tsa

Certifying transient stability of inverter-based power systems whose dynamics are not disclosed.

Inverter manufacturers do not release the control software running inside their hardware, so the
dynamics that govern a modern grid's transient behavior are a black box. This repository collects the
code behind three successive frameworks from my PhD work on certifying stability under that
constraint.

## Three frameworks, developed in sequence

These are three separate frameworks built one after another, not three modules of a single
integrated system. Each one exists because the previous one ran a measurable deficit.

| | Framework | What it added | What it cost |
|---|---|---|---|
| **01** | [Neural ODE + neural Lyapunov](01-neural-ode-lyapunov/) | Certification without access to the vendor control code | Verification too slow to be usable |
| **02** | [Lipschitz-bounded verification](02-lipschitz-verification/) | Replaced the SMT solver; verification becomes fast and scalable | Tighter bounds contract the certified region of attraction |
| **03** | [GP-guided active learning](03-gp-active-learning/) | Recovers the lost region by sampling where it matters | More training and sampling cost |

They share the problem and the certification formulation, not a codebase. Verification speed,
certified volume, and training cost trade against each other across the three, and locating that
trade-off surface — rather than tuning any single framework further — is what I am working on next.

## Running the code

Frameworks **02** and **03** run end to end in Google Colab with no local setup.

Framework **01** has two stages: the neural ODE surrogate is trained in MATLAB and exported as a
`.pt` file, which the Colab notebook then loads.

**Every notebook is committed with its outputs saved.** Figures, certified regions, and training
curves are visible directly on GitHub without running anything.

## Repository layout

```
01-neural-ode-lyapunov/    Framework 1 — MATLAB training + Colab certification
02-lipschitz-verification/ Framework 2 — Colab only
03-gp-active-learning/     Framework 3 — Colab only
```

## Test systems

All case studies use standard benchmark models.

## Author

Zhong Liu — PhD candidate, Elmore Family School of Electrical and Computer Engineering,
Purdue University. <liu3746@purdue.edu>

Released under the MIT License.
