# 01 — Neural ODE surrogate + neural Lyapunov certification

The first framework. It certifies transient stability without any access to the vendor control code,
by learning a surrogate of the inverter dynamics and then training a Lyapunov function on that
surrogate.

This one works, but verification is slow enough to limit the system size it can handle. That
limitation is what [framework 02](../02-lipschitz-verification/) addresses.

## Two stages

**Stage 1 — MATLAB.** Train a neural ODE on inverter input/output trajectories and export the
trained weights to a `.pt` file.

**Stage 2 — Colab.** Load the exported surrogate, train a neural Lyapunov function on it, and
verify the certification conditions.

## How to run

### Stage 1 (optional — a pretrained surrogate is already provided)

Requires MATLAB with the Deep Learning Toolbox.

```matlab
run('Neural_ODE_1MG.m')     % trains the neural ODE surrogate
run('Save_w_b.m')    % exports weights to ../models/<MODEL_NAME>.pt
```

### Stage 2

Open `Neural_ODE_1MG.ipynb` in Google Colab and run all cells. It loads the surrogate from
`../models/`, trains the Lyapunov function, and produces the certified region of attraction.

## Results

**The notebook is committed with its outputs intact**:  training curves, the certified region, and
the verification results are all visible by opening the notebook here on GitHub. Nothing needs to be
rerun to see what the framework produces.
