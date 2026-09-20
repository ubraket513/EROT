# Discrete heat-flow study

The example `examples/heat_flow.py` uses cell centers on [0,1], uniform volumes,
zero-flux boundaries, and initial density `1+0.2*cos(pi*x)`. For continuum
entropy/Wasserstein flow, the reference density at time t is
`1+0.2*exp(-pi²*t)*cos(pi*x)`. Reports compare masses at the actual accepted
physical time; an incomplete trajectory is not compared as though it reached
the requested horizon. No GPU timing is inferred.

Run with the installed package from a checkout:

```bash
python examples/heat_flow.py --cells 16 --time-step .02 --steps 2 \
  --output benchmark-results/heat.json
python examples/quadratic_flow.py
```

CPU float64 observations, physical horizon 0.04:

| Backend | Cells | dt | epsilon | Solver tolerance | Heat-mode mass L1 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| PDHG | 16 | .02 | none | 1e-7 | .0415963 |
| PDHG | 32 | .02 | none | 1e-7 | .0415468 |
| PDHG | 64 | .02 | none | 1e-7 | .0145801 |
| PDHG | 128 | .02 | none | 1e-7 | .0093521 |
| PDHG | 16 | .01 | none | 1e-7 | .0415991 |
| PDHG | 16 | .02 | none | 1e-8 | .0415966 |
| Sinkhorn | 16 | .02 | .002 | 1e-7 | .00564557 |
| Sinkhorn | 16 | .02 | .001 | 1e-7 | .0241345 |

All listed runs accepted the requested steps and met their discrete feasibility
and stationarity gates. PDHG used up to 30,000 iterations per step for cells
16/32 and up to 100,000 for 64/128. Sinkhorn used up to 2,000 outer iterations,
10,000 inner sweeps per solve and inner tolerance 1e-10. Work counts and complete
mass vectors are recorded in local `benchmark-results/stage-3f/` JSON files.
Use four steps for dt=.01 so the physical horizon remains fixed.

These are separate parameter checks, not a continuum-convergence certificate.
The coarse unregularized discretization exhibits pinning: remaining at a grid
cell can be optimal when the energy decrease from transferring mass is smaller
than its discrete transport cost. Reducing dt at fixed coarse grid does not
remove that effect. Refining the spatial grid here improves the heat-mode error;
tightening the inner optimization tolerance alone does not.

Finite epsilon changes the modeled JKO objective and can produce extra
smoothing. Its smaller error in one coarse case must not be presented as proof
of a more accurate continuum method. Reducing epsilon at fixed grid can expose
spatial pinning rather than improve the heat error monotonically. Production
studies should jointly resolve space, dt and epsilon, while separately checking
optimization tolerance. The example exposes those controls and the analytic
non-equilibrium reference instead of comparing only to a uniform steady state.

The quadratic example likewise solves the specified finite-grid JKO problem;
it does not promise that every sequence reaches the unconstrained energy
minimizer at a fixed grid/time step. Its snapshots retain this observable
behavior instead of normalizing or replacing the computed state.
