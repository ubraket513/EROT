# Numerical conventions

These contracts characterize the existing solvers and guide subsequent flow
and quantum integrations. Scientific changes must be explicit and independently
tested; a different backend must not silently change the objective.

## Classical transport

For equal positive total masses, Shannon OT minimizes
`sum(C * pi) + epsilon * sum(pi * (log(pi) - 1))` over nonnegative couplings
with the prescribed marginals. Interpret `0 log 0 = 0`.
This is coupling entropy, not KL relative to the marginal product.
Classical OT accepts non-unit equal masses and zero-support entries.

For two marginals, cost-unit potentials satisfy
`pi = exp((f[:, None] + g[None, :] - C) / epsilon)`.
On positive support, `f` is the optimized value's derivative in the first
marginal along mass-preserving directions. Its gauge is arbitrary; compensate
opposite potential shifts. A chosen weighted-mean-zero gauge must omit
zero-weight entries rather than evaluate `0 * -infinity`.
Epsilon changes the mathematical problem, not just convergence speed.

Quadratic classical OT minimizes
`sum(C * pi) + epsilon * ||pi||_F^2 / 2` under the same constraints.
Multi-marginal support and existing dense-result APIs remain available.

## Gradient flows

For squared-distance cost, the planned entropic JKO step minimizes
`F(rho) + T_epsilon(rho, rho_previous) / (2 * time_step)`.
The initial flow contract uses unit-mass vectors. The unregularized PDHG
formulation is equivalently
`time_step * F(rho) + 0.5 * sum(C * pi)`, with
`rho = row_sums(pi)`, `col_sums(pi) = rho_previous`, and `pi >= 0`.
These are distinct objectives at finite epsilon.

Represent finite-volume states as cell masses. Density samples with cell
volumes `w` become `mass = density * w`; integrated entropy is
`sum(mass * (log(mass / w) - 1))`. Record metric, quadrature, and boundary
conditions. Hidden renormalization is not a convergence certificate.

A valid accepted step needs feasibility and an optimality/stationarity check.
Failed inner solves must not silently advance physical time. Warm starts for
related problems differ from complete state for an exact resume.
The historical flow's `2*f` convention is audited independently before porting.

## Quantum transport

The existing solver minimizes
`Re trace(C Gamma) + epsilon * ||Gamma||_F^2 / 2` for Hermitian PSD `Gamma`
with prescribed partial traces. Retain complex/rank-deficient cases and
unequal subsystem dimensions. Dykstra's correction state resumes the same
problem; changed cost/marginals/epsilon require justified reinitialization.

A future entropy solver must state its trace domain and use the matching
primal/dual pair. Spectral `Tr[Gamma(log Gamma - I)]` has a trace-exponential
conjugate over the positive cone. Restricting to trace one gives a log-partition
form with an objective constant that must be preserved consistently.
Trace normalization alone does not enforce partial traces.
Neither static QOT nor a single extremal eigenpair establishes a quantum
gradient-flow method or a full PSD projection.

## Reference evidence

`tests/reference/test_numerical_contract.py` compares Shannon objective and
coupling to a separately expressed CVXPY problem, checks a mass-preserving
potential derivative by finite differences, and tests non-unit mass with zero
support. Existing unit modules also retain marked classical/quantum quadratic
CVXPY references and complex quantum invariants.
Comparisons use declared accuracy, precision, and objective conventions;
nonconverged or nonfinite results cannot win performance comparisons.
