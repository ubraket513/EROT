# Dense quantum entropy transport

For Hermitian cost C and prescribed positive-definite trace-one density
matrices a and b, the solver minimizes

    Tr(C G) + epsilon Tr(G(log G-I))
    subject to G >= 0, Tr_b G = a, Tr_a G = b.

The bipartite tensor order is (a,b,a',b'); unequal subsystem dimensions are
supported. QOTLib informs the operator and spectral conjugate contracts; EROT
uses its own bounded JAX optimizer/state and an explicit trace-one conjugate.
This is dual gradient optimization, not a claimed quantum Sinkhorn algorithm.

With A*(u,v)=u tensor I + I tensor v and Y=A*(u,v)-C, the dual is

    D(u,v) = Tr(a u)+Tr(b v)-epsilon*(log Tr exp(Y/epsilon)+1).

The +1 follows from the primal entropy convention. Omitting it changes
objective agreement by epsilon. The Gibbs matrix exp(Y/epsilon)/Tr exp(Y/epsilon)
is recovered using a shifted eigendecomposition. The unconstrained positive-cone
entropy conjugate instead uses epsilon Tr exp(Y/epsilon); these are different
dual formulations and cannot be interchanged by renaming their routines.

The Hermitian gradient is (a-Tr_b G,b-Tr_a G). Duals use a traceless identity
gauge. Gradient ascent uses up to 30 Armijo backtracking trials with sufficient
increase coefficient 1e-4; the default initial learning rate is 1 and accepted
rates are retained. A machine-precision allowance is used in the line search.
Unsuccessful search returns status 5 without accepting a failed trial.

Success requires both the largest absolute marginal entry residual and absolute
primal-dual difference at or below tolerance, plus finite values. Recovered G
is PSD and trace one by construction, but these facts do not certify the
marginals. Before feasibility, the reported primal is an objective at an
infeasible matrix and its signed difference from the dual is not an optimality
bound. The difference is computed as -<u,gradient_a>-<v,gradient_b> from Gibbs
optimality, avoiding a second dense spectral factorization.

## Public and pure APIs

    from erot import solve, SolverConfig
    result = solve(cost, [a, b], problem="quantum",
                   regularizer="von_neumann", method="dual",
                   config=SolverConfig(epsilon=.5, tolerance=1e-8))

The existing quadratic/cyclic combination and CLI defaults remain unchanged.
For entropy CLI solves specify both --regularizer von_neumann and --method dual.
The host result retains coupling, error, iterations, converged and timing.
Invalid core outcomes map to infinite host error. Complete numerical diagnostics
and resume data are available from
erot.solvers.quantum_entropy.solve_quantum_entropy.

The pure solver accepts state=QuantumEntropyState and an additional iteration
budget, including zero. State includes both duals, recovered coupling,
cumulative accepted iterations and current step size. Resume only an identical
problem and controls; snapshots of a coupling alone are not complete checkpoints.
No per-iteration history is allocated. Status codes 0/1/2/3 denote converged,
iteration limit, invalid input and numerical failure; 5 denotes line-search
failure. Iteration counts count accepted dual updates, not spectral evaluations.
JIT and batching are tested; reverse-mode differentiation through the dynamic
optimization loop is not a supported contract.

## Domain and scaling limits

This finite-dual implementation requires strictly positive-definite marginals.
It does not add a diagonal perturbation or silently replace boundary data.
Rank-deficient marginals remain supported by the separate quadratic solver;
support-restricted entropy optimization needs a separate explicit formulation.
Near-boundary or small-epsilon cases can require many iterations or fail within
the supplied budget. Full-rank input alone is not a convergence-time guarantee.

Precision follows the common cost/marginal dtype. Enable JAX x64 explicitly for
tight tolerances. CPU tests cover a classical diagonal reduction, an independent
NumPy/SciPy complex dual, entropy constants, complex directional derivatives,
unequal dimensions, exact resume, caps, invalid inputs and batching.

Every trial performs a dense eigendecomposition of dimension n*m. Storage is
quadratic and spectral work cubic in this joint dimension; normalized Gibbs
recovery does not create a matrix-free or distributed algorithm. There are no
measured GPU performance claims. Run examples/quantum_entropy.py for a small
complex example. Block/chordal PDHG and Lanczos remain separate experimental
candidates, not implementations promoted by this dense solver's tests.
