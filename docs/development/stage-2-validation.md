# Stage 2 verification

The pure Sinkhorn API, distinct warm-start/resume state, explicit plan
materialization and full quantum Dykstra state are implemented. Existing dense
API signatures and CLI behavior remain. See
[solver state](../numerics/solver-state.md) for contracts and limitations.

Commits `42c48ef`, `81cd944`, and `370cca5` establish the core, compatibility
integration and quantum resume. Focused tests prove cost-unit potential scaling,
zero-support and multi-marginal mass conservation, changed-epsilon warm starts,
bit-identical fixed-work resume, and JAX composition. Quantum coverage includes
unequal dimensions, complex marginals with real costs, and independent CVXPY
references retained from the baseline.

Fresh review of `50cc9f3..370cca5` found inconsistent quantum host/core input
tolerances and iteration-range truncation. Both were reproduced and corrected.
The reported minor complex-classical casting issue was treated as material
because it silently changed input data; it too has a regression test and fix.
Invalid core outcomes now yield infinite error through compatibility wrappers,
which have no status field. No Critical findings were reported.

The final CPU suite reports **98 passed, 5 skipped**: three GPU checks and two
opt-in distribution-content checks. Ruff passes the runtime, test and benchmark
scopes. A built wheel was installed into a fresh base-only environment and its
API/CLI workflow exercised outside the checkout; the final review fixes were
rebuilt/reinstalled, and both the API/CLI smoke and an independent zero-cost
quantum-state smoke passed outside the checkout. Source checkouts and SDPLab
remain outside this change.

No actual GPU placement, throughput, multi-device performance or declared
minimum dependency combination is newly certified by these CPU results. Dense
iterations still form dense intermediates. Dynamic-loop reverse-mode
differentiation is not a supported contract. Full Dykstra resume retains all
correction matrices and does not reduce quantum memory capacity requirements.
