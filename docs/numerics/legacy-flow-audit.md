# Historical gradient-flow audit

Source: numerical-gradient-flows revision
`0c6f5b50aee9ab5378970e28d700ab8fd193f554`. Its original local checkout was removed after integration.
The audit runs through an explicit source path in a separate CPU environment
with historical Flax/Optax imports available. EROT does not acquire those as
mandatory runtime dependencies.

## Historical reproduction

The one-time audit scripts and original local checkouts were removed after
integration. The scripts are recoverable from EROT commit `1d3ca9b` under
`benchmarks/`; source revisions are pinned above and in the provenance record.
Reproducing the historical audit requires those separate sources and their
optional dependencies. Current EROT correctness is covered by independent
reference tests; the recorded historical failures below are not expected
failures hidden in the active test suite.

## Observed outcomes

| Check | Result | Consequence for integration |
|---|---|---|
| Positive rectangular entropic marginal derivative | Returned gradient direction `-1.0553168573`; independent value difference `-0.5276612364`; potential direction `-0.5276584287` | Confirmed factor-of-two mismatch for the stated coupling-entropy objective; Stage 3F uses the independently verified potential convention |
| PDHG class, default steps | TypeError involving a `dataclasses.Field` at `self.tau[0]` | Replace malformed defaults in the new state/config contract |
| PDHG class, explicit steps | ConcretizationTypeError in the traced step-size branch | Keep traced array computation free of Python scalar conversion |
| Functional PDHG, 2,000 iterations with steps 0.01 | Mass `0.5232310919`, row residual `0.1503399944`, column residual `0.1377594560` | This iteration budget is infeasible; do not interpret its objective/timing as a successful solve |
| Entropy prox, alpha zero | Returns negative inputs unchanged | Use the nonnegative-domain limiting projection |
| Entropy prox, alpha `1e-4` | Nonfinite outputs for positive inputs 1 and 10 | Use a stable log-coordinate method |
| Entropy prox, alpha 0.1 and 1 | Matches the independent scalar reference at declared tolerances | Retain these as ordinary correctness cases |

The functional PDHG objective was `0.0092199818`, compared with the independent
feasible optimum `0.0030000007`. Infeasibility prevents concluding that this is
a converged wrong optimum; Stage 3F must implement and test stopping criteria,
the actual coupled-operator bound, and adequate iteration budgets.

The scalar oracle solves `exp(y) - z + alpha*y = 0` in log coordinates.
An initial bracketing error at `z=0` for small alpha was reproduced and
corrected before the outcomes above were recorded. Very small positive roots
may round to zero in float64; that is distinct from nonfinite overflow.

See [numerical conventions](conventions.md) for the objectives. PDHG's
unregularized JKO and finite-epsilon entropic JKO must not be compared as if
they solve the same objective.
