# Historical gradient-flow audit

Source: numerical-gradient-flows revision
`0c6f5b50aee9ab5378970e28d700ab8fd193f554`. Its original checkout is unchanged.
The audit runs through an explicit source path in a separate CPU environment
with historical Flax/Optax imports available. EROT does not acquire those as
mandatory runtime dependencies.

## Reproduction

```bash
python -m venv .venv-legacy-audit
.venv-legacy-audit/bin/python -m pip install -e '.[test]' flax optax
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu .venv-legacy-audit/bin/python benchmarks/audit_legacy_flow.py --source-path /path/to/numerical-gradient-flows --output benchmark-results/stage-1/legacy/audit.json
```

The command intentionally exits nonzero for mismatched, errored, or unavailable
checks while retaining its JSON report. The report includes the exact input,
source revision, resolved library versions, values, and exception details.
These historical failures are not expected failures hidden in the normal suite.

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
