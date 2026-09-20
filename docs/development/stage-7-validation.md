# Stage 7 validation: native decision and measurement tools

The current decision is [JAX with no selected native component](../performance/native-decision.md).
This follows the roadmap's explicit no-native outcome. No measured GPU fraction,
native full-solve speedup or isolated GPU peak-memory reduction is claimed.
Hardware is unavailable; the user requested preparation of validation tools.

`benchmarks/native_gate.py` provides conditional Amdahl screening and a matched
schema2 measurement comparison. Tests cover invalid fractions/nonfinite input,
overhead, the 30%-operation upper bound, speed and memory thresholds,
nonconvergence, objective disagreement, sample count, scientific/runtime
identity and invalid memory scope/provenance. An additional failing regression
showed that checking only input/hardware/dtype could admit memory from a different
epsilon, tolerance or output policy; memory evidence now binds the entire
comparison contract and optional flow metadata.

Validation on the CPU workspace:

```bash
.venv-cleanup/bin/python -m pytest tests/unit/test_native_gate.py tests/unit/test_benchmark_reporting.py -q
.venv-cleanup/bin/ruff check benchmarks/native_gate.py tests/unit/test_native_gate.py
python benchmarks/native_gate.py screen --operation-seconds 8 --solve-seconds 10 --kernel-speedup 4 --overhead-seconds 1
```

The focused tests passed **43/43**. Ruff passed. The synthetic CLI example returns
2× and labels the output a conditional prediction, not a measured result.
The preceding Stage 6 full suite passed 268 tests with 11 explicit skips;
Stage 7 changes only standard-library benchmark tools and documentation.
The integrated release will run the full suite again in Stage 8.

The decision document defines future runtime/ABI, stream, layout, aliasing,
error, batching, sharding, derivative and fallback gates. These are conditional
requirements for a future chosen component, not tests claimed to pass for an
unimplemented backend. C++/CUDA and existing Fortran libraries remain options
when justified by evidence or required capability.

## Final review and correction

The fresh final reviewer found one important issue: timing records could declare
kernel-only/device-core scope yet pass as full-solve results. Four regressions
(missing, kernel-only, device-core and unknown scope, in either/both positions)
failed before correction and pass after requiring the existing producer's exact
`synchronized host API including validation/transfers` scope.
Focused reporting/native checks now pass **47/47**. The complete current CPU
suite passed **286 tests, 11 skips** in 165.91s; nine require unavailable GPUs and
two require explicitly selected build artifacts. Raw local log:
`/tmp/erot-stage7-final-suite.log`. No critical or minor findings remained.

Review rulings: actual GPU gains stay unmeasured because hardware is unavailable
(cost if assumed otherwise: an unsupported performance claim); native ABI/stream/
transform tests remain conditional because no component is selected (cost if
later skipped: an unsafe backend admission); profiler authenticity remains an
experiment-review responsibility because declarations cannot authenticate a
capture (cost if skipped: invalid evidence could enter a performance decision).
