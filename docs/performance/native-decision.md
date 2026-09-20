# Native acceleration decision

Decision for the integrated release: keep the implemented JAX solvers. No native
operation is selected, and no C++, CUDA, Pallas or Fortran build dependency is
introduced. This is a decision from the available evidence, not a claim that JAX
achieves maximal performance. Reopen it when target-hardware profiles or a
required capability justify a particular implementation.

## Evidence and limits

The [Stage 4 report](../development/stage-4-validation.md) records agreement
between dense and blocked CPU solves and compiler storage inspection. For its
257×383 and 514×766 cases, blocked compiler-accounted bytes were 53,492 and
95,092, against 2,479,028 and 9,879,132 for dense execution. The blocked CPU path
was slower in these small cases. Compiler storage is not measured GPU peak live
memory. Representation changes already removed the full cost/coupling allocation
from the selected point-cloud solve without requiring a new kernel language.

The [Stage 6 report](../development/stage-6-validation.md) establishes collective
and restart correctness on virtual CPU devices and two actual CPU processes.
It does not establish GPU speedup, GPU capacity, peer-link performance or Slurm
reliability. No GPU operation profile exists here, so no measured runtime fraction
can honestly be inserted into a native selection calculation. The tools below
prepare that calculation; their examples use synthetic values.

## Implementation choices

| Choice | When it would earn its place | Cost and validation needed |
|---|---|---|
| Ordinary JAX | Existing algorithms, blocked representation, library operations and compiler fusion meet the workload | Baseline precision, allocation, compilation and synchronized execution measurements |
| Pallas Mosaic GPU | A measured tile/reduction/fusion bottleneck benefits from explicit memory and scheduling control | Exact JAX/runtime/GPU/dtype matrix, numerical and transform tests, fallback; no assumption that a Hopper specialization is optimal or valid on every successor |
| C++/CUDA through JAX FFI | A custom kernel or established native library supplies a measured benefit or missing capability | Packaging/ABI, XLA stream use, layouts, aliasing, errors, transformations and total integration cost |
| Existing Fortran library through a C ABI | Its algorithm or proven implementation wins for the actual problem | Measure host/device transfers, conversion, communication and full solve; test language ABI and library/runtime versions |
| Larger native quantum solver loop | Full spectral work or distributed eigensolver capability requires a different ownership/execution model | End-to-end state, data distribution, collectives, restart, precision and full spectral correctness; a kernel-only comparison is insufficient |

JAX lowers computations through XLA, whose GPU pipeline includes generated code
and library calls. Rewriting the same operation in C++ does not by itself prove
a faster device execution path. This is our engineering inference from the
[OpenXLA GPU architecture](https://openxla.org/xla/gpu_architecture).

Current JAX documentation describes Pallas as experimental and Mosaic GPU as
supporting Hopper and newer GPUs. APIs and specialized instructions still need
version-specific testing. See the [Pallas quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html)
and [Mosaic GPU reference](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)
(documentation checked 2026-09-21).

JAX FFI can pass the XLA CUDA stream into a native handler. FFI integration also
needs explicit treatment of batching, differentiation and partitioning; calling
foreign code does not prove those contracts. See the
[JAX FFI guide](https://docs.jax.dev/en/latest/ffi.html). Fortran has standardized
C interoperability facilities, but those do not imply CUDA execution or remove
transfer costs: see the [GNU Fortran interoperability guide](https://gcc.gnu.org/onlinedocs/gfortran/Interoperability-with-C.html).

## Experiment and admission sequence

1. Capture the exact GPU model, driver, CUDA/JAX/jaxlib versions, dtype, memory
   policy, allocation and peer topology using the existing inventory/benchmark
   tools. Use the same scientific input, objective, tolerance and output policy.
2. Profile complete representative solves after compilation/warmup, with
   synchronized timings and at least seven repetitions. Preserve raw samples,
   compile/setup/checkpoint costs and profiles. Count non-overlapping critical-path
   time; summing overlapping kernel spans gives an invalid runtime fraction.
3. Screen operations consuming at least 30% of solve time. Amdahl's estimate is
   `1 / (1-p+p/k+h/T)`, for fraction `p`, kernel speedup `k`, overhead `h` and
   original solve time `T`. Even eliminating a 30% operation entirely yields
   only about 1.43× without overhead, below the 1.5× speed gate.
4. Tune representation, block sizes, JAX composition and existing library calls
   first. Profile again: the dominant operation can change. Compare appropriate
   JAX/Pallas/native candidates for the actual bottleneck, including launch,
   layout conversion, transfer and communication costs.
5. Admit on **at least 1.5× matched-accuracy full-solve speedup**, **at least 25%
   isolated full-solve peak-live-memory reduction**, or a separately justified
   required capability. Kernel-only gains do not qualify. No unexplained
   regression above 10% on the representative workload matrix is acceptable;
   document any deliberately restricted dispatch domain and preserve fallback.
6. Complete the interface matrix below before making the backend available as
   supported. Passing the measurement tool alone is not release certification.

Use separate experiments for many independent jobs and one large solve.
Increasing job throughput does not establish a single-solve speedup. Include
small/large and rectangular classical cases, difficult epsilon/support patterns,
flow chunk/restart workloads, and quantum dimensions/complex dtypes where the
candidate claims applicability. Retain failed/convergence-capped outcomes as
failures, never as fast timing samples.

## Prepared tools

From the checkout (standard library only):

```bash
python benchmarks/native_gate.py screen \
  --operation-seconds 8 --solve-seconds 10 --kernel-speedup 4 \
  --overhead-seconds 1
python benchmarks/native_gate.py compare baseline.json candidate.json
python benchmarks/native_gate.py compare baseline.json candidate.json \
  --baseline-peak baseline-peak.json --candidate-peak candidate-peak.json
```

The first example predicts 2× **conditionally from synthetic assumptions**.
`screen` exits zero for a valid calculation, even below the admission threshold.
`compare` accepts existing schema-version-2 records from the baseline tools,
checks numerical convergence/objective agreement and matched science/runtime/
hardware, requires at least seven samples, and exits one when the measurement
gate fails (invalid input exits two). Its JSON includes SHA256 hashes of evidence
files and the remaining interface/release checks. Source revisions may differ;
comparison fields must match. Objective tolerances default to `atol=1e-8`,
`rtol=1e-6`; choose justified tighter values where the problem requires them.
The Stage 6 distributed report has a different schema and must not be passed as
a schema2 record or compared across device counts as a native-kernel claim.

Optional peak evidence must contain `measurement_kind: isolated_peak_live_bytes`,
`scope: full_solve`, a positive `peak_live_bytes`, and every field in
`benchmarks.reporting.COMPARISON_FIELDS` copied from the corresponding solve
record. Include the same `flow` contract if present. This binds objective,
shape, epsilon, tolerance, iteration/output policy, input identity, dtype,
hardware and runtime, not just a filename. Both baseline and candidate evidence
are required. Attach the isolated profiler capture, measurement procedure and
source revision to the experiment record for review; JSON declarations alone
cannot prove the measurement was actually isolated. Compiler byte estimates and
process-cumulative allocator counters are rejected and cannot substitute for
this evidence. Missing memory evidence remains unknown, never zero.

## Required matrix for any future selected component

| Contract | Required evidence |
|---|---|
| Installation and fallback | Base install remains usable without native extras; optional wheel/source build and missing/incompatible binary behavior tested; fallback numerically equivalent |
| Platform and ABI | Explicit GPU architectures, driver/CUDA/JAX/jaxlib/compiler/library versions, dtype/shape support and ABI checks; test each advertised successor architecture |
| Streams and ownership | Launch on supplied XLA stream, correct lifetimes/events/workspaces, no hidden default-stream race; concurrent/asynchronous repeated-call tests |
| Layout and aliasing | Accepted strides/layouts, explicit conversion cost, donation/alias guarantees, output shape/dtype and repeated-call noncorruption |
| Failure handling | Invalid shape/dtype, nonfinite inputs, library errors and allocation failure handled without fabricated convergence; preserve public status semantics |
| Transformations | `jit`, batching and sharding contracts tested or explicitly rejected; no silent gathering or accidental replication; AD advertised only with independent derivative checks |
| Numerics | Independent objectives/residuals, difficult precision/support cases and exact/warm restart invariants; no reduced-precision shortcut concealed as the same method |
| Performance | Raw matched samples, full-solve overhead, isolated peak evidence if used, representative regression checks and dispatch-domain documentation |

Quantum transport still requires the full spectral functions for the dense
entropy path. A leading eigenpair does not replace a matrix exponential, log or
full PSD projection. [Quantum feasibility](quantum-feasibility.md) documents
dense storage limits and the cuSOLVERMp investigation. Distributed classical
Sinkhorn does not imply distributed quantum eigensolver support. No native
quantum backend, sparse/chordal equivalence or quantum gradient flow is selected
for this release.
