# Stage 7 evidence-based native acceleration decision

> Execute inline under the full-roadmap authorization, using
> superpowers:executing-plans. No native implementation is a valid outcome when
> no measured bottleneck or selected required capability justifies one.

**Goal:** make the JAX/native choice explicit and provide reproducible admission
calculations for future Hopper-or-newer profiling and candidate measurements.
**Architecture:** reuse the existing scientific/environment benchmark comparison
contract; add a small pure measurement gate and a documented decision record.
Do not scaffold native build systems or adapters without a selected operation.
**Tech stack:** standard-library reporting and existing JAX benchmark evidence.
**Spec:** ../specs/2026-09-21-erot-flows-hpc-design.md, backend admission criteria;
roadmap Stage7; docs/performance/quantum-feasibility.md.

## Global constraints

No GPU profile or real GPU peak-live-memory evidence exists in this workspace.
Do not claim JAX is maximally fast or that C++/Fortran can never be needed.
Retain source compatibility and optional-dependency boundaries. A native
performance extension requires >=1.5x matched-accuracy full-solve speedup or
>=25% measured peak-live-memory reduction, or a separately documented required
capability. Kernel speed alone and compiler-buffer estimates do not qualify.
No new architecture-specific code without its own runtime/dtype/fallback tests.

## Review focus

A large isolated-kernel gain may not pass the full-solve threshold.
Compiler bytes or cumulative allocator high-water values must not masquerade as
measured isolated peak-live memory.
Nonconverged or mismatched scientific/hardware reports cannot win admission.
Profile time fractions require consistent scopes and explicit overlap handling.
A measurement gate is not a substitute for ABI/stream/transform support tests.

## Task 1: native screening and measurement tools

Files: benchmarks/native_gate.py, tests/unit/test_native_gate.py.
Consume benchmarks.reporting.assert_comparable/validate_record schema2 reports.
Produce pure estimate_speedup(operation_seconds,solve_seconds,kernel_speedup,
overhead_seconds=0) and compare_candidate(candidate,baseline,objective_atol,
objective_rtol,baseline_peak=None,candidate_peak=None).

- [ ] Write missing-module tests: eliminating30% of runtime cannot reach1.5x;
  fraction0.8 at4x yields2.5x absent overhead; overhead reduces the prediction.
  Reject nonfinite/negative measurements and fractions outside[0,1].
- [ ] Implement Amdahl screening with explicit assumptions. CLI labels the
  prediction conditional on supplied profile measurements; it is not a result.
- [ ] Test candidate/baseline mismatch, nonconvergence, objective disagreement,
  1.5x speed threshold,25% peak-memory threshold, and invalid memory provenance.
  Use existing schema2 contract; raw sample medians and numerical tolerances
  must validate before comparison.
- [ ] Accept memory evidence only with an explicit isolated_peak_live_bytes
  measurement kind and matching case/hardware identity. Reject compiler or
  process-cumulative counters. No missing memory value becomes zero.
- [ ] Report pass/fail for the measurement gate, never automatic backend release
  approval. Preserve evidence file hashes in CLI output and list unmet gates.
- [ ] Run focused checks and commit the tool.

## Task 2: explicit backend decision, validation matrix and review

Files: docs/performance/native-decision.md, docs/development/stage-7-validation.md,
roadmap and support documentation.

- [ ] Record current evidence: dense/blocked CPU reference agreement, bounded
  compiler storage, virtual-device/process collectives and unmeasured GPU limits.
  Select JAX for the current release; no optional native operation is selected.
- [ ] Compare the role and required evidence of ordinary JAX, Pallas Mosaic,
  CUDA/C++ FFI, existing Fortran/C-ABI libraries and a larger native quantum loop.
  Use current primary documentation for version-sensitive capability statements.
- [ ] Define target experiments: profile complete representative solves, screen
  >=30% operations with Amdahl, tune representation/JAX/library calls first,
  then compare candidates at matched precision/accuracy and total overhead.
- [ ] For a future selected backend require stream ownership, alias/layout/error
  contracts, optional installation, exact runtime/architecture matrix, fallback,
  and explicit batching/sharding/derivative support or rejection. Include the
  no-unexplained->10%-regression rule across representative workloads.
- [ ] State quantum dense-spectral limit and cuSOLVERMp investigation separately;
  do not claim a distributed eigensolver or substitute an extremal eigenpair.
- [ ] Run reporting tests, Ruff and documented CPU tool examples. Request one
  fresh final review, fix material findings, commit evidence and proceed Stage8.

## Plan self-review

The no-native outcome is explicitly permitted by the roadmap. Stage7 does not
invent an operation profile, measured GPU speedup or memory reduction. It supplies
admission tools and a concrete selection decision, while runtime/ABI tests remain
conditional on an actual future native component. User-confirmed lack of GPU
access is handled by prepared validation tools and explicit evidence gaps.
