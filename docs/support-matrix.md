# 0.2 support and validation matrix

This is a research release with validated CPU implementations and prepared GPU
validation tools. Hopper-or-newer NVIDIA GPUs are the optimization target;
no actual GPU, Slurm allocation or interconnect was available for this release
work. A prepared test or launch script is not a validated hardware combination.

## Numerical capabilities

| Capability | API / status | Limits and evidence |
|---|---|---|
| Dense Shannon classical OT | `erot.solve`, `erot.solvers.solve_sinkhorn`; CPU validated | Rectangular and multi-marginal costs, zero supports; dense tensor memory grows as product of marginal sizes |
| Dense quadratic classical OT | `erot.solve`; CPU validated | Two marginals, cyclic dual projection; explicit dense cost/coupling |
| Implicit Shannon classical OT | `solve_blocked_sinkhorn`; CPU validated | Two point clouds with squared Euclidean cost, or tiled existing dense cost; no full plan allocated for point clouds; arithmetic remains quadratic |
| Unregularized JKO | `erot.flows`, PDHG; CPU validated | Entropy/quadratic/potential energies on cell masses; dense coupling state; independent discrete references |
| Finite-epsilon JKO | `erot.flows`, Sinkhorn; CPU validated | Different objective from unregularized JKO; dense or blocked inner transport, positive-density optimization, explicit inner/line-search failures |
| Quadratic QOT | `erot.solve`; CPU validated | General Hermitian trace-one density matrices, including rank-deficient inputs and unequal subsystem dimensions; full dense PSD projections |
| Entropy QOT | `erot.solve` with `von_neumann` / `dual`; CPU validated | Strictly positive-definite trace-one marginals, unequal dimensions and complex values; dense full eigendecomposition on each trial |
| Independent experiments | `erot-run`, `erot-launch`, `erot-array`; CPU workflows validated | Configurable CPU/GPU resources; GPU throughput and real Slurm submission untested |
| Distributed individual solve | `erot-distributed`; experimental | Row-sharded point-cloud Shannon Sinkhorn; virtual CPU devices and two real CPU processes validated; replicated target vectors/cloud limit capacity; actual GPUs/nodes untested |
| Native acceleration | No selected component | JAX retained; [future admission gates](performance/native-decision.md) prepared |

No quantum gradient flow, distributed PDHG, distributed quantum eigensolver,
chordal equivalence, low-rank entropy approximation, separable-grid metric, or
end-to-end differentiable solve is advertised. The historical
`erot.experimental.classical` module is retained for compatibility, not promoted
by tests of the new solver paths.

## Runtime and platform policy

The base distribution declares Python >=3.11, JAX >=0.4.30 and NumPy >=1.26.
The [minimum runtime constraints](../requirements/base-minimum.txt) pin
JAX/jaxlib0.4.30, NumPy1.26.4, SciPy1.12.0, ml-dtypes0.4.1 and opt-einsum3.3.0
for Python3.11 validation. The full current CPU environment uses Python3.12.13,
JAX/jaxlib0.11.2, NumPy2.5.3, SciPy1.18.1 and ml-dtypes0.6.0.
See [release evidence](development/stage-8-validation.md) for actual completed
commands, counts and installed-wheel results, rather than assuming every
intermediate version or operating system has been tested.

Experimental distributed mode uses newer `shard_map` and process-local-array
APIs. Its validated CPU runtime is pinned separately in
[distributed-cpu.txt](../requirements/distributed-cpu.txt), for Python3.12.
The base minimum is not a distributed support promise. The driver reports an
explicit runtime requirement when the required APIs are absent. Linux process
launch/affinity and POSIX filesystem checkpoint behavior are the exercised
platform. Windows/macOS, heterogeneous device meshes, elastic/resharded restart,
process-loss recovery inside collectives and multi-node fabrics are unvalidated.

CUDA extras (`cuda12`, `cuda13`) select JAX's GPU dependencies; they are not
lockfiles or proof of a working driver/GPU combination. Follow the current
[JAX installation requirements](https://docs.jax.dev/en/latest/installation.html),
then capture the installed environment and run the checks below. Allocate CPUs,
GPUs, nodes, memory and scheduler options per site; the older one/two-GPU and
16-core setup is an example, not a library limit. Do not infer peer links or
bandwidth from a GPU model name.

## Precision, transformations and state

The validated host wrapper defaults to float64 and enables JAX x64 before input
conversion. Pure kernels follow array dtype and require the caller to configure
x64 before constructing inputs. Float32 needs accuracy-appropriate tolerances;
complex quantum inputs use the corresponding complex precision. Reduced
precision/mixed precision is not an advertised solver mode.

Pure dense Sinkhorn is tested under `jit`, `vmap` and a short `scan`; quantum
entropy has JIT/batching checks. Flow chunks use static step/snapshot counts and
retain full continuation state. None of these establishes reverse-mode AD through
a dynamically terminated solve. Independent derivative checks of energy or dual
formulas are different from differentiating the full optimizer. The validated
host `erot.solve` wrapper inspects/synchronizes inputs and is not the interface
to wrap inside `jit`.

Exact resume requires unchanged inputs, objective, numerical controls, dtype,
source/runtime identity and compatible topology. Warm starts are separate and
only reuse compatible potentials for a new problem. A saved coupling or a flow
snapshot alone cannot reconstruct Dykstra/PDHG/entropy continuation state.
Workers validate manifests and never advance physical time after a failed step.

## Hardware validation when an allocation is available

Run GPU tests only in the target CUDA environment. Keep CPU fallback available
for independent references where the tests require it:

```bash
JAX_PLATFORMS=cuda,cpu python -m pytest tests/gpu -q
python benchmarks/distributed_sinkhorn.py --help
python benchmarks/experiment_throughput.py --help
```

Follow the [baseline protocol](performance/baseline-protocol.md),
[blocked transport protocol](performance/blocked-transport.md),
[distributed transport protocol](performance/distributed-transport.md), and
[HPC launch instructions](../hpc/README.md). Validate one GPU, one controller
with two GPUs, and two processes with one GPU each before considering multiple
nodes. Record scientific agreement, actual isolated peak live memory,
communication share, synchronized samples and restart failures. Report compiler
bytes and cumulative allocator counters under their own names. Promote only the
actual runtime/device/topology combinations measured. These hardware gates are
open; the repository currently supplies tools to execute them.
