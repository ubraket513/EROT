# Independent experiments on Slurm

Install EROT into the Python environment used by each job. Resource allocation
belongs to the cluster: pass the site's account, partition, time, memory and
GPU flags to sbatch. There are no fixed GPU models or CPU counts in the script.

For example, from the repository root on a configured cluster:

```bash
export EROT_CONFIG_LIST="$PWD/hpc/example-array.json"
export EROT_RESOURCE_PROFILE="$PWD/hpc/profiles/gpu.json"
export EROT_OUTPUT_ROOT="$PWD/results/array"
export EROT_PYTHON="$(command -v python)"
sbatch --array=0-1 --cpus-per-task=8 --gpus-per-task=1 hpc/run_array.sbatch
```

The example numbers are choices, not requirements. Add the site-specific flags
and request one task per array element. Every element selects the corresponding
zero-based manifest entry and launches one worker. Manifest paths are relative
to the manifest directory. Configurations must have distinct scientific run
identities, typically different names or seeds. Existing results require an
explicit `--resume` argument after the script path.

Profiles contain `device` (cpu or gpu), and optional positive `cpu_budget` and
`threads_per_worker`. Unspecified budgets use the current process affinity,
bounded by SLURM_CPUS_PER_TASK. Thread limits are set before numerical imports.
GPU profiles use scheduler-scoped CUDA_VISIBLE_DEVICES, preserving opaque UUID,
MIG or ordinal tokens. A missing visible allocation fails explicitly. Sites
that expose GPU visibility only within srun must invoke the Python command in
that job step using their local launch convention; do not invent physical GPU
ordinals. See the official [Slurm GPU resource documentation](https://slurm.schedmd.com/gres.html)
and [array documentation](https://slurm.schedmd.com/job_array.html).

A CPU-only template smoke can use the cpu profile and execute the script with
SLURM_ARRAY_TASK_ID=0 and SLURM_CPUS_PER_TASK set to the intended budget. This
checks selection and process isolation, not scheduler submission or GPU access.
The supplied templates have not been run on an actual Slurm allocation yet.

For several independent experiments within one allocation, use `erot-launch`
with several config paths. It defaults to one worker per visible allocated GPU
with `--device gpu`; use `--workers` to reduce concurrency. CPU operation defaults
to one worker and permits explicit `--workers 2`. Resources must fit the actual
allocation; oversubscription is rejected. Each worker gets disjoint CPU affinity
and one GPU token. This is independent experiment parallelism, not a distributed
individual solve.

## One distributed classical solve

The experimental erot-distributed command cooperates across all ranks. Use
experiments/configs/distributed-classical.json, a shared EROT_RUN_DIRECTORY and
EROT_CONFIG path. Set EROT_RANK_SCRIPT to the absolute shared path of
hpc/distributed_rank.sh (for example, "$PWD/hpc/distributed_rank.sh" from the
repository root). Slurm spools the batch script alone, so a sibling path beside
that copied script is not valid. hpc/run_distributed.sbatch starts a bound srun job step;
hpc/distributed_rank.sh maps SLURM_NTASKS and SLURM_PROCID to explicit JAX process
arguments. Set EROT_DEVICE=gpu for CUDA (the rank script defaults to gpu), and
EROT_PYTHON to the installed interpreter. Request the desired resources using
site flags, e.g. one node, two tasks, one GPU per task and an appropriate number
of CPUs per task. The numbers are allocation choices, not library limits.

For more than one process, set EROT_COORDINATOR to a reachable host:port on the
allocated coordinator node before invoking the rank script. The site wrapper
can derive that hostname from its allocation; choose a free port permitted by
cluster policy. This template does not guess a network interface or port.
Every rank must see the same coordinator and shared run path. Proxy/network
configuration must permit the JAX coordination and collective connections.

srun establishes CPU and GPU visibility before Python imports numerical code.
The default local device ID is 0 inside each task's visible allocation, suitable
for one GPU per process; EROT_LOCAL_DEVICE_IDS can specify a comma-separated
list for another explicitly allocated layout. CUDA_VISIBLE_DEVICES is preserved.
With one process the rank script omits distributed initialization and uses all
visible devices, allowing a one-controller/two-GPU validation run.

Pass --stop-after-chunks 1 after the script path to test a deliberate checkpoint
boundary, then --resume on the next launch with the same configuration/software
and topology. Do not reuse a run directory for a different topology. Multiple
ranks share one global ownership lock held by rank zero and publish one complete
checkpoint manifest. Per-rank resource reports record actual affinity and thread
settings. Actual Slurm, GPU peer communication and interconnect validation have
not run. Local CPU tests cover both one-process/two-device and two-process modes.
