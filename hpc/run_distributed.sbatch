#!/usr/bin/env bash
# Request task/CPU/GPU resources using site sbatch flags. No site defaults here.
set -euo pipefail
: "${EROT_CONFIG:?Set EROT_CONFIG to the classical blocked configuration}"
: "${EROT_RUN_DIRECTORY:?Set a shared run directory}"
: "${SLURM_NTASKS:?Request the intended process count}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec srun --cpu-bind=cores bash "$script_dir/distributed_rank.sh" "$@"
