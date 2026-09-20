#!/usr/bin/env bash
# Request task/CPU/GPU resources using site sbatch flags. No site defaults here.
set -euo pipefail
: "${EROT_CONFIG:?Set EROT_CONFIG to the classical blocked configuration}"
: "${EROT_RUN_DIRECTORY:?Set a shared run directory}"
: "${SLURM_NTASKS:?Request the intended process count}"
: "${EROT_RANK_SCRIPT:?Set the absolute shared path to hpc/distributed_rank.sh}"
# sbatch executes a spooled copy; sibling paths relative to this file are invalid.
if [[ "$EROT_RANK_SCRIPT" != /* || ! -r "$EROT_RANK_SCRIPT" ]]; then
  printf '%s\n' 'EROT_RANK_SCRIPT must name a readable absolute shared script' >&2
  exit 2
fi
exec srun --cpu-bind=cores bash "$EROT_RANK_SCRIPT" "$@"
