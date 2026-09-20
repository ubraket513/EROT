#!/usr/bin/env bash
# Supply array range, CPUs, GPUs, account and partition through site sbatch flags.
# Each array element runs one independent experiment on its allocated resources.
set -euo pipefail
: "${EROT_CONFIG_LIST:?Set EROT_CONFIG_LIST to the configuration manifest}"
: "${EROT_RESOURCE_PROFILE:?Set EROT_RESOURCE_PROFILE to a resource profile}"
: "${EROT_OUTPUT_ROOT:?Set EROT_OUTPUT_ROOT to a shared output directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm job array}"
# Inherit scheduler-scoped CUDA_VISIBLE_DEVICES unchanged. The launcher chooses
# one visible token and bounds affinity before any numerical backend import.
exec "${EROT_PYTHON:-python}" -m erot.slurm \
  --configs "$EROT_CONFIG_LIST" \
  --profile "$EROT_RESOURCE_PROFILE" \
  --output-root "$EROT_OUTPUT_ROOT" "$@"
