#!/usr/bin/env bash
# Invoked within the site job step, after CPU/GPU binding is established.
set -euo pipefail
: "${EROT_CONFIG:?Set EROT_CONFIG}"
: "${EROT_RUN_DIRECTORY:?Set EROT_RUN_DIRECTORY}"
: "${SLURM_NTASKS:?Missing process count}"
: "${SLURM_PROCID:?Missing process rank}"
command=("${EROT_PYTHON:-python}" -m erot.distributed "$EROT_CONFIG"
  --run-directory "$EROT_RUN_DIRECTORY" --device "${EROT_DEVICE:-gpu}")
if (( SLURM_NTASKS > 1 )); then
  : "${EROT_COORDINATOR:?Set a reachable coordinator host:port shared by all ranks}"
  command+=(--coordinator "$EROT_COORDINATOR" --processes "$SLURM_NTASKS"
    --process-id "$SLURM_PROCID" --local-device-ids "${EROT_LOCAL_DEVICE_IDS:-0}")
fi
exec "${command[@]}" "$@"
