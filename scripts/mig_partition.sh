#!/usr/bin/env bash
# Create / destroy MIG slices on a single GPU. Privileged (needs root / sudo).
#
# This is the ONLY part of the bootstrap that mutates GPU hardware state, so it
# is deliberately a thin, explicit shell script you run intentionally — never
# invoked implicitly by `infra render`. MIG layout is NOT reboot-persistent;
# re-run after a node reboot.
#
# Usage:
#   mig_partition.sh create  --gpu 1 --profile 3g.40gb --count 2
#   mig_partition.sh destroy --gpu 1
#   mig_partition.sh disable --gpu 0
#
# Multi-node: set MIG_SSH_HOST=user@host to run the nvidia-smi calls remotely.
#
# WARNING: destroying / re-partitioning a GPU that has running CUDA processes
# (training OR inference OR vLLM) will kill them. Drain first.
set -euo pipefail

ACTION="${1:-}"; shift || true
GPU=""; PROFILE="3g.40gb"; COUNT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SMI=(nvidia-smi)
if [[ -n "${MIG_SSH_HOST:-}" ]]; then
  SMI=(ssh "${MIG_SSH_HOST}" nvidia-smi)
fi

[[ -n "$GPU" ]] || { echo "ERROR: --gpu is required" >&2; exit 2; }

case "$ACTION" in
  create)
    [[ -n "$COUNT" ]] || { echo "ERROR: --count is required for create" >&2; exit 2; }
    echo ">> enabling MIG on GPU $GPU"
    "${SMI[@]}" -i "$GPU" -mig 1 || true
    # Idempotent: if the requested number of GIs already exist, do nothing.
    existing=$("${SMI[@]}" mig -i "$GPU" -lgi 2>/dev/null | grep -c "$PROFILE" || true)
    if [[ "$existing" -ge "$COUNT" ]]; then
      echo ">> GPU $GPU already has $existing '$PROFILE' GIs (>= $COUNT requested); skipping"
      exit 0
    fi
    # Build a comma-separated profile list of length COUNT and create + wire
    # compute instances in one shot (-C).
    spec="$PROFILE"; for ((i=1; i<COUNT; i++)); do spec="$spec,$PROFILE"; done
    echo ">> creating $COUNT x '$PROFILE' on GPU $GPU"
    "${SMI[@]}" mig -i "$GPU" -cgi "$spec" -C
    echo ">> done. Current slices:"
    "${SMI[@]}" -L | sed -n "/^GPU $GPU:/,/^GPU /p" | grep MIG || true
    ;;
  destroy)
    echo ">> destroying compute + GPU instances on GPU $GPU"
    "${SMI[@]}" mig -i "$GPU" -dci || true
    "${SMI[@]}" mig -i "$GPU" -dgi || true
    ;;
  disable)
    echo ">> disabling MIG on GPU $GPU"
    "${SMI[@]}" -i "$GPU" -mig 0
    ;;
  *)
    echo "usage: $0 {create|destroy|disable} --gpu N [--profile P] [--count C]" >&2
    exit 2
    ;;
esac
