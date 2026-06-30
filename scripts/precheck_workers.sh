#!/bin/bash
# precheck_workers.sh — sanity-check every Ray worker before submitting a campaign.
#
# WHY THIS EXISTS:
# Newly-recreated or long-idle Ray worker pods can fail any new trial with
# either:
#   (a) `RuntimeError: No CUDA GPUs are available` (NVML cold-start)
#   (b) `Could not find requested nnunet trainer nnUNetTrainerSmallStructuresMLflow`
#       (the trainer image was built before the shim was added; long-lived
#       workers were manually patched; recreated workers come up cold)
#
# Ray Tune does NOT retry on a different worker after these failures — the
# trial errors out and the Tune driver Job sits in `Running` for hours
# because of a pandas summary bug in the failed-trial post-mortem. You
# lose GPU time silently.
#
# Run this BEFORE every `modelfactory campaign smoke` / `campaign run-trio`
# submission. Exits 0 if all workers pass, 1 if any need a fixup.
#
# Auto-fix mode: pass `--fix` to apply scripts/patch_worker.sh to every
# failing worker before re-checking.
#
# Usage:
#   ./scripts/precheck_workers.sh                  # report only
#   ./scripts/precheck_workers.sh --fix            # auto-patch failures
#   ./scripts/precheck_workers.sh --fix --recycle  # additionally kubectl-delete
#                                                  # any worker whose nvidia-smi
#                                                  # also fails (NVML cold)

set -uo pipefail

NS=${NS:-model-factory}
FIX=0
RECYCLE=0
for arg in "$@"; do
  case "$arg" in
    --fix)     FIX=1 ;;
    --recycle) RECYCLE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

PATCH=${PATCH:-/data/model-factory-nfs/raw_external/longitudinal_ct_tubingen/patch_worker.sh}
[ -x "$PATCH" ] || PATCH="$(dirname "$0")/patch_worker.sh"

fail=0
recycle_pods=()
patch_pods=()

echo "=== Ray worker precheck ($(date -Iseconds)) ==="
for w in $(kubectl -n "$NS" get pods -o name 2>/dev/null | grep factory-ray-mig | cut -d/ -f2 | sort); do
  # 1. SmallStructures trainer import
  err_import=$(kubectl -n "$NS" exec "$w" -c ray-worker -- \
    python3 -c "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerSmallStructuresMLflow, nnUNetTrainerPartialLabelMLflow, nnUNetTrainerPartialLabelBalancedMLflow" 2>&1 | grep -E "ImportError|ModuleNotFound" | head -1)
  # 2. Shim file presence (small-structures, partial-label, and balanced shims)
  has_shim=$(kubectl -n "$NS" exec "$w" -c ray-worker -- bash -c \
    'F=/usr/local/lib/python3.12/dist-packages/nnunetv2/training/nnUNetTrainer/variants/factory; test -f $F/small_structures_trainer.py && test -f $F/partial_label_trainer.py && test -f $F/partial_label_balanced_trainer.py && echo OK')
  needs_patch=0
  [ -n "$err_import" ] && needs_patch=1
  [ "$has_shim" != "OK" ] && needs_patch=1

  if [ "$needs_patch" = "1" ]; then
    echo "  ✗ $w  (import: $err_import / shim: $has_shim)"
    patch_pods+=("$w")
    fail=1
  else
    echo "  ✓ $w"
  fi
done

if [ "$fail" = "0" ]; then
  echo "All workers pass — safe to submit."
  exit 0
fi

if [ "$FIX" != "1" ]; then
  echo
  echo "FAIL — $((${#patch_pods[@]})) worker(s) need patching."
  echo "Re-run with --fix to apply patch_worker.sh, or --fix --recycle to"
  echo "also recycle any worker with stuck NVML state."
  exit 1
fi

echo
echo "=== Applying patch to ${#patch_pods[@]} worker(s) ==="
for w in "${patch_pods[@]}"; do
  "$PATCH" "$w" || true
done

if [ "$RECYCLE" = "1" ]; then
  echo
  echo "=== Recycle policy: not yet implemented (skipping; reserve for cold NVML cases)"
  # TODO: add detection of cold NVML via test that doesn't conflict with the
  # exclusive MIG slice. For now, --recycle is a no-op; recycle by hand:
  #   kubectl -n model-factory delete pod <worker-name>
fi

echo
echo "=== Re-checking ==="
exec "$0"  # rerun without --fix to confirm
