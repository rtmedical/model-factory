#!/usr/bin/env bash
# claim-mig.sh — atomically lease one MIG slice for this Ray worker pod.
#
# Runs as an init container in every Ray worker pod (see
# infra/kustomize/factory-ray-cluster.yaml). On success, writes a single line
# `export NVIDIA_VISIBLE_DEVICES=MIG-<uuid>` to /shared/mig.env which the main
# Ray container sources at startup. On failure (no free slice), exits non-zero
# and lets the pod fail; kubelet will restart it and retry.
#
# Atomicity contract: a JSON-Patch `test` op verifies the slot is still
# "available" before the `replace` op stamps this pod's name into it. If two
# pods race for the same slot, exactly one wins; the loser moves on.

set -euo pipefail

NS="${POD_NAMESPACE:-model-factory}"
POD="${POD_NAME:?POD_NAME env required (downward API)}"
UUIDS_FILE="${UUIDS_FILE:-/etc/mig/uuids.txt}"
OUT="${OUT:-/shared/mig.env}"
LEASE_CM="${LEASE_CM:-factory-mig-leases}"
MAX_TRIES="${MAX_TRIES:-3}"

mkdir -p "$(dirname "$OUT")"

if [[ ! -r "$UUIDS_FILE" ]]; then
  echo "[claim-mig] missing $UUIDS_FILE" >&2
  exit 2
fi

# Collect candidate UUIDs, skip comments and blanks.
mapfile -t CANDIDATES < <(grep -Ev '^[[:space:]]*(#|$)' "$UUIDS_FILE")
if (( ${#CANDIDATES[@]} == 0 )); then
  echo "[claim-mig] no UUIDs in $UUIDS_FILE" >&2
  exit 2
fi

# Shuffle once per pod to reduce thundering-herd collisions when many workers
# come up together (deterministic permutation seeded by pod name).
shuf_seed="$(echo -n "$POD" | cksum | awk '{print $1}')"
mapfile -t CANDIDATES < <(printf '%s\n' "${CANDIDATES[@]}" | shuf --random-source=<(yes "$shuf_seed" 2>/dev/null) || printf '%s\n' "${CANDIDATES[@]}")

for try in $(seq 1 "$MAX_TRIES"); do
  for uuid in "${CANDIDATES[@]}"; do
    current="$(kubectl -n "$NS" get cm "$LEASE_CM" -o "jsonpath={.data.${uuid}}" 2>/dev/null || echo "")"
    if [[ "$current" != "available" ]]; then
      continue
    fi
    # Atomic test+replace. Returns non-zero if the lease changed under us.
    if kubectl -n "$NS" patch cm "$LEASE_CM" --type json -p \
        "[{\"op\":\"test\",\"path\":\"/data/${uuid}\",\"value\":\"available\"},
          {\"op\":\"replace\",\"path\":\"/data/${uuid}\",\"value\":\"${POD}\"}]" \
        >/dev/null 2>&1; then
      echo "[claim-mig] $POD claimed $uuid"
      {
        echo "export NVIDIA_VISIBLE_DEVICES=${uuid}"
        echo "export NVIDIA_DRIVER_CAPABILITIES=compute,utility"
        echo "export MFACTORY_MIG_UUID=${uuid}"
      } > "$OUT"
      exit 0
    fi
  done
  echo "[claim-mig] try ${try}: all slots taken or contended, sleeping..." >&2
  sleep $((RANDOM % 10 + 5))
done

echo "[claim-mig] failed to claim any MIG slice after $MAX_TRIES tries" >&2
exit 1
