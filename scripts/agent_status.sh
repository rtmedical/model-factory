#!/usr/bin/env bash
# Single-screen status summary across one or more DatasetSpecs.
# Optimised for Claude agents — tabular text, no JSON-only output.
#
# Usage:
#   scripts/agent_status.sh                                 # all registered specs
#   scripts/agent_status.sh pelvis_male_prostate            # one spec
#   scripts/agent_status.sh pelvis_male_prostate hn_pddca_optic  # multiple
#   scripts/agent_status.sh --watch <spec_keys...>
#
# Reads:
#   /data/model-factory-nfs/results/<DatasetName>/*/fold_<N>/metrics.jsonl
#   kubectl -n model-factory get jobs -l app=factory-train

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FACTORY_ROOT="${FACTORY_ROOT:-/data/model-factory-nfs}"

WATCH=0
SPEC_KEYS=()
for arg in "$@"; do
  case "$arg" in
    --watch) WATCH=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) SPEC_KEYS+=("$arg") ;;
  esac
done

# Resolve spec key → DatasetName via the Python registry. If no keys
# given, list every registered spec.
resolve_specs() {
  PYTHONPATH="$ROOT/src" python3 - "$@" <<'PY'
import sys
from modelfactory.datasets.specs import SPECS
keys = sys.argv[1:] or list(SPECS.keys())
for k in keys:
    s = SPECS.get(k)
    if s is None:
        print(f"# UNKNOWN_SPEC {k}", file=sys.stderr)
        continue
    print(f"{k}\t{s.dataset_id}\t{s.folder}")
PY
}

print_dataset() {
  local key="$1" dataset_id="$2" name="$3"
  local results_root="$FACTORY_ROOT/results/$name"
  echo ""
  echo "=== ${key} (d${dataset_id} → $name) ==="
  if [[ ! -d "$results_root" ]]; then
    echo "  (no results dir yet at $results_root)"
    return
  fi
  printf "  %-6s %-10s %-7s %-10s %-10s %s\n" "FOLD" "JOB" "EPOCH" "VAL_LOSS" "MEAN_DICE" "PER_CLASS"
  for fold in 0 1 2 3 4; do
    local jsonl
    jsonl="$(find "$results_root" -path "*/fold_${fold}/metrics.jsonl" 2>/dev/null | head -1)"

    local job_name="train-d$(printf '%03d' "$dataset_id")-f${fold}"
    local job_status="-"
    if command -v kubectl >/dev/null 2>&1; then
      job_status="$(kubectl -n model-factory get job "$job_name" \
        -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null || true)"
      [[ -z "$job_status" ]] && job_status="run"
    fi

    if [[ -z "$jsonl" || ! -s "$jsonl" ]]; then
      printf "  %-6s %-10s %-7s %-10s %-10s %s\n" "$fold" "$job_status" "-" "-" "-" "-"
      continue
    fi

    local last
    last="$(grep -E '"phase":\s*"val"' "$jsonl" | tail -1)"
    if [[ -z "$last" ]]; then
      printf "  %-6s %-10s %-7s %-10s %-10s %s\n" "$fold" "$job_status" "-" "-" "-" "(no val yet)"
      continue
    fi

    local epoch vloss mdice pcd
    epoch="$(echo "$last" | jq -r '.epoch // "-"')"
    vloss="$(echo "$last" | jq -r '(.val_loss | tostring)[0:8] // "-"')"
    mdice="$(echo "$last" | jq -r '(.mean_fg_dice | tostring)[0:6] // "-"')"
    pcd="$(echo "$last"   | jq -c '.per_class_dice // {} | to_entries | map("\(.key)=\((.value | tostring)[0:5])") | join(",")')"
    printf "  %-6s %-10s %-7s %-10s %-10s %s\n" "$fold" "$job_status" "$epoch" "$vloss" "$mdice" "$pcd"

    grep -E '"event"' "$jsonl" | tail -3 | while read -r ev; do
      echo "    EVENT: $(echo "$ev" | jq -c '{event, epoch, ts}' 2>/dev/null || echo "$ev")"
    done
  done
}

run_once() {
  date -u +"=== status @ %Y-%m-%dT%H:%M:%SZ ==="
  if command -v kubectl >/dev/null 2>&1; then
    echo "factory-cq quota:"
    kubectl get clusterqueue factory-cq -o jsonpath='{.status.flavorsUsage}' 2>/dev/null \
      | jq -c '.[].resources[]? | {n: .name, total: .total}' 2>/dev/null \
      | sed 's/^/  /' || echo "  (kubectl unavailable)"
    echo ""
    echo "doserad coexistence:"
    kubectl -n doserad get jobs,pods -o wide 2>/dev/null | sed 's/^/  /' || echo "  (no doserad ns)"
  fi

  while IFS=$'\t' read -r key did name; do
    [[ "$key" == \#* ]] && continue
    print_dataset "$key" "$did" "$name"
  done < <(resolve_specs "${SPEC_KEYS[@]}")
}

if (( WATCH )); then
  while true; do clear; run_once; sleep 30; done
else
  run_once
fi
