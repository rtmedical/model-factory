#!/usr/bin/env bash
# Submit nnUNetv2 fold-training Jobs for one or more DatasetSpecs,
# pinning each pod to a safe-pool GPU UUID (CLAUDE.md task #12).
# Renders infra/kustomize/train-job.yaml.j2 per (dataset, fold, uuid).
#
# Usage:
#   scripts/submit_folds.sh --spec pelvis_male_prostate
#   scripts/submit_folds.sh --spec pelvis_male_prostate --spec hn_pddca_optic
#   scripts/submit_folds.sh --spec pelvis_male_prostate --dry-run
#   scripts/submit_folds.sh --spec pelvis_male_prostate --max-parallel 4
#
# Requires on the host: kubectl, jq, python3 (for jinja2 rendering),
# and a SAFE_UUIDS_FILE (default /data/model-factory-nfs/safe_uuids.env)
# containing one safe GPU UUID per line.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$ROOT/infra/kustomize/train-job.yaml.j2"
RENDER_DIR="$ROOT/.render/train"
UUIDS_FILE="${SAFE_UUIDS_FILE:-/data/model-factory-nfs/safe_uuids.env}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
PLANS="${PLANS:-nnUNetResEncUNetXLPlans}"
TRAINER="${TRAINER:-nnUNetTrainerMLflow}"

DRY_RUN=0
SPEC_KEYS=()
FOLDS=()  # empty = all 5 folds (0..4)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec)         shift; SPEC_KEYS+=("$1"); shift ;;
    --fold)         shift; FOLDS+=("$1"); shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --max-parallel) shift; MAX_PARALLEL="$1"; shift ;;
    --plans)        shift; PLANS="$1"; shift ;;
    --trainer)      shift; TRAINER="$1"; shift ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if (( ${#SPEC_KEYS[@]} == 0 )); then
  echo "must pass at least one --spec <key>" >&2; exit 2
fi
if (( ${#FOLDS[@]} == 0 )); then
  FOLDS=(0 1 2 3 4)
fi

# Resolve spec key → (dataset_id, dataset_name, spec_slug) by importing the registry.
declare -A DATASET_ID DATASET_NAME SPEC_SLUG
for key in "${SPEC_KEYS[@]}"; do
  read -r did dname dslug < <(
    PYTHONPATH="$ROOT/src" python3 -c "
from modelfactory.datasets.specs import SPECS
s = SPECS['$key']
slug = s.name.lower().replace('_', '-')
print(s.dataset_id, s.folder, slug)
"
  )
  DATASET_ID[$key]="$did"
  DATASET_NAME[$key]="$dname"
  SPEC_SLUG[$key]="$dslug"
done

mapfile -t SAFE_UUIDS < <(grep -v '^[[:space:]]*#' "$UUIDS_FILE" | grep -v '^[[:space:]]*$')
if (( ${#SAFE_UUIDS[@]} == 0 )); then
  echo "no safe UUIDs in $UUIDS_FILE" >&2; exit 2
fi
echo "loaded ${#SAFE_UUIDS[@]} safe UUIDs"

GIT_SHA="$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
mkdir -p "$RENDER_DIR"

render() {
  local key="$1" fold="$2" uuid="$3"
  local out="$RENDER_DIR/d${DATASET_ID[$key]}-f${fold}.yaml"
  python3 - "$TEMPLATE" "$out" \
    "${DATASET_ID[$key]}" "${SPEC_SLUG[$key]}" "${DATASET_NAME[$key]}" \
    "$fold" "$uuid" "$GIT_SHA" "$PLANS" "$TRAINER" <<'PY'
import sys, jinja2
tmpl, out, dataset_id, slug, name, fold, uuid, sha, plans, trainer = sys.argv[1:11]
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
rendered = env.from_string(open(tmpl).read()).render(
    dataset_id=int(dataset_id), spec_slug=slug, dataset_name=name,
    fold=int(fold), visible_uuid=uuid, git_sha=sha,
    plans=plans, trainer=trainer,
)
open(out, 'w').write(rendered)
PY
  echo "$out"
}

uuid_in_use() {
  local uuid="$1"
  kubectl -n model-factory get pods -l app=factory-train -o json 2>/dev/null \
    | jq -e --arg u "$uuid" \
        '.items[] | select(.status.phase=="Running" or .status.phase=="Pending")
                  | .spec.containers[0].env[] | select(.name=="NVIDIA_VISIBLE_DEVICES" and .value==$u)' \
    >/dev/null
}

WORK=()
for key in "${SPEC_KEYS[@]}"; do
  for f in "${FOLDS[@]}"; do WORK+=("$key:$f"); done
done

echo "work: ${#WORK[@]} fold-jobs"

submitted=0
for item in "${WORK[@]}"; do
  key="${item%:*}"; fold="${item#*:}"

  uuid=""
  while [[ -z "$uuid" ]]; do
    for cand in "${SAFE_UUIDS[@]}"; do
      if ! uuid_in_use "$cand"; then uuid="$cand"; break; fi
    done
    if [[ -z "$uuid" ]]; then
      echo "[$(date -u +%H:%M:%S)] all UUIDs busy, waiting 60s..."
      sleep 60
    fi
  done

  manifest="$(render "$key" "$fold" "$uuid")"
  echo "[$(date -u +%H:%M:%S)] -> ${DATASET_NAME[$key]} fold ${fold} on UUID ${uuid:0:18}..."
  if (( DRY_RUN )); then
    echo "  (dry-run) rendered to $manifest"
  else
    kubectl apply -f "$manifest"
  fi
  submitted=$((submitted+1))

  active=$(kubectl -n model-factory get jobs -l app=factory-train -o json \
    | jq '[.items[] | select(.status.active==1)] | length')
  while (( active >= MAX_PARALLEL )); do
    sleep 30
    active=$(kubectl -n model-factory get jobs -l app=factory-train -o json \
      | jq '[.items[] | select(.status.active==1)] | length')
  done
done

echo "submitted $submitted folds."
echo "status: bash scripts/agent_status.sh ${SPEC_KEYS[*]}"
