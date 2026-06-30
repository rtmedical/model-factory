#!/usr/bin/env bash
# One-time cutover: move proprietary / site-specific content out of the public
# tree into a git-ignored private overlay (overlays/private/), so the public
# repo passes the CI "forbidden-strings" gate while your working deployment
# keeps running off the overlay.
#
#   ./scripts/extract_overlay.sh            # dry-run: show what WOULD move
#   ./scripts/extract_overlay.sh --apply    # actually move
#
# SAFETY: run this in a maintenance window. The host CLI resolves dataset specs
# from these files; after the move, the overlay-discovery hook
# (modelfactory.datasets.specs._load_overlay_specs) re-registers them. VERIFY
# with `modelfactory dataset list` BEFORE relying on it for new submissions.
set -euo pipefail
cd "$(dirname "$0")/.."

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

OVERLAY=overlays/private
mkdir_p() { [[ $APPLY -eq 1 ]] && mkdir -p "$1" || true; }
move() {  # move <src> <dest-dir>
  local src="$1" dst="$2"
  [[ -e "$src" ]] || { echo "  skip (absent): $src"; return; }
  echo "  $src  ->  $dst/"
  if [[ $APPLY -eq 1 ]]; then mkdir -p "$dst"; git mv "$src" "$dst/" 2>/dev/null || mv "$src" "$dst/"; fi
}

echo "== model-factory overlay extraction (apply=$APPLY) =="
echo
echo "[1] proprietary dataset catalogue (128 internal specs) + clinical aliases"
move src/modelfactory/datasets/specs.py    "$OVERLAY/specs"
move src/modelfactory/datasets/aliases.py  "$OVERLAY/datasets"

echo "[2] cohort-specific source adapters (embed internal layouts / aliases)"
for a in clinical_rtstruct vendor_rtstruct index_paired_rtstruct rtstruct_ts_fused mixed_optic; do
  move "src/modelfactory/datasets/sources/$a.py" "$OVERLAY/sources"
done

echo "[3] internal region docs + runbooks"
for d in brain_oar_models hn_oar_models abdomen_ct_models pelvis_cancer_archive_ref \
         pelvis_thorax_training_ref training_schedule open_datasets; do
  move "docs/$d.md" "$OVERLAY/docs"
done
move docs/runbooks/next_wave_2026_06_08.md "$OVERLAY/docs/runbooks"

echo "[4] internal helper scripts + site-specific infra patches"
for s in download_segrap2023.sh audit_rtstruct_volumes.py inspect_clinical_cohort.py \
         curate_fomo_subjects.py run_ts_optic_labeller.py run_ts_pelvis_labeller.py \
         backfill_parent_tags.py; do
  move "scripts/$s" "$OVERLAY/scripts"
done
move infra/patches "$OVERLAY/infra-patches"

echo
cat <<'EOF'
NEXT STEPS (manual, by design — these need human judgement):

  a) Replace the public src/modelfactory/datasets/specs.py with a SLIM module
     that keeps ONLY the framework (DatasetSpec/StructureMapping/_register/SPECS,
     the _load_overlay_specs hook) plus a few PUBLIC example specs (MSD, PDDCA,
     LUNA16, TotalSegmentator). The moved file becomes
     overlays/private/specs/_proprietary_specs.py and re-registers via the hook.
  b) Provide a slim public datasets/aliases.py (generic TG-263 examples); keep
     the internal alias tables in the overlay.
  c) Point any cohort adapters imported by overlay specs at overlays/private/sources.
  d) Run: modelfactory dataset list   # confirm every spec still resolves
     and  PYTHONPATH=src pytest tests/ -q
  e) Run the CI forbidden-strings gate locally (see .github/workflows/ci.yml).

cluster.yaml and secrets.yaml are already git-ignored; keep them out of git.
EOF
[[ $APPLY -eq 0 ]] && echo && echo "(dry-run — re-run with --apply to move files)"
