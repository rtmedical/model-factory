#!/usr/bin/env bash
# Download LUNA16 dataset for Dataset036.
#
# Source: Zenodo records 3723295 (Part 1/2) + 4121926 (Part 2/2)
#   Originally hosted on grand-challenge.org/LUNA16 (now redirects to Zenodo)
# License: CC BY 3.0 (LIDC inheritance) — commercial-OK with attribution
# Subset of LIDC-IDRI with nodule localizations + lung segmentation masks.
#
# Output layout:
#   <dest>/
#       subset0/...subset9/ (10 directories, ~70 GB total, MHD+RAW pairs)
#       annotations.csv
#       candidates.csv  candidates_V2.csv
#       sampleSubmission.csv
#       seg-lungs-LUNA16/    (per-case lung masks)
#       evaluationScript/    (eval code, optional)
#
# Usage:
#   scripts/download_luna16.sh [dest_dir]

set -euo pipefail

DEST="${1:-/data/model-factory-nfs/intermediate/Dataset036_LUNA16}"

PART1="https://zenodo.org/api/records/3723295/files-archive"
# (Direct per-file URLs below; we fetch each individually so a partial run can resume.)

ZENODO_PART1_BASE="https://zenodo.org/records/3723295/files"
ZENODO_PART2_BASE="https://zenodo.org/records/4121926/files"

# part 1 files
P1_FILES=(
  "candidates.csv"
  "annotations.csv"
  "sampleSubmission.csv"
  "candidates_V2.zip"
  "evaluationScript.zip"
  "seg-lungs-LUNA16.zip"
  "subset0.zip"
  "subset1.zip"
  "subset2.zip"
  "subset3.zip"
  "subset4.zip"
  "subset5.zip"
  "subset6.zip"
)
P2_FILES=(
  "subset7.zip"
  "subset8.zip"
  "subset9.zip"
)

if [ -d "$DEST/subset0" ] && [ "$(find "$DEST/subset0" -name '*.mhd' | wc -l)" -gt 50 ]; then
  echo "[$(date -u +%H:%M:%S)] already extracted: $(find "$DEST" -name '*.mhd' | wc -l) MHD files"
  exit 0
fi

sudo mkdir -p "$DEST"
sudo chgrp -R nvidia "$DEST" 2>/dev/null || true
sudo chmod -R 0775 "$DEST" 2>/dev/null || true

download_one() {
  local base="$1" name="$2"
  local out="$DEST/$name"
  if [ -f "$out" ]; then
    echo "[$(date -u +%H:%M:%S)] skip $name (already downloaded)"
    return
  fi
  echo "[$(date -u +%H:%M:%S)] fetching $name ..."
  curl -fL --retry 8 --retry-delay 15 -o "$out.part" "$base/$name"
  mv "$out.part" "$out"
}

for f in "${P1_FILES[@]}"; do download_one "$ZENODO_PART1_BASE" "$f"; done
for f in "${P2_FILES[@]}"; do download_one "$ZENODO_PART2_BASE" "$f"; done

echo "[$(date -u +%H:%M:%S)] extracting subset zips ..."
for f in "$DEST"/subset*.zip; do
  unzip -o -q "$f" -d "$DEST"
  rm -f "$f"
done

echo "[$(date -u +%H:%M:%S)] extracting lung-mask zip ..."
unzip -o -q "$DEST/seg-lungs-LUNA16.zip" -d "$DEST"
rm -f "$DEST/seg-lungs-LUNA16.zip"

echo "[$(date -u +%H:%M:%S)] extracting candidates_V2 + evaluationScript ..."
unzip -o -q "$DEST/candidates_V2.zip" -d "$DEST"
rm -f "$DEST/candidates_V2.zip"
unzip -o -q "$DEST/evaluationScript.zip" -d "$DEST"
rm -f "$DEST/evaluationScript.zip"

mhd_count=$(find "$DEST" -name '*.mhd' | grep -v seg-lungs | wc -l)
echo "[$(date -u +%H:%M:%S)] extracted: $mhd_count CT MHD/RAW pairs (expected ~888)"

if [ "$mhd_count" -lt 800 ]; then
  echo "WARNING: expected ~888 MHD pairs, got $mhd_count" >&2
fi

echo "[$(date -u +%H:%M:%S)] done"
