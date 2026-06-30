#!/usr/bin/env bash
# Download TotalSegmentator v2 dataset for Dataset001.
#
# Source: Zenodo 10047292
# License: CC BY 4.0 — commercial-OK with attribution
# 1228 CT volumes with 117 structure labels per case.
#
# Output layout (zip extracts directly into <dest>, no nested folder):
#   <dest>/
#       meta.csv
#       s0000/  ct.nii.gz  segmentations/<117 *.nii.gz>
#       s0001/  ...
#       ...
#
# Usage:
#   scripts/download_totalseg_v2.sh [dest_dir]

set -euo pipefail

DEST="${1:-/data/model-factory-nfs/intermediate/Dataset001_TotalSeg_v2}"
ZIP_URL="${TOTALSEG_V2_URL:-https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip}"

if [ -d "$DEST" ]; then
  count=$(find "$DEST" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
  if [ "$count" -gt 100 ]; then
    echo "[$(date -u +%H:%M:%S)] already extracted: $count case subdirs at $DEST"
    exit 0
  fi
fi

sudo mkdir -p "$DEST"
sudo chgrp -R nvidia "$DEST" 2>/dev/null || true
sudo chmod -R 0775 "$DEST" 2>/dev/null || true

ZIP="$DEST/Totalsegmentator_dataset_v201.zip"
echo "[$(date -u +%H:%M:%S)] downloading $ZIP_URL (~23 GB, can take 10-30 min) ..."
curl -fL --retry 8 --retry-delay 15 -o "$ZIP" "$ZIP_URL"

echo "[$(date -u +%H:%M:%S)] extracting to $DEST ..."
unzip -o -q "$ZIP" -d "$DEST"

case_count=$(find "$DEST" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
ct_count=$(find "$DEST" -maxdepth 2 -name 'ct.nii.gz' | wc -l)
seg_count=$(find "$DEST" -maxdepth 3 -path '*/segmentations/*.nii.gz' | wc -l)
echo "[$(date -u +%H:%M:%S)] extracted: $case_count cases, $ct_count CT files, $seg_count seg files"

if [ "$case_count" -lt 1000 ]; then
  echo "WARNING: expected ~1228 cases, got $case_count" >&2
fi

rm -f "$ZIP"
echo "[$(date -u +%H:%M:%S)] done; zip removed"
