#!/usr/bin/env bash
# Download MSD Task04 Hippocampus dataset for Dataset046.
#
# Source: Medical Segmentation Decathlon — http://medicaldecathlon.com
# License: CC BY-SA 4.0 (commercial use OK with attribution)
#
# Output layout (matches MSDDecathlonSource expectations):
#   <dest>/Task04_Hippocampus/
#       imagesTr/<case>.nii.gz
#       labelsTr/<case>.nii.gz
#       dataset.json
#
# Usage:
#   scripts/download_msd_task04.sh [dest_dir]   # default /data/model-factory-nfs/intermediate/Dataset046_MSD_Task04

set -euo pipefail

DEST="${1:-/data/model-factory-nfs/intermediate/Dataset046_MSD_Task04}"
TAR_URL="${MSD_TASK04_URL:-https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar}"

if [ -d "$DEST/Task04_Hippocampus/imagesTr" ]; then
  count=$(find "$DEST/Task04_Hippocampus/imagesTr" -name '*.nii.gz' 2>/dev/null | wc -l)
  if [ "$count" -gt 0 ]; then
    echo "[$(date -u +%H:%M:%S)] already extracted: $count training images at $DEST/Task04_Hippocampus/imagesTr"
    echo "(set DEST to a different path or rm the existing tree to re-download)"
    exit 0
  fi
fi

sudo mkdir -p "$DEST"
sudo chgrp -R nvidia "$DEST" 2>/dev/null || true
sudo chmod -R 0775 "$DEST" 2>/dev/null || true

TAR="$DEST/Task04_Hippocampus.tar"
echo "[$(date -u +%H:%M:%S)] downloading $TAR_URL ..."
curl -fL -o "$TAR" "$TAR_URL"

echo "[$(date -u +%H:%M:%S)] extracting to $DEST ..."
tar -xf "$TAR" -C "$DEST"

# Sanity check
img_count=$(find "$DEST/Task04_Hippocampus/imagesTr" -name '*.nii.gz' 2>/dev/null | wc -l)
lbl_count=$(find "$DEST/Task04_Hippocampus/labelsTr" -name '*.nii.gz' 2>/dev/null | wc -l)
echo "[$(date -u +%H:%M:%S)] extracted: $img_count images, $lbl_count labels"

if [ "$img_count" -lt 200 ] || [ "$lbl_count" -lt 200 ]; then
  echo "WARNING: expected ~260 training cases, got $img_count/$lbl_count" >&2
fi

echo "[$(date -u +%H:%M:%S)] dataset.json:"
cat "$DEST/Task04_Hippocampus/dataset.json" 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(f'  name:        {d.get(\"name\")}')
    print(f'  description: {d.get(\"description\",\"\")[:80]}')
    print(f'  labels:      {d.get(\"labels\")}')
    print(f'  modality:    {d.get(\"modality\")}')
    print(f'  numTraining: {d.get(\"numTraining\")}')
except Exception as e:
    print('  (could not parse dataset.json:', e, ')')
"

# Remove the tarball to save space
rm -f "$TAR"
echo "[$(date -u +%H:%M:%S)] done; tarball removed"
