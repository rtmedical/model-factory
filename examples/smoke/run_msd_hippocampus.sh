#!/usr/bin/env bash
# End-to-end smoke test for the factory.  Downloads MSD Task04 Hippocampus (small,
# ~30 min per fold on H100), registers it, runs 5 folds via Kueue, registers the
# ensemble in MLflow.
#
# Idempotent.  Re-runs skip already-completed steps.
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/data/model-factory-nfs}"
DATASET_NAME="Dataset100_Hippocampus"
DOWNLOAD_DIR="${NFS_ROOT}/incoming/Task04_Hippocampus"
DATA_URL="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"

echo "[1/4] Download MSD Task04 Hippocampus (if not already present)"
if [[ ! -d "${DOWNLOAD_DIR}" ]]; then
  mkdir -p "${DOWNLOAD_DIR}"
  curl -L "${DATA_URL}" | tar -x -C "${NFS_ROOT}/incoming"
fi

echo "[2/4] Convert MSD format → nnUNetv2 Dataset format (if not already done)"
if [[ ! -d "${NFS_ROOT}/datasets/${DATASET_NAME}" ]]; then
  # nnUNetv2 ships a converter for MSD datasets; run it inside the trainer image.
  kubectl run msd-converter -n model-factory --rm -i --restart=Never \
    --image=registry.model-factory.svc:5000/nnunet-trainer:0.1.0 \
    --overrides='{"spec":{"containers":[{"name":"msd-converter","image":"registry.model-factory.svc:5000/nnunet-trainer:0.1.0","command":["nnUNetv2_convert_MSD_dataset"],"args":["-i","/factory/incoming/Task04_Hippocampus","-overwrite_id","100"],"env":[{"name":"nnUNet_raw","value":"/factory/datasets"},{"name":"nnUNet_preprocessed","value":"/factory/preprocessed"},{"name":"nnUNet_results","value":"/factory/results"}],"volumeMounts":[{"name":"factory-data","mountPath":"/factory"}]}],"volumes":[{"name":"factory-data","persistentVolumeClaim":{"claimName":"factory-data-pvc"}}],"restartPolicy":"Never"}}'
fi

echo "[3/4] Preprocess + plan (CPU-only)"
kubectl run msd-preprocess -n model-factory --rm -i --restart=Never \
  --image=registry.model-factory.svc:5000/nnunet-trainer:0.1.0 \
  --overrides='{"spec":{"containers":[{"name":"msd-preprocess","image":"registry.model-factory.svc:5000/nnunet-trainer:0.1.0","command":["nnUNetv2_plan_and_preprocess"],"args":["-d","100","-c","3d_fullres","--verify_dataset_integrity"],"env":[{"name":"nnUNet_raw","value":"/factory/datasets"},{"name":"nnUNet_preprocessed","value":"/factory/preprocessed"},{"name":"nnUNet_results","value":"/factory/results"}],"resources":{"requests":{"cpu":"16","memory":"32Gi"},"limits":{"cpu":"64","memory":"128Gi"}},"volumeMounts":[{"name":"factory-data","mountPath":"/factory"}]}],"volumes":[{"name":"factory-data","persistentVolumeClaim":{"claimName":"factory-data-pvc"}}],"restartPolicy":"Never"}}'

echo "[4/4] Submit 5-fold training"
modelfactory train nnunet --dataset "${DATASET_NAME}" --configuration 3d_fullres --folds all

echo "Done.  Watch progress with:  modelfactory runs list --dataset ${DATASET_NAME}"
