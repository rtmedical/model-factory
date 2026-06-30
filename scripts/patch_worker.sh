#!/bin/bash
# Apply the recreated-worker fixup (see memory: recreated_worker_image_drift).
# Usage: ./patch_worker.sh <worker-pod-name>
set -euo pipefail
worker="$1"
echo "[$(date -Iseconds)] patching $worker"
kubectl -n model-factory cp /data/model-factory/src/modelfactory/trainers/mlflow_trainer.py "$worker:/opt/modelfactory/src/modelfactory/trainers/mlflow_trainer.py"
# Also sync ray_driver.py so the worker's _stage_preprocessed has the ENOSPC->NFS
# fallback (cloudpickled _run_one_fold resolves _stage_preprocessed by reference
# from the worker's module). Needed for preprocessed dirs > the 110Gi /factory-ram
# tmpfs (e.g. D153 HighRes at 185G).
kubectl -n model-factory cp /data/model-factory/src/modelfactory/jobs/ray_driver.py "$worker:/opt/modelfactory/src/modelfactory/jobs/ray_driver.py"
kubectl -n model-factory exec "$worker" -c ray-worker -- bash -c '
F=/usr/local/lib/python3.12/dist-packages/nnunetv2/training/nnUNetTrainer/variants/factory
echo "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerSmallStructuresMLflow" > $F/small_structures_trainer.py
echo "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerPartialLabelMLflow" > $F/partial_label_trainer.py
echo "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerPartialLabelBalancedMLflow" > $F/partial_label_balanced_trainer.py
find /opt/modelfactory/src/modelfactory/{trainers,jobs}/__pycache__ -name "mlflow_trainer*" -o -name "ray_driver*" 2>/dev/null | xargs -r rm -f
python3 -c "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerSmallStructuresMLflow, nnUNetTrainerPartialLabelMLflow, nnUNetTrainerPartialLabelBalancedMLflow; from modelfactory.jobs.ray_driver import _is_enospc" && echo OK
'
