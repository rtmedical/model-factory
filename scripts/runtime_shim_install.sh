#!/bin/bash
# Runtime-install the nnUNetv2 shims so the SmallStructures trainer + HighRes
# planner are discoverable by `recursive_find_python_class` inside an old
# trainer image that pre-dates Dockerfile lines 135-139.
#
# Prereqs (caller's responsibility):
#   - mount /data/model-factory at /code (read-only is fine)
#   - PYTHONPATH includes /code/src so `modelfactory.*` is importable
#
# After this runs, `from modelfactory.planners.highres_planner import …` works
# AND nnUNet's class finder picks up the shim files. Then exec the real
# command.

set -euo pipefail

SP=$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')

# Planner shim (mirrors Dockerfile lines 136-139)
mkdir -p "$SP/experiment_planning/factory"
: > "$SP/experiment_planning/factory/__init__.py"
echo "from modelfactory.planners.highres_planner import nnUNetPlannerResEncL_HighRes" \
    > "$SP/experiment_planning/factory/highres_planner.py"

# Trainer shim (mirrors Dockerfile lines 130-135)
TRAINER_FACTORY="$SP/training/nnUNetTrainer/variants/factory"
mkdir -p "$TRAINER_FACTORY"
: > "$TRAINER_FACTORY/__init__.py"
echo "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerMLflow" \
    > "$TRAINER_FACTORY/mlflow_trainer.py"
echo "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerSmallStructuresMLflow" \
    > "$TRAINER_FACTORY/small_structures_trainer.py"
echo "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerPartialLabelMLflow" \
    > "$TRAINER_FACTORY/partial_label_trainer.py"
echo "from modelfactory.trainers.mlflow_trainer import nnUNetTrainerPartialLabelBalancedMLflow" \
    > "$TRAINER_FACTORY/partial_label_balanced_trainer.py"

echo "shim install OK: $SP/experiment_planning/factory + $TRAINER_FACTORY"

exec "$@"
