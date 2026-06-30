"""Helper that ensures `nnUNetv2_train -tr nnUNetTrainerMLflow` finds our class.

Two discovery mechanisms exist in nnUNetv2:
  1. Modules under `nnunetv2.training.nnUNetTrainer.variants.*` are imported on demand.
  2. Any module imported into Python before `nnUNetv2_train` runs that subclasses
     `nnUNetTrainer` will be discovered by its `recursive_find_python_class` walk
     of the loaded module tree (so long as the parent package is one of the
     standard search roots).

We rely on (1) by re-exporting via a small shim module. The pyproject entry-point
group `nnunetv2.trainers` is honoured if the upstream version supports it (≥2.5);
this file is the fallback that always works.
"""

from modelfactory.trainers.hpo_trainer import nnUNetTrainerHPO
from modelfactory.trainers.mlflow_trainer import nnUNetTrainerMLflow

__all__ = ["nnUNetTrainerHPO", "nnUNetTrainerMLflow"]
