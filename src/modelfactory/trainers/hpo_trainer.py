"""nnUNetTrainerHPO — hyperparameter-override-driven subclass of nnUNetTrainerMLflow.

Reads hyperparameter overrides from `MFACTORY_*` env vars and applies them to
the trainer instance so a Ray Tune sweep can drive nnUNetv2 without forking
upstream code. The MLflow logging from the parent class is inherited as-is;
applied overrides are surfaced as `hpo.<attr>` tags on the run so trial
hyperparameters show up in the MLflow UI next to the per-epoch metrics.

Env vars consumed (all optional — empty / unset leaves the default in place):

    MFACTORY_LR              float    nnUNetTrainer.initial_lr                  (default 1e-2)
    MFACTORY_WD              float    nnUNetTrainer.weight_decay                (default 3e-5)
    MFACTORY_OS_FG           float    nnUNetTrainer.oversample_foreground_percent  (default 0.33)
    MFACTORY_PROB_OS         bool     nnUNetTrainer.probabilistic_oversampling  (default False)
    MFACTORY_NUM_EPOCHS      int      nnUNetTrainer.num_epochs                  (default 1000)
    MFACTORY_NUM_ITERS       int      nnUNetTrainer.num_iterations_per_epoch    (default 250)
    MFACTORY_NUM_VAL_ITERS   int      nnUNetTrainer.num_val_iterations_per_epoch  (default 50)
    MFACTORY_DEEP_SUP        bool     nnUNetTrainer.enable_deep_supervision     (default True)

Override timing — important:
    `nnUNetTrainer.__init__` assigns `self.initial_lr = 1e-2` (and friends)
    unconditionally as part of its own attribute init. Setting these BEFORE
    `super().__init__()` therefore gets overwritten by the base class. We
    apply the overrides AFTER `super().__init__()` instead. This is safe
    because the optimizer (which reads `self.initial_lr`) is not built in
    `__init__` — it is built later in `on_train_start` via the base trainer's
    `configure_optimizers` path.
"""

from __future__ import annotations

import os
from typing import Any, Callable

try:
    import mlflow
except ImportError as e:
    raise RuntimeError("mlflow is required for nnUNetTrainerHPO") from e

from modelfactory.trainers.mlflow_trainer import nnUNetTrainerMLflow


def _env_bool(raw: str) -> bool:
    """Parse a bool from an env-var string. Empty / unrecognized → False."""
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_OVERRIDES: dict[str, tuple[str, Callable[[str], Any]]] = {
    "initial_lr":                    ("MFACTORY_LR", float),
    "weight_decay":                  ("MFACTORY_WD", float),
    "oversample_foreground_percent": ("MFACTORY_OS_FG", float),
    "probabilistic_oversampling":    ("MFACTORY_PROB_OS", _env_bool),
    "num_epochs":                    ("MFACTORY_NUM_EPOCHS", int),
    "num_iterations_per_epoch":      ("MFACTORY_NUM_ITERS", int),
    "num_val_iterations_per_epoch":  ("MFACTORY_NUM_VAL_ITERS", int),
    "enable_deep_supervision":       ("MFACTORY_DEEP_SUP", _env_bool),
}


class nnUNetTrainerHPO(nnUNetTrainerMLflow):
    """nnUNetTrainerMLflow + env-var-driven hyperparameter overrides."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hpo_overrides: dict[str, Any] = {}
        for attr, (envname, cast) in _OVERRIDES.items():
            raw = os.environ.get(envname)
            if raw is None or raw == "":
                continue
            try:
                value = cast(raw)
            except (ValueError, TypeError) as e:
                raise RuntimeError(
                    f"[HPO] cannot cast {envname}={raw!r} to {cast.__name__}: {e}"
                ) from e
            setattr(self, attr, value)
            self._hpo_overrides[attr] = value
        if self._hpo_overrides:
            try:
                self.print_to_log_file(
                    f"[HPO] applied overrides: {self._hpo_overrides}"
                )
            except Exception:
                pass

    def on_train_start(self):
        super().on_train_start()
        # Mirror the applied overrides into MLflow as tags so each trial in a
        # sweep is identifiable by its hyperparameter cell without expanding
        # the per-epoch metrics. set_tags is idempotent and safe to call once.
        if self._hpo_overrides:
            try:
                mlflow.set_tags(
                    {f"hpo.{k}": str(v) for k, v in self._hpo_overrides.items()}
                )
            except Exception as e:  # MLflow tracking down / DNS unresolved
                self.print_to_log_file(f"[HPO] mlflow.set_tags failed: {e}")
