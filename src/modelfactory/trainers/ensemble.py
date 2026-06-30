"""5-fold ensemble registration as a single MLflow pyfunc model.

Usage (from the CLI):
    modelfactory model register-ensemble --dataset DatasetXXX --configuration 3d_fullres

This pulls all 5 fold MLflow runs (matched by tags dataset+configuration+fold),
collects their checkpoint_final.pth artifacts, and wraps them in a pyfunc that
loads all 5 networks, runs sliding-window inference, averages softmax, applies
nnUNet's standard post-processing, and returns a label volume.

The registered model becomes one entry in the MLflow Model Registry, one
version per ensemble. Promotion to Staging/Production gates downstream use.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel


class EnsemblePredictor(PythonModel):
    """MLflow pyfunc that wraps 5 nnUNetv2 fold checkpoints into a single predictor."""

    def load_context(self, context):
        # Deferred import — keeps the registration path lightweight on the orchestrator.
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        # Each fold's checkpoint is at context.artifacts[f"fold_{i}"]
        fold_paths = [context.artifacts[f"fold_{i}"] for i in range(5)]
        # All 5 folds share plans + dataset; the predictor loads them in a single call.
        predictor.initialize_from_trained_model_folder(
            str(Path(fold_paths[0]).parent.parent),  # the …/nnUNetTrainerMLflow__plans__cfg dir
            use_folds=tuple(range(5)),
            checkpoint_name="checkpoint_final.pth",
        )
        self._predictor = predictor

    def predict(self, context, model_input):
        """model_input is a list/array of file paths to NIfTI input volumes."""
        outputs = []
        for in_path in np.atleast_1d(model_input):
            out = self._predictor.predict_from_files(
                list_of_lists_or_source_folder=[[str(in_path)]],
                output_folder_or_list_of_truncated_output_files=None,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
            )
            outputs.append(out[0])
        return outputs


def register_ensemble(
    dataset: str,
    configuration: str,
    parent_run_id: str | None = None,
    registered_model_name: str | None = None,
) -> str:
    """Find 5 fold runs, attach their checkpoints, log as a registered pyfunc model.

    Returns the URI of the registered model version.
    """
    client = mlflow.tracking.MlflowClient()

    if parent_run_id:
        runs = client.search_runs(
            experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
            filter_string=(
                f"tags.dataset = '{dataset}' "
                f"and tags.configuration = '{configuration}' "
                f"and tags.parent_run_id = '{parent_run_id}'"
            ),
        )
    else:
        experiment = mlflow.get_experiment_by_name(f"{dataset}__{configuration}")
        if experiment is None:
            raise ValueError(f"No experiment {dataset}__{configuration} found")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.dataset = '{dataset}' and tags.configuration = '{configuration}'",
        )

    fold_runs = {int(r.data.tags["fold"]): r for r in runs if "fold" in r.data.tags}
    if set(fold_runs.keys()) != set(range(5)):
        raise ValueError(
            f"Need 5 completed folds; got: {sorted(fold_runs.keys())}"
        )

    artifacts = {}
    for fold_idx, run in fold_runs.items():
        ckpt = client.download_artifacts(run.info.run_id, f"fold_{fold_idx}/checkpoint_final.pth")
        artifacts[f"fold_{fold_idx}"] = ckpt
    artifacts["dataset_json"] = client.download_artifacts(
        list(fold_runs.values())[0].info.run_id, "dataset_meta/dataset.json"
    )

    name = registered_model_name or f"{dataset}__{configuration}"
    with mlflow.start_run(run_name=f"ensemble__{dataset}__{configuration}"):
        model_info = mlflow.pyfunc.log_model(
            artifact_path="ensemble",
            python_model=EnsemblePredictor(),
            artifacts=artifacts,
            registered_model_name=name,
            pip_requirements=[
                "nnunetv2==2.5.*",
                "torch",
                "SimpleITK",
                "nibabel",
                "numpy",
            ],
        )
    return model_info.model_uri
