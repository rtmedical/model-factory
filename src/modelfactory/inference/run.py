"""Single-case inference entrypoint used by the QA web API.

Operates on pre-staged `.npz` + `.pkl` produced by
`modelfactory.qa.preprocess.preprocess_cohort_for_model` (Layer A of the
two-layer cache). Skipping the resample/normalize step is the "preprocessed
once, infer fast" pattern.

If the pre-staged inputs are missing, we run the same sequence
*in-process* — preprocess → predict_logits → export — instead of
`predictor.predict_from_files`. The multiprocessing-worker pool that
`predict_from_files` spawns swallows per-worker failures (CUDA OOM,
NaN in CTNormalization, segfaults in the resampler) and returns
successfully without writing an output file, which surfaces to the QA
viewer as a silent `seg_ready` → 404. Running the sequence directly
means real exceptions bubble up to `_run_predict_background`, where they
land in `status.json.error_message` and the UI shows a real error pill.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Centralized record of the runtime flags every QA predictor is built
# with — see modelfactory.inference.predictor_cache.PredictorCache._build.
# The QA API reads this dict at seg_ready time to populate
# PredictionStatus.postprocessing, so the panel can show what was actually
# applied. If predictor_cache.py changes its defaults, update this dict
# in lockstep (and the docstring above).
PREDICTOR_FLAGS: dict[str, Any] = {
    "tile_step_size": 0.25,
    "use_gaussian": True,
    "use_mirroring": True,
    "perform_everything_on_device": True,
}


@dataclass
class InferenceResult:
    seg_path: Path
    elapsed_s: float
    used_preprocessed_cache: bool


def run_inference(
    predictor,
    raw_image_paths: list[Path],
    output_seg_path: Path,
    preprocessed_npz: Path | None = None,
    preprocessed_pkl: Path | None = None,
) -> InferenceResult:
    """Run one inference and write the labelmap to disk.

    Parameters
    ----------
    predictor
        A fully-initialized `nnUNetPredictor`.
    raw_image_paths
        One path per input channel. Used either directly (cache miss) or
        only as metadata (cache hit, when we already have `case.npz`).
    output_seg_path
        Destination `.nii.gz`. Parent dir is created if missing.
    preprocessed_npz / preprocessed_pkl
        If both exist on disk, we use the fast pre-staged path.
    """
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    used_cache = (
        preprocessed_npz is not None
        and preprocessed_pkl is not None
        and preprocessed_npz.is_file()
        and preprocessed_pkl.is_file()
    )

    if used_cache:
        _predict_from_preprocessed(
            predictor, preprocessed_npz, preprocessed_pkl, output_seg_path
        )
    else:
        logger.info("running in-process inference for %s",
                    raw_image_paths[0].name)
        _predict_in_process(predictor, raw_image_paths, output_seg_path)

    # Post-condition: the export step is what writes the file. If we got
    # here without it being written, something silently failed deeper in
    # the stack (most commonly a multiprocessing worker — though we no
    # longer use a Pool, third-party libraries inside nnUNet still can).
    # Surface that as a hard error so the API turns it into a real
    # `status="error"` instead of `seg_ready` followed by a 404.
    if not output_seg_path.is_file():
        raise RuntimeError(
            f"inference completed but {output_seg_path.name} was not "
            "written — check pod logs for stderr"
        )
    # NIfTI header alone is ~352 B; anything under 1 KB indicates a
    # truncated/empty file from a partial write.
    if output_seg_path.stat().st_size < 1024:
        raise RuntimeError(
            f"inference wrote {output_seg_path.name} "
            f"({output_seg_path.stat().st_size} B) — looks truncated; "
            "check pod logs for stderr"
        )

    elapsed = time.monotonic() - t0
    return InferenceResult(
        seg_path=output_seg_path,
        elapsed_s=elapsed,
        used_preprocessed_cache=used_cache,
    )


def _predict_from_preprocessed(
    predictor, npz_path: Path, pkl_path: Path, output_seg_path: Path
) -> None:
    """Hit the fast path: skip resample/normalize, run network + export."""
    import torch

    with np.load(npz_path) as bundle:
        data = bundle["data"].astype(np.float32, copy=False)
    with pkl_path.open("rb") as f:
        properties = pickle.load(f)

    data_t = torch.from_numpy(data)
    logits = predictor.predict_logits_from_preprocessed_data(data_t)

    # `convert_predicted_logits_to_segmentation_with_correct_shape` and
    # `export_prediction_from_logits` both live in
    # nnunetv2.inference.export_prediction; we use the export helper so
    # cropping/resampling back to the original spacing happens once.
    from nnunetv2.inference.export_prediction import export_prediction_from_logits

    output_stem = str(output_seg_path).removesuffix(".nii.gz")
    export_prediction_from_logits(
        predicted_array_or_file=logits,
        properties_dict=properties,
        configuration_manager=predictor.configuration_manager,
        plans_manager=predictor.plans_manager,
        dataset_json_dict_or_file=predictor.dataset_json,
        output_file_truncated=output_stem,
        save_probabilities=False,
    )


def _predict_in_process(
    predictor, raw_image_paths: list[Path], output_seg_path: Path
) -> None:
    """Preprocess → predict → export the case in this Python process.

    Mirrors the pre-staged fast path but runs `DefaultPreprocessor.run_case`
    on the fly. Same export helper, same predictor — the only difference
    is that resample + normalize happen here instead of in a separate
    `preprocess_cohort_for_model` pass.

    No `multiprocessing.Pool` is involved, so any failure (CUDA OOM,
    CTNormalization NaN, dtype mismatch, …) raises in this thread and
    propagates through `run_inference` to `_run_predict_background`.
    """
    import torch
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
        DefaultPreprocessor,
    )

    preprocessor = DefaultPreprocessor(verbose=False)
    data, _seg, properties = preprocessor.run_case(
        image_files=[str(p) for p in raw_image_paths],
        seg_file=None,
        plans_manager=predictor.plans_manager,
        configuration_manager=predictor.configuration_manager,
        dataset_json=predictor.dataset_json,
    )

    data_t = torch.from_numpy(np.asarray(data, dtype=np.float32))
    logits = predictor.predict_logits_from_preprocessed_data(data_t)

    output_stem = str(output_seg_path).removesuffix(".nii.gz")
    export_prediction_from_logits(
        predicted_array_or_file=logits,
        properties_dict=properties,
        configuration_manager=predictor.configuration_manager,
        plans_manager=predictor.plans_manager,
        dataset_json_dict_or_file=predictor.dataset_json,
        output_file_truncated=output_stem,
        save_probabilities=False,
    )
