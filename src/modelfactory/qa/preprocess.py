"""Pre-stage nnUNetv2-preprocessed `.npz`+`.pkl` for every (model, case) pair.

This is Layer A of the two-layer cache described in
`~/.claude/plans/your-task-is-create-clever-torvalds.md`. Once preprocessed,
the API's per-click cost on a warm model is reduced to forward-pass + export
(no resample, no normalize).

A `plan_hash.txt` sidecar records the fingerprint of (plans.json,
dataset_fingerprint.json) — if either changes upstream (e.g. retrained with
new plans), the cache invalidates and re-preprocesses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_cohort_for_model(
    model_dir: Path,
    cohort_root: Path,
    cohort_manifest_path: Path | None = None,
) -> list[Path]:
    """For each compatible case in the cohort, write case.npz + case.pkl.

    Returns the list of `case.npz` paths written (or already present).
    """
    if cohort_manifest_path is None:
        cohort_manifest_path = cohort_root / "manifest.json"
    manifest = json.loads(cohort_manifest_path.read_text())

    plan_hash = _plan_hash(model_dir)
    dataset_name = model_dir.parent.name
    config_name = model_dir.name
    model_id = f"{dataset_name}::{config_name}"

    # Compatible cases: the ones the cohort already tagged as compatible with
    # THIS model. The cohort computed `compatible_models` using the full
    # region resolver (REGION_DATASETS → dataset.json tags → name hints), so
    # filtering on it here fixes pre-staging silently no-opping for datasets
    # whose names aren't in REGION_DATASETS but resolve via tags/hints
    # (D110-117 HN-clinical, melanoma, pelvis-MR). The old `region ==` filter
    # returned [] for those, leaving every QA click on the slow cold path.
    compat_cases = [
        c for c in manifest["cases"]
        if model_id in (c.get("compatible_models") or [])
    ]
    if not compat_cases:
        logger.info("no compatible cohort cases for %s yet", model_id)
        return []

    out_root = cohort_root / "preprocessed" / dataset_name / config_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Deferred — these pull in torch/SimpleITK.
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
        DefaultPreprocessor,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets  # noqa: F401

    plans_path = _meta_path(model_dir, "plans.json")
    ds_json_path = _meta_path(model_dir, "dataset.json")
    plans = json.loads(plans_path.read_text())
    dataset_json = json.loads(ds_json_path.read_text())
    plans_manager = PlansManager(plans)
    # `config_name` looks like "nnUNetTrainerMLflow__nnUNetResEncUNetXLPlans__3d_fullres"
    # The last segment is the configuration name (3d_fullres).
    configuration = config_name.split("__")[-1]
    configuration_manager = plans_manager.get_configuration(configuration)

    preprocessor = DefaultPreprocessor(verbose=False)

    written: list[Path] = []
    for case in compat_cases:
        case_out = out_root / case["case_id"]
        case_out.mkdir(parents=True, exist_ok=True)
        npz = case_out / "case.npz"
        pkl = case_out / "case.pkl"
        hash_file = case_out / "plan_hash.txt"

        if (
            npz.is_file()
            and pkl.is_file()
            and hash_file.is_file()
            and hash_file.read_text().strip() == plan_hash
        ):
            written.append(npz)
            continue

        image_paths = [
            str((cohort_root / p).resolve()) for p in case["image_paths"]
        ]
        seg_path = (
            str((cohort_root / case["groundtruth_path"]).resolve())
            if case.get("groundtruth_path") else None
        )

        data, _seg, properties = preprocessor.run_case(
            image_files=image_paths,
            seg_file=seg_path,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            dataset_json=dataset_json,
        )

        import numpy as np
        np.savez_compressed(npz, data=data.astype(np.float32, copy=False))
        with pkl.open("wb") as f:
            pickle.dump(properties, f)
        hash_file.write_text(plan_hash)
        written.append(npz)
        logger.info("preprocessed %s -> %s", case["case_id"], npz)

    return written


def _plan_hash(model_dir: Path) -> str:
    plans = _meta_path(model_dir, "plans.json").read_bytes()
    fp_path = _meta_path(model_dir, "dataset_fingerprint.json")
    fp = fp_path.read_bytes() if fp_path.is_file() else b""
    return hashlib.sha256(plans + b"::" + fp).hexdigest()[:16]


def _meta_path(model_dir: Path, name: str) -> Path:
    """nnUNet writes plans/dataset/fingerprint json inside the config dir,
    not at the dataset dir. Prefer the config dir, fall back to legacy."""
    inside = model_dir / name
    if inside.is_file():
        return inside
    return model_dir.parent / name
