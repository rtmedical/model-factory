"""Build the QA validation cohort at /factory/qa-cohort/.

Selection is deterministic (seeded RNG) and idempotent — re-running won't
re-copy cases that already exist. Cases are hard-copies, never symlinks
(CLAUDE.md gotcha #8).

Design: **one case per trained dataset**, donated from *that dataset's
own* imagesTr + labelsTr. This keeps GT label codes aligned with what
the model predicts — `dice_per_label` is meaningful, not a cross-dataset
label collision. The previous "shared-donor per region" design produced
near-zero dice whenever the donor and model dataset had different
structure indexes (e.g. D056 PosteriorFossa run against D045 CoreOAR).

Layout produced:
    qa-cohort/
        manifest.json
        brain_mr/d056_case_001/{image_0000.nii.gz, label_groundtruth.nii.gz, source.json}
        brain_mr/d045_case_001/...
        hn_ct/d087_case_001/...
        pelvis_ct/d023_case_001/...

Each case's `compatible_models` lists the trained models whose dataset_name
matches the case's source_dataset (typically a single model, since we
usually train one trainer__plans__cfg per dataset).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Region → which `Dataset###_*` folders belong to it. Used to tag each case
# with its region (so the web UI groups them) and to compute per-case
# `compatible_models` from the discovered trained models.
#
# `_region_for(dataset_name, datasets_root=...)` consults this table first,
# then `dataset.json:tags.region` (when datasets_root is supplied), then the
# name-based heuristic — so freshly-trained datasets we forgot to register
# still get a region (and therefore a donate-case button in the UI).
REGION_DATASETS: dict[str, list[str]] = {
    "brain_mr": [
        "Dataset045_Brain_MR_CoreOAR",
        "Dataset047_Brain_MR_Lobes_Bilateral",
        "Dataset048_Brain_MR_DeepNuclei",
        "Dataset056_Brain_MR_PosteriorFossa",
        "Dataset057_Brain_MR_VentricularSystem",
        "Dataset058_Brain_MR_BasalGanglia",
        "Dataset059_Brain_MR_Diencephalon",
        "Dataset060_Brain_MR_MedialTemporal",
        "Dataset061_Brain_MR_CorticalLobes_4pair",
        "Dataset062_Brain_MR_LimbicCortical",
        "Dataset063_Brain_MR_FullBrain_Generalist",
    ],
    "hn_ct": [
        "Dataset011_HN_PDDCA_Optic",
        "Dataset013_HN_PDDCA_Glands",
        "Dataset054_OpticNerves_HN_CT",
        "Dataset083_SegRap_OpticPathway_CT",
        "Dataset084_SegRap_Eyes_CT",
        "Dataset085_SegRap_InnerEar_CT",
        "Dataset086_SegRap_MidEar_HN_CT",
        "Dataset087_SegRap_Glands_HN_CT",
        "Dataset090_SegRap_Aerodigestive_CT",
        "Dataset091_SegRap_Mandible_TMJ_CT",
    ],
    "pelvis_ct": [
        "Dataset023_PelvisMaleProstate",
        "Dataset050_PenileBulb_Pelvis",
        "Dataset051_SeminalVes_Pelvis",
    ],
    "abdomen_ct": [
        "Dataset032_Pancreas_Tumor",
        "Dataset033_HepaticVessel_Tumor",
    ],
    "thorax_ct": [
        "Dataset036_LUNA16_Nodules",
    ],
}


# Name-based region inference for datasets not in REGION_DATASETS yet.
# Tuples are (substring, region); first match wins. Used as a fallback by
# `_region_for` so a brand-new dataset like `Dataset099_Brain_MR_Foo` still
# gets routed without a code change.
#
# Order matters: more-specific hints come first. `segrap` (a head-and-neck
# CT challenge) outranks the generic `brain` substring so e.g.
# `Dataset093_SegRap_BrainstemSpine_CT` resolves to hn_ct, not brain_mr.
_REGION_NAME_HINTS: tuple[tuple[str, str], ...] = (
    ("brain_mr", "brain_mr"),
    ("segrap", "hn_ct"),
    ("_hn_", "hn_ct"),
    ("hn_", "hn_ct"),
    ("brain", "brain_mr"),
    ("pelvis", "pelvis_ct"),
    ("prostate", "pelvis_ct"),
    ("pancreas", "abdomen_ct"),
    ("hepatic", "abdomen_ct"),
    ("liver", "abdomen_ct"),
    ("kidney", "abdomen_ct"),
    ("luna", "thorax_ct"),
    ("lung", "thorax_ct"),
    ("thorax", "thorax_ct"),
)


# `dataset.json:tags.region` → cohort-region key. This is the authoritative
# source for datasets like D100–D106 (melanoma specialists) whose names don't
# match any substring hint. Consulted by `_region_for` between the explicit
# REGION_DATASETS table and the name-hint fallback when a `datasets_root` is
# available.
_TAGS_REGION_TO_COHORT: dict[str, str] = {
    "brain": "brain_mr",
    "head_neck": "hn_ct",
    "brain_head_neck": "hn_ct",
    "pelvis": "pelvis_ct",
    "abdomen": "abdomen_ct",
    "thorax": "thorax_ct",
    "whole_body": "whole_body_ct",
}


# Every cohort region the builder knows about — the union of the explicit
# REGION_DATASETS table and the dataset.json tag mapping. `qa cohort prepare`
# fills `per_region` for ALL of these (not just the original three) so newer
# regions — abdomen_ct, thorax_ct, whole_body_ct — get cases built too.
KNOWN_REGIONS: tuple[str, ...] = tuple(
    dict.fromkeys([*REGION_DATASETS.keys(), *_TAGS_REGION_TO_COHORT.values()])
)


# Axial-resolution filter (added 2026-05-20, CT extended 2026-05-21). Cohort
# cases are biased toward truly thin-cut acquisitions so the QA viewer's
# reformatted coronal / sagittal planes don't look like a 25-slice staircase.
#
# Two threshold sets — MR is stricter (HCP-style brain MR is routinely
# isotropic 0.7 mm), CT is looser because clinical CT routinely lives at
# 2.5-3.75 mm axial. The iso shortcut is MR-only: every sampled CT case
# in the cohort is anisotropic.
_MR_MIN_THROUGH_SLICE_COUNT = 80          # ≥ 80 slices in the largest-spacing axis
_MR_MAX_THROUGH_SLICE_SPACING_MM = 2.5    # ≤ 2.5 mm in that axis
_MR_ISO_TOLERANCE = 0.10                  # zooms within 10 % of each other = isotropic

_CT_MIN_THROUGH_SLICE_COUNT = 60          # HN CT runs 107-144 slices, pelvis 156-290
_CT_MAX_THROUGH_SLICE_SPACING_MM = 4.0    # HN CT 3.0 mm, pelvis 2.5-3.75 mm


@dataclass
class CaseRecord:
    case_id: str  # cohort-local id, e.g. "case_001"
    region: str
    source_dataset: str
    source_case_stem: str
    image_paths: list[str]  # cohort-relative
    groundtruth_path: str | None
    compatible_models: list[str] = field(default_factory=list)
    # True for ad-hoc cases uploaded through the QA viewer (DICOM / NIfTI).
    # Their `source_dataset` is "uploaded" and `compatible_models` is set
    # explicitly to the model the reviewer uploaded against — so
    # `_merge_into_manifest` must NOT re-derive it from `trained_models`
    # (there is no trained model for the synthetic "uploaded" dataset).
    uploaded: bool = False


@dataclass
class CohortManifest:
    version: int
    regions: list[str]
    cases: list[CaseRecord]
    trained_models: list[dict]


def build_cohort(
    datasets_root: Path,
    results_root: Path,
    output_root: Path,
    per_region: dict[str, int],
    seed: int = 7,
) -> CohortManifest:
    """Materialize the cohort under `output_root` — N cases per trained dataset.

    `datasets_root` — e.g. /data/model-factory-nfs/datasets
    `results_root`  — e.g. /data/model-factory-nfs/results
    `output_root`   — e.g. /data/model-factory-nfs/qa-cohort
    `per_region`    — cases-per-trained-dataset within each region. E.g.
                      `{"brain_mr": 3, "hn_ct": 3, "pelvis_ct": 3}` ensures
                      three cases per trained dataset in each region. A value
                      of 0 skips that region.

    Delegates the per-dataset picking + materialization to
    `build_cohort_for_dataset`, which is **additive**: re-running with a
    larger count tops each dataset up to the new total without disturbing
    existing case ids (so reviewers can grow a model's test set safely).
    """
    output_root.mkdir(parents=True, exist_ok=True)
    trained_models = _discover_trained_models(results_root, datasets_root=datasets_root)

    # Bucket trained models by region so we can iterate region-then-dataset.
    by_region: dict[str, list[dict]] = {}
    for m in trained_models:
        region = m.get("region") or _region_for(
            m["dataset_name"], datasets_root=datasets_root,
        )
        if region is None:
            continue
        by_region.setdefault(region, []).append(m)

    for region, per_dataset in per_region.items():
        if per_dataset == 0:
            continue
        models_in_region = by_region.get(region, [])
        if not models_in_region:
            logger.warning("no trained models in region %s", region)
            continue

        # Deterministic per-dataset iteration so case ids don't shift when a
        # new dataset is trained alongside.
        seen_datasets: set[str] = set()
        for m in sorted(models_in_region, key=lambda x: x["dataset_name"]):
            ds_name = m["dataset_name"]
            if ds_name in seen_datasets:
                continue
            seen_datasets.add(ds_name)
            try:
                build_cohort_for_dataset(
                    ds_name,
                    datasets_root=datasets_root,
                    results_root=results_root,
                    output_root=output_root,
                    region=region,
                    n_pick=per_dataset,
                    seed=seed,
                    trained_models=trained_models,
                )
            except DatasetNotFoundError as exc:
                logger.warning("no imagesTr for %s — skipping cohort: %s",
                               ds_name, exc)
            except ValueError as exc:
                logger.warning("cannot build cohort for %s: %s", ds_name, exc)

    return _read_cohort_manifest(output_root, trained_models)


def _read_cohort_manifest(
    output_root: Path, trained_models: list[dict] | None = None,
) -> CohortManifest:
    """Load `manifest.json` into a CohortManifest (empty when absent)."""
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        return CohortManifest(
            version=2, regions=[], cases=[],
            trained_models=trained_models or [],
        )
    data = json.loads(manifest_path.read_text())
    cases = [_case_record_from_dict(c) for c in data.get("cases", [])]
    return CohortManifest(
        version=int(data.get("version", 2)),
        regions=list(data.get("regions", [])),
        cases=cases,
        trained_models=data.get("trained_models", trained_models or []),
    )


def _case_record_from_dict(c: dict) -> CaseRecord:
    """Reconstruct a CaseRecord from a manifest case dict (forward-compatible
    with manifests written before the `uploaded` field existed)."""
    return CaseRecord(
        case_id=c["case_id"],
        region=c["region"],
        source_dataset=c["source_dataset"],
        source_case_stem=c["source_case_stem"],
        image_paths=list(c.get("image_paths", [])),
        groundtruth_path=c.get("groundtruth_path"),
        compatible_models=list(c.get("compatible_models", [])),
        uploaded=bool(c.get("uploaded", False)),
    )


def _dataset_short(dataset_name: str) -> str:
    """``Dataset056_Brain_MR_PosteriorFossa`` → ``d056``."""
    m = re.match(r"Dataset(\d+)_", dataset_name)
    return f"d{m.group(1)}" if m else dataset_name.lower()


def _discover_trained_models(
    results_root: Path, *, datasets_root: Path | None = None,
) -> list[dict]:
    """Walk `results_root` for `<Dataset>/<config>/fold_N/checkpoint_{best,final}.pth`.

    Returns one entry per (dataset, config) — the union of folds with a
    `checkpoint_best.pth` OR `checkpoint_final.pth` becomes that model's
    `available_folds`. `checkpoint_final.pth` is what survives a production
    transfer to a downstream product (the operator deletes `checkpoint_best.pth` to save
    space); discovering on either keeps those shipped models in the catalog.

    `datasets_root` (optional) is forwarded to `_region_for` so the
    tags-tier lookup activates and D100–D106 get a non-None region.
    """
    found: dict[str, dict] = {}
    if not results_root.is_dir():
        return []
    # A fold is "trained" if it has either checkpoint. Glob both and dedupe
    # by (model_id, fold) via the membership check below — a fold with both
    # best and final is visited twice but added once.
    ckpts = sorted(
        list(results_root.glob("*/*/fold_*/checkpoint_best.pth"))
        + list(results_root.glob("*/*/fold_*/checkpoint_final.pth"))
    )
    for ckpt in ckpts:
        # backup folders look like fold_0.epoch50_backup — skip those
        fold_dir = ckpt.parent
        if "." in fold_dir.name.removeprefix("fold_"):
            continue
        m = re.match(r"fold_(\d+)$", fold_dir.name)
        if not m:
            continue
        fold = int(m.group(1))
        config_dir = fold_dir.parent
        dataset_dir = config_dir.parent

        # Parse trainer__plans__configuration
        parts = config_dir.name.split("__")
        if len(parts) != 3:
            continue
        trainer, plans, configuration = parts

        model_id = f"{dataset_dir.name}::{config_dir.name}"
        entry = found.setdefault(model_id, {
            "model_id": model_id,
            "dataset_name": dataset_dir.name,
            "configuration": configuration,
            "trainer": trainer,
            "plans": plans,
            "model_dir": str(config_dir),
            "available_folds": [],
            "region": _region_for(dataset_dir.name, datasets_root=datasets_root),
        })
        if fold not in entry["available_folds"]:
            entry["available_folds"].append(fold)

    for e in found.values():
        e["available_folds"].sort()
    return sorted(found.values(), key=lambda x: x["model_id"])


def _region_for(
    dataset_name: str, *, datasets_root: Path | None = None,
) -> str | None:
    """Map a dataset name to its region.

    Lookup priority:
      1. Explicit REGION_DATASETS membership.
      2. `dataset.json:tags.region` on disk, when `datasets_root` is given
         and the file exists. Translated via _TAGS_REGION_TO_COHORT.
      3. Substring match against _REGION_NAME_HINTS (case-insensitive).

    The tags-tier is what makes the donate-case button work for D100–D106
    (melanoma specialists) — their names don't match any substring hint
    but their `dataset.json` carries `tags.region=whole_body` /
    `abdomen`. Callers without a `datasets_root` skip tier 2 and keep
    the pre-existing behaviour.
    """
    for region, datasets in REGION_DATASETS.items():
        if dataset_name in datasets:
            return region
    if datasets_root is not None:
        tag = _read_tags_region(datasets_root / dataset_name)
        if tag:
            mapped = _TAGS_REGION_TO_COHORT.get(tag.lower())
            if mapped:
                return mapped
    lower = dataset_name.lower()
    for hint, region in _REGION_NAME_HINTS:
        if hint in lower:
            return region
    return None


def _read_tags_region(ds_dir: Path) -> str | None:
    """Read `tags.region` from `<ds_dir>/dataset.json`, or None if absent.

    Defensive read — matches the pattern of _is_mr_dataset / _is_ct_dataset.
    """
    js = ds_dir / "dataset.json"
    if not js.is_file():
        return None
    try:
        tag = json.loads(js.read_text()).get("tags", {}).get("region")
    except (json.JSONDecodeError, OSError):
        return None
    return str(tag) if tag else None


def _is_mr_dataset(ds_dir: Path) -> bool:
    """Return True iff `ds_dir/dataset.json` declares an MR modality.

    Reads `tags.modality` (e.g. "MR-T1") rather than `channel_names` —
    several brain MR datasets carry the wrong channel_names ("CT") from a
    known dataset-conversion bug, while `tags.modality` is reliable.
    """
    js = ds_dir / "dataset.json"
    if not js.is_file():
        return False
    try:
        modality = json.loads(js.read_text()).get("tags", {}).get("modality", "")
    except (json.JSONDecodeError, OSError):
        return False
    return str(modality).upper().startswith("MR")


def _is_ct_dataset(ds_dir: Path) -> bool:
    """Return True iff `ds_dir/dataset.json` describes a CT dataset.

    CT tagging is inconsistent across the cohort: SegRap datasets (D083-D091)
    + D050 carry `tags.modality: "CT"`, but older datasets (D011 PDDCA,
    D023 PelvisMaleProstate) have empty tags and rely on `channel_names.0`
    saying "CT". Try `tags.modality` first; fall back to channel-names
    inspection. Anything explicitly tagged MR returns False so we don't
    mis-classify the brain-MR datasets whose channel_names is bugged.
    """
    js = ds_dir / "dataset.json"
    if not js.is_file():
        return False
    try:
        d = json.loads(js.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    mod = str(d.get("tags", {}).get("modality", "")).upper()
    if mod.startswith("CT"):
        return True
    if mod.startswith("MR"):
        return False
    ch = d.get("channel_names", {}) or {}
    return any(str(v).upper().startswith("CT") for v in ch.values())


def _is_specialist_dataset(ds_dir: Path) -> bool:
    """Lesion / specialist datasets where GT foreground is sparse.

    Detected via `dataset.json:tags.scope == "specialist"` OR
    `tags.region == "whole_body"`. For these, the seeded RNG sample over
    axially-thin cases routinely lands on a case with little or no
    visible foreground — useless for an inference demo. Caller flips
    `rank_by_gt_volume=True` to add a GT-volume ranking step.

    Multi-organ OAR datasets (brain MR pack, body-CT pelvis/HN) keep
    today's axial-only filter because every case has every organ.
    """
    js = ds_dir / "dataset.json"
    if not js.is_file():
        return False
    try:
        tags = json.loads(js.read_text()).get("tags", {}) or {}
    except (json.JSONDecodeError, OSError):
        return False
    if str(tags.get("scope", "")).lower() == "specialist":
        return True
    if str(tags.get("region", "")).lower() == "whole_body":
        return True
    return False


def _gt_foreground_voxels(gt_path: Path) -> int:
    """Count of non-zero voxels in a GT label volume. 0 on read failure.

    Used to rank specialist-dataset candidates by lesion size so the
    auto-pick lands on a clear demonstration case. Label volumes are
    int-typed and sparse — full mmap read is cheap (~tens of MB).
    """
    if not gt_path.is_file():
        return 0
    try:
        import nibabel as nib  # lazy
        arr = nib.load(str(gt_path)).get_fdata(caching="unchanged")
        return int((arr != 0).sum())
    except Exception as e:  # noqa: BLE001 — defensive on disk + nib errors
        logger.warning("gt-volume read failed for %s: %s", gt_path, e)
        return 0


def _axial_quality(
    image_path: Path,
    *,
    min_slices: int,
    max_spacing_mm: float,
    treat_iso_as_pass: bool,
) -> tuple[bool, float]:
    """Score a case's through-slice resolution.

    Returns `(passes, score)` where:
      - `score` is the slice count along the through-slice axis (or the
        minimum dimension when the volume is isotropic) — used for ranking
        the fallback when no case passes outright.
      - `passes` is True when slices >= `min_slices` AND through-slice
        spacing <= `max_spacing_mm`. Isotropic MR volumes (HCP-style 0.7 mm)
        auto-pass when `treat_iso_as_pass=True`; CT is always anisotropic
        so the iso path is disabled there.

    The through-slice axis convention matches highres_planner.py:55 — it's
    the largest-zoom axis. Header-only read via nibabel mmap; no voxel data.
    """
    import nibabel as nib  # lazy

    img = nib.load(str(image_path), mmap=True)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    shape = [int(s) for s in img.shape[:3]]

    z_max, z_min = max(zooms), min(zooms)
    if treat_iso_as_pass and z_max > 0 and (z_max - z_min) / z_max < _MR_ISO_TOLERANCE:
        # Isotropic: every axis is equally fine; reformat preserves resolution.
        return True, float(min(shape))

    axial = max(range(3), key=lambda i: zooms[i])
    n_slices = float(shape[axial])
    spacing = zooms[axial]
    passes = n_slices >= min_slices and spacing <= max_spacing_mm
    return passes, n_slices


def _filter_axial_quality(
    case_groups: dict[str, list[Path]],
    candidate_stems: list[str],
    dataset_name: str,
    n_pick: int,
    *,
    min_slices: int,
    max_spacing_mm: float,
    treat_iso_as_pass: bool,
    rank_by_gt_volume: bool = False,
    labels_tr: Path | None = None,
) -> list[str]:
    """Restrict a dataset's case pool to thin-cut acquisitions.

    Falls back to the highest-slice-count case(s) when nothing passes the
    threshold outright, with a warning so the gap is discoverable. Returns a
    sorted list of stems that downstream sampling treats as the eligible pool.
    Thresholds are passed by the caller — MR + CT use different values.

    When `rank_by_gt_volume=True` (set by callers for specialist / lesion
    datasets), the passing pool is further narrowed to the top quartile by
    GT foreground voxel count. Stems with zero foreground are dropped
    outright. The clipped pool is large enough to keep the seeded RNG
    sample non-degenerate (>= max(n_pick * 4, 10) survivors when possible).
    """
    scored: list[tuple[str, bool, float]] = []
    for stem in candidate_stems:
        # Through-slice geometry is shared across nnUNet channels, so we
        # only score channel 0.
        ch0 = case_groups[stem][0]
        try:
            passes, score = _axial_quality(
                ch0,
                min_slices=min_slices,
                max_spacing_mm=max_spacing_mm,
                treat_iso_as_pass=treat_iso_as_pass,
            )
        except Exception as e:
            logger.warning("axial-score failed for %s/%s: %s", dataset_name, stem, e)
            continue
        scored.append((stem, passes, score))

    passing = [stem for stem, ok, _ in scored if ok]
    if not passing:
        scored.sort(key=lambda x: x[2], reverse=True)
        fallback = [stem for stem, _, _ in scored[: max(n_pick, 1)]]
        logger.warning(
            "no cases in %s pass axial filter (>= %d slices & <= %.1f mm); "
            "falling back to highest-slice-count: %s",
            dataset_name,
            min_slices,
            max_spacing_mm,
            fallback,
        )
        return sorted(fallback)

    if rank_by_gt_volume and labels_tr is not None and labels_tr.is_dir():
        passing = _rank_by_gt_volume(
            passing, labels_tr=labels_tr, dataset_name=dataset_name, n_pick=n_pick,
        )
    return sorted(passing)


def _rank_by_gt_volume(
    stems: list[str],
    *,
    labels_tr: Path,
    dataset_name: str,
    n_pick: int,
) -> list[str]:
    """Narrow a passing pool to the top quartile by GT foreground volume.

    Stems with zero foreground are dropped. The cap of
    `max(n_pick * 4, 10)` keeps the seeded RNG sample non-degenerate
    (so re-running with a different `n_pick` still has room to vary).
    Returns the full passing list unchanged if every stem reads as zero
    (don't silently empty the pool — fall through to axial-only ranking).
    """
    volumes: list[tuple[str, int]] = []
    for stem in stems:
        v = _gt_foreground_voxels(labels_tr / f"{stem}.nii.gz")
        if v > 0:
            volumes.append((stem, v))

    if not volumes:
        logger.warning(
            "gt-volume rank found zero foreground across %d cases in %s — "
            "keeping axial-pass list", len(stems), dataset_name,
        )
        return stems

    volumes.sort(key=lambda x: x[1], reverse=True)
    keep = max(n_pick * 4, 10)
    top = volumes[:keep]
    logger.info(
        "gt-volume rank for %s: kept top %d/%d (volume range %d–%d voxels)",
        dataset_name, len(top), len(volumes), top[-1][1], top[0][1],
    )
    return [stem for stem, _ in top]


# ── per-dataset donation (used by /api/cohort/cases on-demand) ────────────


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the requested dataset has no on-disk imagesTr."""


def _group_channels(images_tr: Path) -> dict[str, list[Path]]:
    """Group nnUNet imagesTr files by case stem → sorted channel paths.

    nnUNet convention: ``<case>_<channel:04d>.nii.gz``.
    """
    case_groups: dict[str, list[Path]] = {}
    for p in sorted(images_tr.glob("*_0000.nii.gz")):
        stem = re.sub(r"_0000\.nii\.gz$", "", p.name)
        case_groups[stem] = sorted(images_tr.glob(f"{stem}_*.nii.gz"))
    return case_groups


def _eligible_stems(
    ds_dir: Path,
    case_groups: dict[str, list[Path]],
    candidate_stems: list[str],
    dataset_name: str,
    n_pick: int,
    labels_tr: Path,
) -> list[str]:
    """Apply the MR/CT axial-quality filter to a candidate stem pool.

    MR uses stricter thresholds + an iso shortcut; CT uses looser ones and a
    GT-volume ranking step for specialist datasets. Datasets that are neither
    MR nor CT are returned unfiltered (sorted).
    """
    if _is_mr_dataset(ds_dir):
        return _filter_axial_quality(
            case_groups=case_groups,
            candidate_stems=candidate_stems,
            dataset_name=dataset_name,
            n_pick=n_pick,
            min_slices=_MR_MIN_THROUGH_SLICE_COUNT,
            max_spacing_mm=_MR_MAX_THROUGH_SLICE_SPACING_MM,
            treat_iso_as_pass=True,
        )
    if _is_ct_dataset(ds_dir):
        return _filter_axial_quality(
            case_groups=case_groups,
            candidate_stems=candidate_stems,
            dataset_name=dataset_name,
            n_pick=n_pick,
            min_slices=_CT_MIN_THROUGH_SLICE_COUNT,
            max_spacing_mm=_CT_MAX_THROUGH_SLICE_SPACING_MM,
            treat_iso_as_pass=False,
            rank_by_gt_volume=_is_specialist_dataset(ds_dir),
            labels_tr=labels_tr,
        )
    return sorted(candidate_stems)


def _case_index(case_id: str) -> int:
    """Trailing ``_case_NNN`` index of a cohort case_id (0 if unparseable)."""
    m = re.search(r"_case_(\d+)$", case_id)
    return int(m.group(1)) if m else 0


def existing_cohort_cases(output_root: Path, dataset_name: str) -> list[CaseRecord]:
    """CaseRecords already in `manifest.json` for `dataset_name` (sorted)."""
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    recs = [
        _case_record_from_dict(c)
        for c in data.get("cases", [])
        if c.get("source_dataset") == dataset_name
    ]
    return sorted(recs, key=lambda r: _case_index(r.case_id))


def build_cohort_for_dataset(
    dataset_name: str,
    *,
    datasets_root: Path,
    results_root: Path,
    output_root: Path,
    region: str | None = None,
    n_pick: int = 1,
    seed: int = 7,
    trained_models: list[dict] | None = None,
) -> list[CaseRecord]:
    """Ensure at least `n_pick` cohort cases exist for `dataset_name`.

    **Additive / top-up semantics.** Existing cases (matched by
    `source_case_stem`) are left untouched; only the shortfall
    ``n_pick - len(existing)`` is materialized, and new case dirs are
    numbered *after* the highest existing ``_case_NNN`` index. This is what
    makes "add more cases" safe: an existing case_id is never reassigned to a
    different source stem (the old re-sample-from-scratch approach could do
    that, leaving the on-disk images and the manifest stem out of sync).

    Re-running with the same or a smaller `n_pick` is a no-op (returns ``[]``).
    Picks use the same axial-resolution heuristic as `build_cohort`. The
    manifest is re-read / re-written on append — concurrent callers MUST be
    serialised by the caller (the API layer wraps it in an asyncio.Lock).

    Returns the **newly materialized** CaseRecords (empty when nothing was
    added: dataset already at target, or no usable cases on disk).
    """
    import random

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_region = region or _region_for(
        dataset_name, datasets_root=datasets_root,
    )
    if resolved_region is None:
        raise ValueError(
            f"cannot determine region for {dataset_name!r} — set "
            f"tags.region in dataset.json or extend REGION_DATASETS / "
            f"_REGION_NAME_HINTS / _TAGS_REGION_TO_COHORT"
        )

    ds_dir = datasets_root / dataset_name
    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    if not images_tr.is_dir():
        raise DatasetNotFoundError(
            f"no imagesTr at {images_tr} — cannot donate a case"
        )

    case_groups = _group_channels(images_tr)
    all_stems = sorted(case_groups.keys())
    if not all_stems:
        return []

    if trained_models is None:
        trained_models = _discover_trained_models(
            results_root, datasets_root=datasets_root,
        )
    compatible = sorted(
        m["model_id"] for m in trained_models
        if m["dataset_name"] == dataset_name
    )

    # Additive top-up: keep existing cases, only fill the shortfall, and
    # never reuse a stem already materialized for this dataset.
    existing = existing_cohort_cases(output_root, dataset_name)
    used_stems = {r.source_case_stem for r in existing}
    target = max(1, min(n_pick, len(all_stems)))
    n_new = target - len(existing)
    if n_new <= 0:
        return []

    candidates = [s for s in all_stems if s not in used_stems]
    if not candidates:
        return []
    candidates = _eligible_stems(
        ds_dir, case_groups, candidates, dataset_name, n_new, labels_tr,
    )
    n_new = min(n_new, len(candidates))
    sub_rng = random.Random(f"{seed}:{dataset_name}")
    picked = sub_rng.sample(candidates, n_new)

    region_dir = output_root / resolved_region
    region_dir.mkdir(parents=True, exist_ok=True)
    short = _dataset_short(dataset_name)
    next_index = max((_case_index(r.case_id) for r in existing), default=0) + 1

    new_records: list[CaseRecord] = []
    for offset, stem in enumerate(sorted(picked)):
        idx = next_index + offset
        case_id_short = f"{short}_case_{idx:03d}"
        case_id = f"{resolved_region}/{case_id_short}"
        case_dir = region_dir / case_id_short
        case_dir.mkdir(parents=True, exist_ok=True)

        image_rel: list[str] = []
        for channel_idx, src in enumerate(case_groups[stem]):
            dst_name = f"image_{channel_idx:04d}.nii.gz"
            dst = case_dir / dst_name
            if not dst.exists():
                shutil.copyfile(src, dst)
            image_rel.append(f"{resolved_region}/{case_id_short}/{dst_name}")

        gt_rel: str | None = None
        gt_src = labels_tr / f"{stem}.nii.gz"
        if gt_src.is_file():
            gt_dst = case_dir / "label_groundtruth.nii.gz"
            if not gt_dst.exists():
                shutil.copyfile(gt_src, gt_dst)
            gt_rel = f"{resolved_region}/{case_id_short}/label_groundtruth.nii.gz"

        src_json = case_dir / "source.json"
        if not src_json.is_file():
            src_json.write_text(json.dumps({
                "source_dataset": dataset_name,
                "source_case_stem": stem,
            }, indent=2))

        new_records.append(CaseRecord(
            case_id=case_id,
            region=resolved_region,
            source_dataset=dataset_name,
            source_case_stem=stem,
            image_paths=image_rel,
            groundtruth_path=gt_rel,
            compatible_models=compatible,
        ))

    _merge_into_manifest(output_root, new_records, trained_models)
    return new_records


def _merge_into_manifest(
    output_root: Path,
    new_records: list[CaseRecord],
    trained_models: list[dict],
) -> None:
    """Merge new CaseRecords into `manifest.json` (creating it if absent).

    De-duplicates on `case_id`. Also refreshes the `regions` list so that
    a brand-new region (e.g. `abdomen_ct`) is visible to the UI on first
    donation. Compatible-model lists are kept in sync by re-deriving from
    `trained_models` on every merge.
    """
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
    else:
        existing = {}

    cases_by_id: dict[str, dict] = {}
    for c in existing.get("cases", []):
        cases_by_id[c["case_id"]] = c
    for r in new_records:
        cases_by_id[r.case_id] = r.__dict__

    # Refresh compatible_models for every case from the latest trained_models.
    models_by_dataset: dict[str, list[str]] = {}
    for m in trained_models:
        models_by_dataset.setdefault(m["dataset_name"], []).append(m["model_id"])
    for c in cases_by_id.values():
        # Ad-hoc uploaded cases carry an explicit compatible_models list (the
        # model the reviewer uploaded against); there is no trained model for
        # the synthetic "uploaded" dataset, so re-deriving would wipe it.
        if c.get("uploaded") or c.get("source_dataset") == "uploaded":
            continue
        ds = c.get("source_dataset")
        if ds:
            c["compatible_models"] = sorted(models_by_dataset.get(ds, []))

    regions = sorted({c["region"] for c in cases_by_id.values() if c.get("region")})

    payload = {
        "version": existing.get("version", 2),
        "regions": regions,
        "cases": sorted(cases_by_id.values(), key=lambda c: c["case_id"]),
        "trained_models": trained_models,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(manifest_path)
