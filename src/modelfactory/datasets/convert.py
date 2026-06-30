"""Generic nnUNetv2 raw-dataset writer.

Workflow:
    1. Look up `DatasetSpec` from the registry (specs.SPECS).
    2. Instantiate the requested `DatasetSource` with the spec's
       per-source constraints.
    3. Source.discover() yields CaseRefs that have all required structures.
    4. For each case (in parallel): load_image, load each structure mask,
       combine into a multi-label uint8 volume, write imagesTr/labelsTr.
    5. Emit dataset.json (with the spec's label_map) and
       preprocessed/<dataset>/splits_final.json (patient-stratified 5-fold).

CLI:
    python -m modelfactory.datasets.convert \\
        --spec pelvis_male_prostate \\
        --source rtstruct \\
        --out /factory \\
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .specs import SPECS, DatasetSpec
from .sources.base import CaseRef, DatasetSource


log = logging.getLogger("convert")


# ── source registry ──────────────────────────────────────────────────────

def _build_source(source_type: str, spec: DatasetSpec) -> DatasetSource:
    cfg = spec.source_constraints.get(source_type)
    if cfg is None:
        raise ValueError(
            f"spec {spec.name!r} has no source_constraints for {source_type!r}"
        )
    if source_type == "rtstruct":
        from .sources.rtstruct import RTStructDicomSource
        # rtstruct_root/tcia_root/series_csv may each be a single path OR a
        # parallel list (multi-tree union, e.g. abdome/f_ct + abdome/m_ct →
        # Dataset161). RTStructDicomSource normalises str|list → list[Path].
        return RTStructDicomSource(
            rtstruct_root=cfg["rtstruct_root"],
            tcia_root=cfg["tcia_root"],
            series_csv=cfg["series_csv"],
            volume_ranges=cfg.get("volume_ranges"),
            allow_manifest=Path(cfg["allow_manifest"]) if cfg.get("allow_manifest") else None,
            partial_label=spec.partial_label,
        )
    if source_type == "pddca":
        from .sources.pddca import PDDCASource
        return PDDCASource(
            src_root=Path(cfg["src_root"]),
            version=cfg.get("version", "1.4.1"),
        )
    if source_type == "synthseg":
        from .sources.synthseg import SynthSegSource
        extra_roots = [Path(p) for p in cfg.get("extra_roots", [])]
        return SynthSegSource(
            intermediate_root=Path(cfg["intermediate_root"]),
            modality=cfg.get("modality", "T1w"),
            extra_roots=extra_roots or None,
        )
    if source_type == "msd":
        from .sources.msd import MSDDecathlonSource
        return MSDDecathlonSource(
            src_root=Path(cfg["src_root"]),
            task=cfg.get("task"),
        )
    if source_type == "totalseg":
        from .sources.totalseg import TotalSegSource
        return TotalSegSource(
            src_root=Path(cfg["src_root"]),
            meta_csv=Path(cfg["meta_csv"]) if cfg.get("meta_csv") else None,
        )
    if source_type == "luna16":
        from .sources.luna16 import LunaSource
        return LunaSource(
            src_root=Path(cfg["src_root"]),
            annotations_csv=Path(cfg["annotations_csv"]) if cfg.get("annotations_csv") else None,
        )
    if source_type == "btcv":
        from .sources.btcv import BTCVSource
        return BTCVSource(src_root=Path(cfg["src_root"]))
    if source_type == "verse":
        from .sources.verse import VerSeSource
        return VerSeSource(src_root=Path(cfg["src_root"]))
    if source_type == "fused":
        from .sources.rtstruct_ts_fused import RTStructTSFusedSource
        return RTStructTSFusedSource(
            rtstruct_root=Path(cfg["rtstruct_root"]),
            tcia_root=Path(cfg["tcia_root"]),
            series_csv=Path(cfg["series_csv"]),
            ts_root=Path(cfg["ts_root"]),
            rtstruct_structures=cfg["rtstruct_structures"],
            ts_structures=cfg["ts_structures"],
        )
    if source_type == "mixed_optic":
        from .sources.mixed_optic import MixedOpticSource
        return MixedOpticSource(
            pddca_root=Path(cfg["pddca_root"]),
            pddca_version=cfg.get("pddca_version", "1.4.1"),
            ts_v2_image_root=Path(cfg["ts_v2_image_root"]),
            ts_orbit_manifest=Path(cfg["ts_orbit_manifest"]),
            ts_pseudo_root=Path(cfg["ts_pseudo_root"]),
        )
    if source_type == "segrap":
        from .sources.segrap import SegRapSource
        return SegRapSource(
            src_root=Path(cfg["src_root"]),
            phase=cfg.get("phase", "both"),
            splits=tuple(cfg.get("splits", ("Training", "Validation"))),
            merge_subparts=cfg.get("merge_subparts"),
        )
    if source_type == "longct":
        from .sources.longitudinal_ct import LongitudinalCTSource
        return LongitudinalCTSource(src_root=Path(cfg["src_root"]))
    if source_type == "clinical_rtstruct":
        from .sources.clinical_rtstruct import ClinicalRTStructSource
        return ClinicalRTStructSource(
            root=Path(cfg["root"]),
            aliases=cfg.get("aliases"),
            partial_label=spec.partial_label,
        )
    if source_type == "index_paired_rtstruct":
        from .sources.index_paired_rtstruct import IndexPairedRTStructSource
        return IndexPairedRTStructSource(
            root=Path(cfg["root"]),
            image_subdir=cfg.get("image_subdir", "dicomTr"),
            rtstruct_subdir=cfg.get("rtstruct_subdir", "rtstructTr"),
            aliases=cfg.get("aliases"),
            rtstruct_glob=cfg.get("rtstruct_glob", "*.dcm"),
            image_glob=cfg.get("image_glob"),
            partial_label=spec.partial_label,
        )
    if source_type == "vendor_rtstruct":
        from .sources.vendor_rtstruct import VendorRTStructSource
        return VendorRTStructSource(
            root=Path(cfg["root"]),
            aliases=cfg.get("aliases"),
            partial_label=spec.partial_label,
        )
    raise ValueError(f"unknown source_type {source_type!r}")


# ── per-case worker ──────────────────────────────────────────────────────

def _convert_one_case(
    source_type: str,
    spec_key: str,
    case: CaseRef,
    out_root_str: str,
) -> tuple[str, str | None, dict[str, int], list[str]]:
    """Convert a single case.

    Returns (case_id, error_or_None, overlap_stats, annotated_canonicals).
    `annotated_canonicals` is the subset of the spec's structures actually
    contoured in this case (== all of them in normal mode; a per-case subset in
    partial-label mode). Called from worker processes; re-imports fresh per call.
    """
    out_root = Path(out_root_str)
    spec = SPECS[spec_key]
    source = _build_source(source_type, spec)

    try:
        image = source.load_image(case)
    except Exception as e:
        return case.case_id, f"image_read: {e}", {}, []

    arr_shape = image.GetSize()[::-1]  # (Z, Y, X)
    label = np.zeros(arr_shape, dtype=np.uint8)
    overlaps: dict[str, int] = {}
    annotated: list[str] = []

    for idx, sm in enumerate(spec.structures, start=1):
        source_name = sm.name_in(source_type)
        try:
            mask_img = source.load_mask(case, source_name, image)
        except KeyError:
            if spec.partial_label:
                # Structure not contoured on this case: skip it (don't paint,
                # don't abort). The trainer masks this organ's channel for this
                # case so background-painted voxels never become a false "not
                # this organ" signal. label id `idx` is intentionally left unused
                # for this case, keeping ids stable across the cohort.
                continue
            return case.case_id, f"missing_struct: {sm.canonical}", {}, []
        except Exception as e:
            return case.case_id, f"mask_read({sm.canonical}): {e}", {}, []

        mask = sitk.GetArrayFromImage(mask_img).astype(bool)
        if spec.partial_label and not mask.any():
            # This organ is not contoured on this case. The clinical_rtstruct
            # source returns an all-zero mask (rather than raising KeyError) when
            # the ROI is absent, so the KeyError branch above doesn't catch it.
            # Treat it as un-annotated: don't paint, don't record — so the
            # trainer masks this channel for this case instead of supervising it
            # as all-background (which would teach a false negative).
            continue
        already = label > 0
        overlap = int((mask & already).sum())
        if overlap:
            overlaps[sm.canonical] = overlap
        # earlier classes win
        label[mask & ~already] = idx
        annotated.append(sm.canonical)

    if not label.any():
        return case.case_id, "empty_label", overlaps, []

    ds_dir = out_root / "datasets" / spec.folder
    (ds_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    img_path = ds_dir / "imagesTr" / f"{case.case_id}_0000.nii.gz"
    lbl_path = ds_dir / "labelsTr" / f"{case.case_id}.nii.gz"

    sitk.WriteImage(image, str(img_path), useCompression=True)

    label_img = sitk.GetImageFromArray(label)
    label_img.CopyInformation(image)
    sitk.WriteImage(label_img, str(lbl_path), useCompression=True)

    return case.case_id, None, overlaps, annotated


# ── orchestration ────────────────────────────────────────────────────────

def _channel_names_for(spec: DatasetSpec) -> dict[str, str]:
    """nnUNet channel_names for the emitted dataset.json.

    nnUNet selects CTNormalization iff the channel name is the literal "ct"
    (case-insensitive); every other name falls through to ZScoreNormalization,
    which is the correct per-image scheme for MR. So MR datasets must NOT be
    labelled "CT" or they silently get CT HU-normalization. Resolution order:
    explicit ``spec.channel_names`` > derive from ``tags["modality"]`` >
    default CT (preserves legacy behaviour for every CT spec).
    """
    if spec.channel_names:
        return dict(spec.channel_names)
    modality = (spec.tags.get("modality") or "CT").upper()
    if modality.startswith("CT"):
        return {"0": "CT"}
    if modality.startswith("MR"):
        return {"0": "MRI"}
    return {"0": modality}


def _partial_label_payload(spec: DatasetSpec, annotations: dict[str, list[str]]) -> dict:
    """Build the partial-label map: per-case annotated organs + per-organ coverage.

    Embedded in dataset.json under "partial_label_annotations" (so it travels
    through nnUNet preprocessing into self.dataset_json at train time) AND written
    standalone as partial_label_annotations.json for inspection.
    """
    label_ids = {s.canonical: i for i, s in enumerate(spec.structures, start=1)}
    coverage: dict[str, int] = {s.canonical: 0 for s in spec.structures}
    for ann in annotations.values():
        for name in ann:
            coverage[name] = coverage.get(name, 0) + 1
    return {
        "structures": [s.canonical for s in spec.structures],  # order == label id
        "label_ids": label_ids,
        "coverage": coverage,
        "per_case": annotations,
    }


def _write_dataset_json(
    out_root: Path, spec: DatasetSpec, n_train: int, partial_payload: dict | None = None,
) -> None:
    dj = {
        "channel_names": _channel_names_for(spec),
        "labels": spec.label_map,
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": spec.folder,
        "description": spec.description,
        "tags": spec.tags,
    }
    if spec.partial_label:
        # Region-based: one INDEPENDENT binary region per organ (sigmoid output),
        # so the partial-label trainer can mask an un-annotated organ's channel
        # for a given case without coupling it to the other classes (as softmax
        # would). Each organ is a single-label region [i]; regions_class_order
        # maps the N sigmoid channels back to label ids. Background stays 0.
        # No `ignore` label is painted — un-annotated organs are simply absent
        # from the label file and masked per-case by nnUNetTrainerPartialLabelMLflow
        # via partial_label_annotations.json (NOT by a voxel ignore value, whose
        # location would be unknown).
        regions: dict[str, list[int] | int] = {"background": 0}
        order: list[int] = []
        for i, s in enumerate(spec.structures, start=1):
            regions[s.canonical] = [i]
            order.append(i)
        dj["labels"] = regions
        dj["regions_class_order"] = order
        dj["partial_label"] = True
        if partial_payload is not None:
            # Embed so the per-case annotation map survives nnUNet preprocessing
            # and is readable as self.dataset_json["partial_label_annotations"]
            # in nnUNetTrainerPartialLabelMLflow (no nnUNet_raw path-guessing).
            dj["partial_label_annotations"] = partial_payload
    (out_root / "datasets" / spec.folder / "dataset.json").write_text(
        json.dumps(dj, indent=2)
    )


def _write_splits(
    out_root: Path,
    spec: DatasetSpec,
    cases: Sequence[CaseRef],
    seed: int = 12345,
) -> None:
    """Patient-stratified 5-fold: all series for one patient share a fold."""
    if not cases:
        return
    # group case_ids by patient_id, then split patients into 5 folds
    by_patient: dict[str, list[str]] = {}
    for c in cases:
        by_patient.setdefault(c.patient_id, []).append(c.case_id)

    patients = sorted(by_patient)
    rng = random.Random(seed)
    rng.shuffle(patients)
    fold_patients = [patients[i::5] for i in range(5)]

    splits = []
    for fold_idx in range(5):
        val_pids = set(fold_patients[fold_idx])
        val = sorted(c for p in val_pids for c in by_patient[p])
        train_pids = [p for p in patients if p not in val_pids]
        train = sorted(c for p in train_pids for c in by_patient[p])
        splits.append({"train": train, "val": val})

    preproc = out_root / "preprocessed" / spec.folder
    preproc.mkdir(parents=True, exist_ok=True)
    (preproc / "splits_final.json").write_text(json.dumps(splits, indent=2))
    log.info(
        "[%s] wrote 5-fold splits (n_cases=%d, n_patients=%d, seed=%d)",
        spec.folder, len(cases), len(patients), seed,
    )


def _write_lineage(out_root: Path, spec: DatasetSpec, cases: Sequence[CaseRef]) -> None:
    """Per-case source provenance — read by the MLflow trainer for tags."""
    rows = []
    for c in cases:
        rows.append({
            "case_id": c.case_id,
            "patient_id": c.patient_id,
            "image_path": str(c.image_path),
            **c.metadata,
        })
    out_path = out_root / "datasets" / spec.folder / "lineage.json"
    out_path.write_text(json.dumps(rows, indent=2))


def convert(
    spec_key: str,
    source_type: str,
    out_root: Path,
    workers: int,
    limit: int | None,
) -> int:
    if spec_key not in SPECS:
        log.error("unknown spec %r; available: %s", spec_key, sorted(SPECS))
        return 2
    spec = SPECS[spec_key]
    log.info("=== building %s from %s source ===", spec.folder, source_type)

    source = _build_source(source_type, spec)
    structure_source_names = [sm.name_in(source_type) for sm in spec.structures]
    cases = source.discover(structure_source_names)
    if limit and limit > 0:
        cases = cases[:limit]
    if not cases:
        log.error("no cases discovered — aborting")
        return 2

    converted: list[CaseRef] = []
    errors: list[tuple[str, str]] = []
    overlap_summary: dict[str, int] = {}
    annotations: dict[str, list[str]] = {}  # case_id -> annotated canonicals (partial mode)

    workers = max(1, workers)
    log.info("converting %d cases with %d workers", len(cases), workers)

    if workers == 1:
        for c in cases:
            cid, err, ovr, ann = _convert_one_case(source_type, spec_key, c, str(out_root))
            if err:
                errors.append((cid, err))
                log.warning("[%s] %s", cid, err)
            else:
                converted.append(c)
                annotations[cid] = ann
                log.info("[%s] ok", cid)
            for k, v in ovr.items():
                overlap_summary[k] = overlap_summary.get(k, 0) + v
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_convert_one_case, source_type, spec_key, c, str(out_root)): c
                for c in cases
            }
            for fut in as_completed(futures):
                c = futures[fut]
                cid, err, ovr, ann = fut.result()
                if err:
                    errors.append((cid, err))
                    log.warning("[%s] %s", cid, err)
                else:
                    converted.append(c)
                    annotations[cid] = ann
                    log.info("[%s] ok", cid)
                for k, v in ovr.items():
                    overlap_summary[k] = overlap_summary.get(k, 0) + v

    if not converted:
        log.error(
            "0/%d cases converted — nothing written. First errors: %s",
            len(cases), errors[:10],
        )
        return 3

    partial_payload = _partial_label_payload(spec, annotations) if spec.partial_label else None
    _write_dataset_json(out_root, spec, n_train=len(converted), partial_payload=partial_payload)
    _write_splits(out_root, spec, converted)
    _write_lineage(out_root, spec, converted)
    if partial_payload is not None:
        (out_root / "datasets" / spec.folder / "partial_label_annotations.json").write_text(
            json.dumps(partial_payload, indent=2)
        )
        log.info("[%s] partial-label coverage (cases per organ): %s",
                 spec.folder, partial_payload["coverage"])

    log.info(
        "=== %s: %d converted, %d errored ===",
        spec.folder, len(converted), len(errors),
    )
    if overlap_summary:
        log.warning("aggregate label-overlap voxels (earlier-class-wins): %s", overlap_summary)
    if errors:
        log.warning("error sample: %s", errors[:5])
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Convert a labelled cohort to nnUNetv2 raw layout")
    p.add_argument("--spec", required=True, help=f"spec key, one of: {sorted(SPECS)}")
    p.add_argument("--source", required=True,
                   choices=("rtstruct", "pddca", "synthseg", "msd", "totalseg", "luna16", "btcv", "verse", "fused", "mixed_optic", "segrap", "longct", "clinical_rtstruct", "index_paired_rtstruct", "vendor_rtstruct"),
                   help="source adapter to use (must be present in the spec's source_constraints)")
    p.add_argument("--out", type=Path, required=True,
                   help="output root containing datasets/ and preprocessed/")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="convert only first N cases (0 = no limit; smoke-test mode)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return convert(args.spec, args.source, args.out, args.workers, args.limit)


if __name__ == "__main__":
    sys.exit(main())
