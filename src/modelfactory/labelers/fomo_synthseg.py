"""Run SynthSeg over a curated FOMO300K T1 manifest and emit cleaned
per-structure binary masks for Dataset045_Brain_MR_CoreOAR.

Input:
    manifest TSV with columns:
        subject_id, cohort, zip_path, internal_t1_path

Per-subject pipeline:
    1. Extract just the T1w file from the subject's session zip into a
       temp working directory.
    2. Run `mri_synthseg --robust --qc` on the T1.
    3. Parse the QC json (single-value score in [0,1]) and reject if < QC_MIN.
    4. From the SynthSeg multi-label output, extract our 4 canonical OAR
       classes by re-mapping FreeSurfer aseg label IDs:
           Brainstem      ← {16}
           Hippocampus_L  ← {17}
           Hippocampus_R  ← {53}
           Cerebellum     ← {7, 8, 46, 47}   (WM + cortex, both sides)
    5. For each class, keep the largest connected component (drops any
       SynthSeg fragmentation noise).
    6. Sanity-check volumes: brainstem 5-60 cm³, hippocampus 1-8 cm³,
       cerebellum 50-250 cm³. Reject subjects with out-of-range volumes.
    7. Write the T1 + 4 binary masks to
       <out_root>/<subject_id>/image.nii.gz
       <out_root>/<subject_id>/structures/{Brainstem,Hippocampus_L,Hippocampus_R,Cerebellum}.nii.gz
       <out_root>/<subject_id>/qc.json

The labeller is restart-safe: if a subject's image.nii.gz already exists
under out_root, the subject is skipped. This lets the Job be re-run after
a node restart without redoing finished work.

Parallelism: the script can be sharded across pods via --shard N --of M
(round-robin assignment of subjects to shards).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk

log = logging.getLogger("synthseg_labeller")


# FreeSurfer aseg IDs that SynthSeg outputs. See
# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
#
# This dict covers BOTH Dataset045 (CoreOAR) and Dataset048 (DeepNuclei).
# Both datasets share the same SynthSeg base aseg run — the labeller
# writes every structure listed here for every subject, and the per-
# dataset convert step picks up only the structures named in the spec.
FS_IDS_BASE: dict[str, frozenset[int]] = {
    # Posterior fossa (Dataset049)
    "Brainstem":             frozenset({16}),
    "Cerebellum":            frozenset({7, 8, 46, 47}),
    # Ventricular system (Dataset057). ChoroidPlexus (FS 31/63) was originally
    # planned here but SynthSeg-2.0 does not emit those IDs (verified
    # 2026-05-14 against a cached seg_full.nii.gz: aseg IDs present are
    # [0,2,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,41,43,44,46,47,49,50,
    # 51,52,53,54,58,60] — 31 and 63 absent). Dropped; Dataset057 has 4 classes.
    "LateralVentricle_L":    frozenset({4}),
    "LateralVentricle_R":    frozenset({43}),
    "ThirdVentricle":        frozenset({14}),
    "FourthVentricle":       frozenset({15}),
    # Diencephalon (Dataset052) — thalamus + ventral diencephalon
    # (hypothalamus + subthalamus + mammillary bodies + substantia nigra)
    "Thalamus_L":            frozenset({10}),
    "Thalamus_R":            frozenset({49}),
    "VentralDC_L":           frozenset({28}),
    "VentralDC_R":           frozenset({60}),
    # Basal ganglia (Dataset051) — striatum + pallidum + nucleus accumbens
    "Caudate_L":             frozenset({11}),
    "Caudate_R":             frozenset({50}),
    "Putamen_L":             frozenset({12}),
    "Putamen_R":             frozenset({51}),
    "Pallidum_L":            frozenset({13}),
    "Pallidum_R":            frozenset({52}),
    "NucleusAccumbens_L":    frozenset({26}),
    "NucleusAccumbens_R":    frozenset({58}),
    # Medial temporal (Dataset053) — hippocampus + amygdala
    "Hippocampus_L":         frozenset({17}),
    "Hippocampus_R":         frozenset({53}),
    "Amygdala_L":            frozenset({18}),
    "Amygdala_R":            frozenset({54}),
}

# Desikan-Killiany ROI codes from SynthSeg --parc, grouped into bilateral
# cortical-region classes. Left hemisphere ROIs are 1000-series, right
# hemisphere 2000-series.
#
# Cingulate codes (1002 caudalanteriorcingulate, 1010 isthmuscingulate,
# 1023 posteriorcingulate, 1026 rostralanteriorcingulate) used to be
# mis-grouped — 1002 under Frontal, 1010 under Parietal, the other two
# silently dropped. They now live in a dedicated Cingulate class, so the
# four lobes are anatomically clean (no medial-wall contamination) and
# Cingulate is recoverable as its own structure for Dataset055.
LOBE_GROUPS: dict[str, frozenset[int]] = {
    # Frontal lobe (no medial cingulate)
    "Frontal_L":   frozenset({1003, 1012, 1014, 1017, 1018, 1019,
                              1020, 1024, 1027, 1028, 1032}),
    "Frontal_R":   frozenset({2003, 2012, 2014, 2017, 2018, 2019,
                              2020, 2024, 2027, 2028, 2032}),
    # Temporal lobe
    "Temporal_L":  frozenset({1001, 1006, 1007, 1009, 1015, 1016, 1030,
                              1033, 1034}),
    "Temporal_R":  frozenset({2001, 2006, 2007, 2009, 2015, 2016, 2030,
                              2033, 2034}),
    # Parietal lobe (no medial isthmus cingulate)
    "Parietal_L":  frozenset({1008, 1022, 1025, 1029, 1031}),
    "Parietal_R":  frozenset({2008, 2022, 2025, 2029, 2031}),
    # Occipital lobe
    "Occipital_L": frozenset({1005, 1011, 1013, 1021}),
    "Occipital_R": frozenset({2005, 2011, 2013, 2021}),
    # Insula
    "Insula_L":    frozenset({1035}),
    "Insula_R":    frozenset({2035}),
    # Cingulate (caudal+rostral anterior + isthmus + posterior, per hemisphere)
    "Cingulate_L": frozenset({1002, 1010, 1023, 1026}),
    "Cingulate_R": frozenset({2002, 2010, 2023, 2026}),
}

# Plausible adult-brain volume ranges in cm³. Subjects outside these get
# rejected as SynthSeg failures rather than pathology — paediatric and
# severely atrophic patients may legitimately fall outside; for a
# silver-label cohort we err toward over-rejection.
VOL_RANGE_CC: dict[str, tuple[float, float]] = {
    # base structures
    "Brainstem":          (5.0,  60.0),
    "Cerebellum":         (50.0, 250.0),
    "LateralVentricle_L": (1.0,  100.0),  # ventricles vary hugely with age
    "LateralVentricle_R": (1.0,  100.0),
    "ThirdVentricle":     (0.2,  8.0),    # midline; expands with age
    "FourthVentricle":    (0.2,  6.0),    # posterior fossa CSF
    "Thalamus_L":         (3.0,  15.0),
    "Thalamus_R":         (3.0,  15.0),
    "VentralDC_L":        (1.5,  9.0),    # hypothalamus + subthalamus + SN
    "VentralDC_R":        (1.5,  9.0),
    "Caudate_L":          (1.5,  7.0),
    "Caudate_R":          (1.5,  7.0),
    "Putamen_L":          (2.5,  10.0),
    "Putamen_R":          (2.5,  10.0),
    "Pallidum_L":         (0.8,  4.0),
    "Pallidum_R":         (0.8,  4.0),
    "NucleusAccumbens_L": (0.1,  2.0),    # ventral striatum; tiny
    "NucleusAccumbens_R": (0.1,  2.0),
    "Hippocampus_L":      (1.0,  8.0),
    "Hippocampus_R":      (1.0,  8.0),
    "Amygdala_L":         (0.3,  4.0),    # adjacent MTL, ~1.7 cm³ ± 0.2
    "Amygdala_R":         (0.3,  4.0),
    # lobar (typical adult per hemisphere). Mins relaxed 2026-05-14 because
    # the LOBE_GROUPS cingulate fix moved DK codes 1002/1010 (caudalanterior
    # + isthmus cingulate) into Cingulate_L/R — Frontal lost ~3-5 cm³,
    # Parietal lost ~2-3 cm³, on a typical hemisphere.
    "Frontal_L":          (80.0, 350.0),
    "Frontal_R":          (80.0, 350.0),
    "Temporal_L":         (40.0, 180.0),
    "Temporal_R":         (40.0, 180.0),
    "Parietal_L":         (45.0, 200.0),
    "Parietal_R":         (45.0, 200.0),
    "Occipital_L":        (20.0, 120.0),
    "Occipital_R":        (20.0, 120.0),
    "Insula_L":           (3.0,  20.0),
    "Insula_R":           (3.0,  20.0),
    "Cingulate_L":        (4.0,  35.0),   # 4 DK sub-regions per hemisphere
    "Cingulate_R":        (4.0,  35.0),
}


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask containing only the largest 6-connected component."""
    if not mask.any():
        return mask
    try:
        import cc3d
    except ImportError:
        from scipy.ndimage import label as _label
        labelled, n = _label(mask.astype(np.uint8))
        if n <= 1:
            return mask
        # background = 0; find the largest non-zero label
        counts = Counter(labelled[labelled > 0].flat)
        biggest = counts.most_common(1)[0][0]
        return labelled == biggest
    labelled, n = cc3d.connected_components(mask.astype(np.uint32), connectivity=6, return_N=True)
    if n <= 1:
        return mask
    counts = np.bincount(labelled.flat)
    counts[0] = 0  # ignore background
    biggest = int(counts.argmax())
    return labelled == biggest


def _volume_cc(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    v_mm3 = float(mask.sum()) * spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    return v_mm3 / 1000.0


def _extract_t1_from_zip(zip_path: Path, internal: str, dest_dir: Path) -> Path:
    """Pull a single file out of the zip without expanding the rest."""
    out_path = dest_dir / "T1w.nii.gz"
    with zipfile.ZipFile(str(zip_path)) as zf:
        with zf.open(internal) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


SYNTHSEG_SCRIPT = "/opt/synthseg/scripts/commands/SynthSeg_predict.py"


def _run_synthseg(
    t1_path: Path,
    seg_path: Path,
    qc_path: Path,
    parc: bool = False,
) -> bool:
    """Invoke SynthSeg's standalone CLI with QC scoring.

    When `parc=True`, SynthSeg also runs the Desikan-Killiany cortical
    parcellation (ROI codes 1000-1035 / 2000-2035 appended to the
    output volume in addition to the standard aseg labels). This adds
    ~20-30 s per subject.

    Uses SynthSeg 1.0 bundled weights — see module docstring for the
    licensing reasoning.
    """
    # Use 2.0 weights when present (better dice ceiling — ~+2pp). If the
    # 2.0 weights aren't mounted, fall back to the bundled 1.0 weights
    # via --v1. We auto-detect each model by file presence (the v2
    # weights live in /data/synthseg_models on the host, hostPath-mounted
    # at /opt/synthseg/models in label-gen pods).
    #
    # Quality flags enabled when their weight files are present:
    #   --robust  →  synthseg_robust_2.0.h5  (better dice on OOD scans,
    #                ~+1-3 pp on clinical data; ~50% slower per subject)
    #   --qc      →  synthseg_qc_2.0.h5      (writes a per-region QC
    #                score CSV; we threshold via QC_MIN)
    threads = int(os.environ.get("SYNTHSEG_THREADS", "4"))
    models = Path("/opt/synthseg/models")
    have_v2     = (models / "synthseg_2.0.h5").is_file()
    have_robust = (models / "synthseg_robust_2.0.h5").is_file()
    have_qc     = (models / "synthseg_qc_2.0.h5").is_file()

    cmd = [
        "python", SYNTHSEG_SCRIPT,
        "--i", str(t1_path),
        "--o", str(seg_path),
        "--cpu",
        "--threads", str(threads),
    ]
    if not have_v2:
        cmd.append("--v1")
    if parc:
        cmd.append("--parc")
    if have_robust and have_v2:
        # robust requires the 2.0 robust weight file
        cmd.append("--robust")
    if have_qc and have_v2:
        cmd += ["--qc", str(qc_path)]
    else:
        # placeholder so downstream code reading qc_path doesn't crash
        qc_path.write_text("")
    log.debug("running: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        log.warning("SynthSeg failed (rc=%d) on %s\nstderr: %s",
                    res.returncode, t1_path, res.stderr[-400:])
        return False
    return True


def _label_set_for_mode(mode: str) -> dict[str, frozenset[int]]:
    """Return the canonical-name → FS-id-set dict for the requested mode."""
    if mode == "base":
        return FS_IDS_BASE
    if mode == "parc":
        return LOBE_GROUPS
    raise ValueError(f"unknown mode {mode!r} (expected 'base' or 'parc')")


def _read_qc_score(qc_path: Path) -> float | None:
    """SynthSeg writes a CSV (or sometimes plain text); parse defensively."""
    if not qc_path.is_file():
        return None
    try:
        text = qc_path.read_text().strip()
        # SynthSeg --qc produces a CSV with header + one numeric row.
        # The "general QC" score is the first numeric value.
        for line in text.splitlines():
            for tok in line.replace(",", " ").split():
                try:
                    v = float(tok)
                    if 0.0 <= v <= 1.0:
                        return v
                except ValueError:
                    continue
        return None
    except Exception:
        return None


def process_one(
    row: dict[str, str],
    out_root: Path,
    qc_min: float,
    mode: str = "base",
    keep_temp: bool = False,
) -> tuple[str, str | None, dict]:
    """Process one manifest row. Returns (subject_id, error_or_None, qc_dict).

    `mode`:
      - "base"  → SynthSeg aseg; writes structures from FS_IDS_BASE
      - "parc"  → SynthSeg --parc; writes lobar groupings from LOBE_GROUPS
                  (LOBE_GROUPS values are sets of cortical Desikan-Killiany
                   ROI codes; the union per class produces a single binary mask)

    Caching: `seg_full.nii.gz` is persisted to `out_root/<sid>/` on every
    successful run. Subsequent invocations — typically after extending
    FS_IDS_BASE / LOBE_GROUPS — reuse the cache and skip the SynthSeg
    subprocess + T1-from-zip extraction. That brings structure-set extensions
    down to ~30 min across all shards instead of the ~5-6 h fresh inference.
    """
    sid = row["subject_id"]
    label_set = _label_set_for_mode(mode)
    subj_out = out_root / sid
    if (subj_out / "image.nii.gz").is_file() and all(
        (subj_out / "structures" / f"{s}.nii.gz").is_file() for s in label_set
    ):
        return sid, "already-done", {}

    with tempfile.TemporaryDirectory(prefix=f"ss_{sid}_") as td:
        td_path = Path(td)
        seg_cache  = subj_out / "seg_full.nii.gz"
        image_path = subj_out / "image.nii.gz"
        qc_score: float | None = None
        from_cache = seg_cache.is_file() and image_path.is_file()

        if from_cache:
            try:
                seg_img = sitk.ReadImage(str(seg_cache))
                t1_img  = sitk.ReadImage(str(image_path))
            except RuntimeError as e:
                return sid, f"read-cache: {e}", {}
        else:
            try:
                t1 = _extract_t1_from_zip(Path(row["zip_path"]), row["internal_t1_path"], td_path)
            except (zipfile.BadZipFile, KeyError, FileNotFoundError) as e:
                return sid, f"extract: {e}", {}

            seg_path = td_path / "seg.nii.gz"
            qc_path  = td_path / "qc.csv"
            if not _run_synthseg(t1, seg_path, qc_path, parc=(mode == "parc")):
                return sid, "synthseg failed", {}

            qc_score = _read_qc_score(qc_path)
            if qc_score is not None and qc_score < qc_min:
                return sid, f"qc {qc_score:.2f} < {qc_min}", {"qc": qc_score}

            try:
                seg_img = sitk.ReadImage(str(seg_path))
                t1_img  = sitk.ReadImage(str(t1))
            except RuntimeError as e:
                return sid, f"read: {e}", {}

            # Persist the SynthSeg output + image immediately, BEFORE the
            # per-structure volume checks. If a structure later fails its
            # volume range, we keep the inference work — future re-runs
            # (with adjusted ranges or label-id mappings) reuse the cache
            # instead of paying SynthSeg's per-subject cost again.
            subj_out.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(seg_img, str(seg_cache), useCompression=True)
            sitk.WriteImage(t1_img, str(image_path), useCompression=True)

        seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.int32)
        spacing = tuple(seg_img.GetSpacing())  # (x, y, z) mm

        qc: dict[str, float] = {}
        if qc_score is not None:
            qc["qc_synthseg"] = qc_score
        structure_masks: dict[str, sitk.Image] = {}

        for name, ids in label_set.items():
            raw = np.isin(seg_arr, list(ids))
            cleaned = _largest_cc(raw)
            vol = _volume_cc(cleaned, spacing)
            lo, hi = VOL_RANGE_CC.get(name, (0.0, float("inf")))
            if not (lo <= vol <= hi):
                return sid, f"{name} volume {vol:.1f} cc outside [{lo},{hi}]", {**qc, name: vol}
            qc[f"{name}_cc"] = round(vol, 2)
            m_img = sitk.GetImageFromArray(cleaned.astype(np.uint8))
            m_img.CopyInformation(seg_img)
            structure_masks[name] = m_img

        # All sanity-checks passed; write per-structure masks. Re-running
        # with an extended label_set will rewrite existing structure masks
        # with the current FS-id/DK-code definitions — intentional so that
        # bug fixes (e.g. the LOBE_GROUPS cingulate split) propagate to
        # cached subjects. The seg_full.nii.gz + image.nii.gz were already
        # persisted above (or are pre-existing from cache).
        (subj_out / "structures").mkdir(parents=True, exist_ok=True)
        for name, m in structure_masks.items():
            sitk.WriteImage(m, str(subj_out / "structures" / f"{name}.nii.gz"), useCompression=True)

        # Merge new per-structure volumes into any existing qc.json so we
        # don't clobber qc_synthseg from the original fresh run.
        qc_json_path = subj_out / "qc.json"
        existing_qc: dict = {}
        if qc_json_path.is_file():
            try:
                existing_qc = json.loads(qc_json_path.read_text())
            except json.JSONDecodeError:
                existing_qc = {}
        final_qc = {**existing_qc, **qc, "cohort": row["cohort"], "mode": mode}
        qc_json_path.write_text(json.dumps(final_qc, indent=2))

        if keep_temp and not from_cache:
            # Legacy debug aid; cache is now always persisted above. This
            # keeps a copy of the freshly extracted T1 for manual inspection.
            shutil.copy(str(t1), str(subj_out / "t1_extracted.nii.gz"))

        return sid, None, final_qc


def main() -> int:
    p = argparse.ArgumentParser(description="SynthSeg labelling pipeline for FOMO300K → brain-MR datasets")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--mode", choices=("base", "parc"), default="base",
                   help="'base' = SynthSeg aseg (used by Dataset045 + 048); "
                        "'parc' = SynthSeg --parc Desikan-Killiany lobar grouping (Dataset047)")
    p.add_argument("--qc-min", type=float, default=0.7)
    p.add_argument("--shard", type=int, default=0, help="this pod's shard index (0-based)")
    p.add_argument("--of", type=int, default=1, help="total number of shards")
    p.add_argument("--limit", type=int, default=0, help="cap subjects (smoke-test)")
    p.add_argument("--keep-temp", action="store_true",
                   help="also save the full SynthSeg output volume per subject")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.manifest.is_file():
        log.error("manifest missing: %s", args.manifest)
        return 2
    if args.of < 1 or not (0 <= args.shard < args.of):
        log.error("bad shard/of: %d/%d", args.shard, args.of)
        return 2

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with args.manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i % args.of == args.shard:
                rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    log.info("shard %d/%d: %d subjects", args.shard, args.of, len(rows))

    ok = 0
    skipped = 0
    errors: Counter = Counter()
    for r in rows:
        sid, err, qc = process_one(r, args.out_root, args.qc_min, args.mode, args.keep_temp)
        if err is None:
            ok += 1
            log.info("[%s] ok %s", sid, qc)
        elif err == "already-done":
            skipped += 1
        else:
            errors[err.split(":")[0]] += 1
            log.warning("[%s] %s", sid, err)

    log.info(
        "shard %d/%d done: %d ok, %d already-done, %d errors. error breakdown: %s",
        args.shard, args.of, ok, skipped, sum(errors.values()), dict(errors),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
