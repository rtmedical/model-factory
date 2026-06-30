# Longitudinal-CT Tübingen — schema, prevalence audit, ingestion notes

**Dataset**: Longitudinal-CT v2 (Küstner et al, Sci Data 2026)
**DOI**: 10.57754/FDAT.75kj1-64747
**FDAT record**: https://fdat.uni-tuebingen.de/records/75kj1-64747
**Licence**: CC-BY-4.0 (Green tier — commercial-OK with attribution)
**Local raw root**: `/data/model-factory-nfs/raw_external/longitudinal_ct_tubingen/`
**Local unpacked root**: `/data/model-factory-nfs/raw_external/longitudinal_ct_tubingen/unpacked/`

300 melanoma patients, whole-body portal-venous CT, baseline + follow-up.
Manual lesion segmentations, labelled by anatomical site of metastasis.

## On-disk layout (after `unzip`)

```
unpacked/
├── data_split.json                       # official 240/30/30 patient split
├── inputsTr/
│   ├── <patient>.csv                     # one CSV per patient (lesion table)
│   ├── <patient>_<BL|FU>_<NN>.json       # "Points of interest" - centroids
│   ├── <patient>_<BL|FU>_img_<NN>.nii.gz # CT volume
│   └── <patient>_<BL|FU>_mask_<NN>.nii.gz# BL masks (FU masks live in targetsTr/)
└── targetsTr/
    └── <patient>_FU_mask_<NN>.nii.gz     # FU masks (challenge-target side)
```

- `<patient>` = 10-char hex hash (e.g. `006f52e910`)
- `<BL|FU>` = baseline vs follow-up timepoint
- `<NN>` = series index within a timepoint (some scans split into 2 series, e.g.
  head/neck + abdomen/pelvis), zero-padded. So one timepoint can produce
  multiple `(image, mask)` pairs.

**Total**: 1004 `_img_*.nii.gz` in inputsTr, 334 `_mask_*.nii.gz` in inputsTr (BL),
336 `_FU_mask_*.nii.gz` in targetsTr. **670 image+mask training pairs across 300
patients** (some have multiple series per timepoint).

## CSV schema (`inputsTr/<patient>.csv`)

One row per lesion. Joins lesion-id (the voxel value in the mask) to anatomy:

| Column | Meaning |
|---|---|
| `lesion_id` | integer; matches mask voxel value (1..N per patient) |
| `cog_bl` | BL centroid as `"x y z"` in image coords |
| `cog_backpropagated` | usually empty |
| `img_id_bl` | which BL series carries this lesion (0 or 1) — selects the mask file `_BL_mask_<NN>.nii.gz` |
| `cog_propagated` | BL→FU propagated centroid |
| `cog_fu` | actual FU centroid (empty if lesion disappeared) |
| `img_id_fu` | which FU series carries this lesion |
| `lesion_type` | **anatomy class string** (drives our segmentation labels) |
| `topology_class` | UNCHANGED / DISAPPEARING / NEW / MERGED (for change-tracking) |
| `merged_into` | for merge events |
| `volume_bl` | mm³ at BL |
| `volume_fu` | mm³ at FU |
| `target_lesion` | challenge-scoring flag |
| `use_for_challenge` | challenge-scoring flag |
| `linking_unclear` | dataset-quality flag |

**Critical mechanic**: mask voxels carry `lesion_id` (per-patient integer 1..N),
NOT the anatomy class directly. The converter must JOIN the CSV to map
`lesion_id → lesion_type` before writing the multi-class label NIfTI.

## Acquisition parameters (observed)

- **Slice thickness**: 3 mm axial (Z spacing). Same across the dataset.
- **In-plane spacing**: varies 0.518 - 0.793 mm (Siemens scanners, FOV-dependent).
- **CT range**: clipped to [-1024, 3071] HU (or [-1024, 1625] on some series).
- **Volume shape**: 512×512 in plane, Z varies 70 - 245 slices depending on FOV.
- **Phase**: portal-venous IV contrast.
- **Scanners**: 5 Siemens models (Force, Sensation 64, Definition AS, Definition
  Flash, Biograph128 PET/CT). Standardised acquisition protocol per the data
  descriptor.

**Implication for planner choice**: 3 mm axial slice thickness makes small
foreground classes (lymph nodes, adrenals, soft-tissue, skeleton) sub-voxel in
the through-plane direction. The `nnUNetPlannerResEncL_HighRes` planner (refines
in-plane to 0.7 mm while preserving axial) is the right pick for the lymph-node
specialist (D102).

## Per-anatomy prevalence (audit, 2026-05-20)

Run: `audit_fast.py` in the dataset root. Joins all 300 CSVs to all 670 mask
volumes. Voxel→ml uses 0.7×0.7×3.0 mm = 0.00147 ml/voxel as a median-spacing
reference (in-plane varies 0.5-0.8 mm, so volumes are accurate within ~10 %).

| anatomy | n_cases | n_lesions | total_voxels | total_ml | median_lesion_vox | median_lesion_ml | max_lesion_vox |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lymph node | 344 | 1674 | 8,211,215 | 12,070 | 555 | 0.82 | 592,952 |
| Lung | 315 | 2286 | 2,294,028 | 3,372 | 50 | 0.07 | 244,151 |
| Soft tissue / Skin | 222 | 1371 | 3,121,448 | 4,588 | 163 | 0.24 | 546,639 |
| Liver | 180 | 1033 | 5,103,736 | 7,502 | 236 | 0.35 | 794,908 |
| Others | 116 | 301 | 5,461,650 | 8,028 | 674 | 0.99 | 1,022,951 |
| Skeleton | 76 | 202 | 384,366 | 565 | 425 | 0.62 | 84,800 |
| Adrenals | 53 | 75 | 3,445,573 | 5,065 | 7,167 | 10.54 | 885,954 |
| unclear | 49 | 116 | 341,449 | 502 | 173 | 0.25 | 119,987 |
| Spleen | 27 | 91 | 87,521 | 129 | 151 | 0.22 | 15,483 |
| Kidney | 4 | 12 | 157,732 | 232 | 4,083 | 6.00 | 89,528 |
| Heart | 4 | 4 | 11,039 | 16 | 2,226 | 3.27 | 6,162 |
| CNS | 4 | 17 | 12,895 | 19 | 386 | 0.57 | 4,735 |

Empty masks (0 foreground voxels, FU scans where lesions disappeared
post-therapy): **25 / 670 — kept, valid background examples.**
Unmatched mask voxel-IDs (mask carries a lesion_id with no CSV row): **0**.

## Decision: D100 (multi-class generalist) label schema

Drop classes with <30 cases (would collapse to dice = 0). Merge "unclear" into
"Other" since both are catch-all classes — preserves foreground signal where
anatomy can't be cleanly assigned.

**Kept (8 foreground classes for D100)**:

| code | canonical | source `lesion_type` strings |
|---:|---|---|
| 1 | `LymphNode` | "Lymph node" |
| 2 | `Lung` | "Lung" |
| 3 | `SoftTissueSkin` | "Soft tissue / Skin" |
| 4 | `Liver` | "Liver" |
| 5 | `Other` | "Others", "unclear" |
| 6 | `Skeleton` | "Skeleton" |
| 7 | `Adrenal` | "Adrenals" |
| 8 | `Spleen` | "Spleen" |

**Dropped from D100 (too sparse — <5 cases each)**:
- Kidney (4 cases) — clinically rare site for melanoma mets
- Heart (4 cases) — clinically rare
- CNS (4 cases) — CT is the wrong modality for brain mets anyway; revisit if
  we add MR

Class-ordering rule (per `_write_dataset_json`): label integers are assigned by
the order of `structures` in the DatasetSpec. We order by case count descending
so the most-prevalent class gets label 1; this is convention, not load-bearing.

## D101 binary spec (any-lesion-vs-background)

Same source, single foreground class. Every non-zero voxel in any mask → 1.
- 645 / 670 cases have ≥1 foreground voxel
- Total foreground: ~28 M voxels (~41 L) across all training cases
- Massive class imbalance vs whole-body background → use
  `nnUNetTrainerSmallStructuresMLflow` (Tversky α=0.3 β=0.7 + 95 % fg oversample)

## Specialist specs

Single-class binary models trained per anatomy. All use the same source
adapter — only the spec differs.

| Spec | n_cases | n_lesions | median ml | trainer | planner | status |
|---|---:|---:|---:|---|---|---|
| D102 LymphNode | 344 | 1674 | 0.82 | SmallStructures | HighRes | **Tier-1, in current campaign** |
| D106 LymphNode_Default | 344 | 1674 | 0.82 | default | default ResEncL | **Tier-1 A/B baseline for D102** |
| D103 Adrenal | 53 | 75 | 10.54 | default | default ResEncL | Tier-2, queued |
| D104 Skeleton | 76 | 202 | 0.62 | default | default ResEncL | Tier-2, queued |
| D105 SoftTissueSkin | 222 | 1371 | 0.24 | SmallStructures | HighRes | Tier-2, queued |

**Tier-1 lymph-node A/B (D102 + D106)** — both specs point at the identical
cohort and label schema; the only difference is the training recipe. Submit
both 5-fold; compare cross-fold mean dice in MLflow to confirm whether
Tversky + HighRes in-plane refinement is actually load-bearing for melanoma
lymph-node mets, or whether the default Dice+CE + default ResEncL planner
already converges on this cohort. Tags `ab_partner` and `ab_role` link the
two specs.

D102 motivation: the smallest lymph nodes are 1-2 voxels thick in Z at the
dataset's 3 mm axial slice, which historically motivates HighRes (refines
in-plane to 0.7 mm while preserving axial) + Tversky FN-weighting. D054
PDDCA optic nerves showed this pipeline can rescue a cohort where the
default recipe stalls at dice=0. But the median lymph-node lesion here is
~555 voxels (~0.82 ml) — a third or fourth of those are big enough that
the default loss might converge fine. D106 is the empirical check.

**Tier-2 (D103-D105) — queued behind D100 per-class dice.** Code-complete
specs ready to launch, but convert + preprocess deferred until D100's
multi-class dice tells us which anatomies the generalist already covers
adequately. The specs carry `tags.deploy_blocked_on=D100_per_class_dice`
so any future scheduler can honour that gate.

Per-spec rationale:

- **D103 Adrenal** — only 53 cases but lesions are LARGE (P90 ~45 ml,
  whole-gland involvement). Standard trainer/planner suffice since the
  structures are big and not sparse. No existing factory specialist;
  clinically top-3 melanoma met site.
- **D104 Skeleton** — 76 cases, lytic/sclerotic mix on bone window.
  Default trainer + default planner — bone mets are not small enough to
  need Tversky/HighRes (median 0.62 ml, 3-5 slices thick in Z), and
  they are well-localised to the skeleton so the default balanced
  Dice+CE loss should converge cleanly. Future enhancement (post-baseline):
  2-channel input with bone window as the second channel.
- **D105 SoftTissueSkin** — 222 cases, 1371 lesions, signature melanoma
  pattern (in-transit, cutaneous, subcutaneous). Median bbox 3x9x8 voxels;
  smallest lesions are 1-2 slices thick → same SmallStructures + HighRes
  pipeline as D102.

**Deferred / dropped from this dataset**:

- Lung-met specialist — 315 cases available but overlaps LUNA16 (D036) +
  MSD-Lung (D080). The multi-class D100 will produce a Lung output channel;
  only build a dedicated specialist if D100's Lung dice is weak.
- Liver-met specialist — 180 cases but overlaps MSD-HepaticVessel (D033).
  Note: probe found 21% of liver lesions are single-slice — flag for
  review before any future liver-met training.
- "Other" / "unclear" specialist — heterogeneous by definition; won't
  converge. Stays as label 5 in D100.

## Official splits

`data_split.json` at archive root:
- **train**: 240 patients
- **val**: 30 patients
- **test**: 30 patients
- Total: 300 patients ✓

For our 5-fold CV the converter ignores `data_split.json` and generates
patient-stratified 5-folds from the train+val pool (270 patients). Test
patients are converted but tagged so we can hold them out as a final eval
matching the challenge cohort. See spec tags `held_out_test_patients: 30`.

## Adapter requirements (Phase 1)

The `LongitudinalCTSource` adapter must:

1. **Discover**: walk `inputsTr/` for `*_img_*.nii.gz`. For each, locate the
   paired mask:
   - BL images → mask in `inputsTr/<patient>_BL_mask_<NN>.nii.gz`
   - FU images → mask in `targetsTr/<patient>_FU_mask_<NN>.nii.gz`
   - Skip if mask missing.
2. **case_id** = `<patient>_<BL|FU>_<NN>` (e.g. `006f52e910_BL_00`)
   **patient_id** = `<patient>` (bare hash) — drives split stratification so
   all of one patient's scans land in the same fold.
3. **Load image**: `sitk.ReadImage(case.image_path)`.
4. **Load mask** (per canonical anatomy name):
   - Read the per-patient CSV once (cached) → build `{lesion_id: lesion_type}`
     restricted to this case's `(timepoint, series_idx)` via `img_id_bl/fu`.
   - Read the mask NIfTI.
   - For each `lesion_id` in this scan, look up its `lesion_type`.
   - Build a binary mask = (mask_voxel ∈ {lesion_ids matching canonical}).
   - Special wildcard `canonical == "*"` for the D101 binary spec → OR of all
     non-zero voxels (no CSV join needed).

The orchestrator's "earlier classes win on overlap" semantics handle the
multi-class composition. There is no cross-anatomy overlap expected (one lesion
has exactly one `lesion_type`), so overlap stats should be 0.

## Operational notes

- **Total disk**: 51 GiB compressed → ~70 GiB uncompressed.
- **Per-patient lesion count**: median 5, max 100+.
- **CT clipping**: nnUNet's standard CT preprocessing (0.5/99.5 percentile clip
  + z-score) is fine; no custom intensity transform needed.
- **No defacing artifacts** to worry about in the chest/abdomen/pelvis FOV —
  this is whole-body but skull/face is defaced before publication; check
  individual cases if training a head-only specialist later.
- **PET pair availability**: 5 of the 300 patients also have PET-CT in
  autoPET-IV Task 1; we don't use PET here.

## Citation

Küstner T, Peisen F, Gatidis S, Wagner A, Megne O, et al.
**Longitudinal-CT: Annotated longitudinal CT studies of 300 melanoma patients
for therapy response assessment.** Scientific Data (Nature), 2026.
DOI: 10.1038/s41597-026-07466-y (paper), 10.57754/FDAT.75kj1-64747 (data).
Licence: CC-BY-4.0.

Every model trained from this data must carry the tags:
`dataset_license=CC-BY-4.0`, `dataset_doi=10.57754/FDAT.75kj1-64747`,
`attribution_required=true`, `base_corpus=Longitudinal-CT-v2`.
