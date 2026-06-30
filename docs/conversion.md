# Converting data into nnU-Net datasets

model-factory builds nnU-Net v2 raw datasets from your source data with a small,
pluggable framework: a **`DatasetSpec`** declares *what* a dataset is (id, name,
structures, source constraints, tags), and a **`DatasetSource`** adapter knows
*how* to read one on-disk format. The orchestrator (`datasets/convert.py`) pairs
them, discovers matching cases, rasterizes labels in parallel, and writes the
nnU-Net `Dataset###_*` layout plus `dataset.json` and `splits_final.json`.

## The model

```python
from modelfactory.datasets.specs import DatasetSpec, StructureMapping

DatasetSpec(
    dataset_id=100,
    name="Hippocampus",
    description="MSD Task04 hippocampus.",
    structures=(
        StructureMapping(canonical="Anterior"),
        StructureMapping(canonical="Posterior"),
    ),
    source_constraints={"msd": {"task_root": "/in/Task04_Hippocampus"}},
    tags={"region": "brain", "modality": "MR", "dataset_license": "CC-BY-SA-4.0"},
)
```

- **`structures`** — the canonical, source-independent class names (the model's
  label ordering). `aliases={"<source>": "<on-disk name>"}` maps a source's own
  naming onto the canonical name.
- **`source_constraints`** — per-source-type settings (roots, CSVs, filters).
- **`tags`** — `region`, `modality` (drives CT vs MR normalization),
  `dataset_license`, `base_model`, etc. Tags flow into MLflow for lineage.
- **`partial_label=True`** — keep cases that contour *at least one* structure
  (not all), emit a per-case annotation sidecar, and train with
  `nnUNetTrainerPartialLabelMLflow` so one generalist can cover organs that never
  co-occur in a single case.

## Built-in source adapters

`src/modelfactory/datasets/sources/` ships adapters for common public layouts:

| `--source` | Format |
|---|---|
| `msd` | Medical Segmentation Decathlon (NIfTI + dataset.json) |
| `pddca` | PDDCA head & neck (NRRD) |
| `luna16` | LUNA16 lung nodules (CSV + raw) |
| `totalseg` | TotalSegmentator bundles (per-case `ct.nii.gz` + `segmentations/`) |
| `btcv`, `verse`, `segrap`, `synthseg` | the respective challenge layouts |
| `rtstruct` | generic DICOM CT/MR + RTSTRUCT (the radiotherapy-planning case) |

Cohort-specific adapters that embed internal directory conventions live in a
private overlay (see `overlays/README.md`).

## Running a conversion

```bash
python -m modelfactory.datasets.convert \
    --spec hippocampus --source msd --out /factory --workers 16
# -> /factory/datasets/Dataset100_Hippocampus/{imagesTr,labelsTr,dataset.json}
# -> /factory/preprocessed/.../splits_final.json
```

In-cluster, conversion runs as a Job (`infra/kustomize/convert-job.yaml.j2`,
`preprocess-job.yaml.j2`) with your source data mounted read-only.

## Adding your own

1. If your data is a standard public layout, write a `DatasetSpec` and use an
   existing `--source`.
2. Otherwise implement a `DatasetSource` (`sources/base.py`: `discover`,
   `load_image`, `load_mask`) and add a `--source` choice in `convert.py`.
3. Register the spec (built-in, or in `overlays/private/specs/` for private data).
4. `modelfactory dataset list` to confirm it resolves, then convert.
