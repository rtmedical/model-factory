# Private overlays

The public `model-factory` tree ships the **framework** (conversion, training,
QA, infra) plus a small set of **public-dataset example specs**. Your own data —
proprietary cohorts, internal directory layouts, site-specific dataset specs,
clinical aliases, brand assets, and your real `cluster.yaml` — lives in a
**private overlay** that is git-ignored and never committed to the public repo.

```
overlays/
  README.md            (this file — public)
  private/             (git-ignored — your stuff)
    specs/             *.py modules registering your DatasetSpecs
    sources/           your cohort-specific DatasetSource adapters (optional)
    docs/              your internal runbooks
    cluster.yaml       your real cluster spec (or keep at repo root, also ignored)
```

## Adding private dataset specs

Drop one or more `*.py` files in `overlays/private/specs/`. Each registers
`DatasetSpec`s exactly like the built-in ones, at import time:

```python
# overlays/private/specs/my_cohort.py
from modelfactory.datasets.specs import _register, DatasetSpec, StructureMapping

_register(
    DatasetSpec(
        dataset_id=900,
        name="MyPelvisCohort",
        description="Internal pelvic OARs from our clinical RTSTRUCT archive.",
        structures=(
            StructureMapping(canonical="Bladder"),
            StructureMapping(canonical="Rectum", aliases={"rtstruct": "Rectum_O"}),
        ),
        source_constraints={"rtstruct": {"rtstruct_root": "/in/mycohort"}},
        tags={"region": "pelvis", "modality": "CT", "dataset_license": "internal"},
    ),
    key="my_pelvis_cohort",
)
```

`modelfactory.datasets.specs` discovers these automatically. The overlay
directory is `$MFACTORY_SPECS_OVERLAY` if set, otherwise
`overlays/private/specs/`. Verify they loaded:

```bash
modelfactory dataset list        # your specs appear alongside the examples
```

Files whose names start with `_` are skipped (use them for shared helpers).

## Adding private source adapters

If your data isn't in a format a built-in adapter handles, implement the
`DatasetSource` interface (see `src/modelfactory/datasets/sources/base.py`) in
`overlays/private/sources/` and import it from your spec module.

## Keep the public tree clean

CI runs a "forbidden-strings" check (see `.github/workflows/ci.yml`) that fails
if site-specific hostnames, node names, internal `/data/*` paths, or proprietary
product names appear in the committed (public) tree. Keep all of that in
`overlays/private/` (git-ignored). `scripts/extract_overlay.sh` helps move
existing proprietary content out of the public tree into an overlay.
