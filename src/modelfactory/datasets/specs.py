"""Catalogue of nnUNet training datasets and the structures they hold.

A `DatasetSpec` is the source-of-truth for one nnUNet `Dataset###_*`
folder: its numeric id, foreground-class ordering, descriptive metadata,
and per-source naming aliases. The orchestrator
(`modelfactory.datasets.convert`) takes one spec plus a
`DatasetSource` instance and writes the nnUNet raw layout.

To add a dataset:
  1. Append a `DatasetSpec(...)` entry to `SPECS` below.
  2. If the structure names already exist in your source's vocabulary,
     the spec needs no aliases. Otherwise add per-source aliases
     mapping the canonical name to the source's name.
  3. Run `modelfactory.datasets.convert --spec <key> --source <type>`.

Naming conventions:
  - canonical structure names are PascalCase + underscore for laterality
    (e.g. NVB_L, Femur_Head_R). They are *not* coupled to any one source.
  - dataset names omit source/quality tags (those land in MLflow tags) —
    so the folder is `Dataset023_PelvisMaleProstate`, not
    `Dataset023_SilverPelvisMaleProstate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any



@dataclass(frozen=True)
class StructureMapping:
    """Canonical structure name + per-source aliases.

    `canonical` is what the trained model and downstream tooling see.
    `aliases[source_type]` is what the on-disk source calls it.
    If a source_type is absent from `aliases`, the source uses
    `canonical` directly.
    """

    canonical: str
    aliases: dict[str, str] = field(default_factory=dict)

    def name_in(self, source_type: str) -> str:
        return self.aliases.get(source_type, self.canonical)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: int
    name: str
    description: str
    structures: tuple[StructureMapping, ...]
    source_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    # nnUNet dataset.json channel_names. None ⇒ derive from tags["modality"]
    # (CT default). MUST be non-"CT" for MR so nnUNet uses ZScoreNormalization
    # instead of CT HU-normalization. See convert._channel_names_for.
    channel_names: dict[str, str] | None = None
    # Partial-label single generalist. When True, convert.py keeps every case
    # that contours AT LEAST ONE declared structure (not all — the AND-filter is
    # relaxed), paints only the structures actually present, emits a region-based
    # dataset.json (one binary region per organ) + a per-case
    # `partial_label_annotations.json` sidecar, and the model must be trained with
    # nnUNetTrainerPartialLabelMLflow so each sample's loss is masked to its
    # annotated organ channels. This lets ONE model cover organs that never
    # co-occur in a single case (e.g. male+female pelvis OARs). See convert.py
    # and trainers/mlflow_trainer.py::nnUNetTrainerPartialLabelMLflow.
    partial_label: bool = False

    @property
    def folder(self) -> str:
        return f"Dataset{self.dataset_id:03d}_{self.name}"

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(s.canonical for s in self.structures)

    @property
    def label_map(self) -> dict[str, int]:
        m: dict[str, int] = {"background": 0}
        for i, s in enumerate(self.structures, start=1):
            m[s.canonical] = i
        return m


# ──────────────────────────────────────────────────────────────────────────
# Registry. Keep grouped by region; add new entries at the bottom of the
# relevant group. Dataset ids are global — never reuse a number.
# ──────────────────────────────────────────────────────────────────────────

SPECS: dict[str, DatasetSpec] = {}


def _register(spec: DatasetSpec, key: str) -> None:
    if key in SPECS:
        raise ValueError(f"duplicate spec key: {key}")
    for other in SPECS.values():
        if other.dataset_id == spec.dataset_id:
            raise ValueError(
                f"dataset_id {spec.dataset_id} clashes between "
                f"{other.name!r} and {spec.name!r}"
            )
    SPECS[key] = spec



# ──────────────────────────────────────────────────────────────────────────
# Public example specs (built on public datasets). Copy one and edit for your
# own data; register private cohorts in overlays/private/specs/ instead.
# ──────────────────────────────────────────────────────────────────────────

_register(
    DatasetSpec(
        dataset_id=900,
        name="ExampleMSDHippocampus",
        description=(
            "EXAMPLE — Medical Segmentation Decathlon Task04 Hippocampus (MRI). "
            "Demonstrates a basic two-class spec with the default trainer."
        ),
        structures=(
            StructureMapping(canonical="Anterior"),
            StructureMapping(canonical="Posterior"),
        ),
        source_constraints={"msd": {"task_root": "/in/Task04_Hippocampus"}},
        tags={"region": "brain", "modality": "MR", "dataset_license": "CC-BY-SA-4.0"},
        channel_names={"0": "MRI"},
    ),
    key="example_msd_hippocampus",
)

_register(
    DatasetSpec(
        dataset_id=901,
        name="ExamplePDDCAHeadNeck",
        description="EXAMPLE — PDDCA head & neck OARs (NRRD). A multi-organ CT spec.",
        structures=(
            StructureMapping(canonical="BrainStem"),
            StructureMapping(canonical="Mandible"),
            StructureMapping(canonical="Parotid_L"),
            StructureMapping(canonical="Parotid_R"),
        ),
        source_constraints={"pddca": {"pddca_root": "/in/PDDCA"}},
        tags={"region": "head_neck", "modality": "CT", "dataset_license": "see PDDCA terms"},
    ),
    key="example_pddca_hn",
)

_register(
    DatasetSpec(
        dataset_id=902,
        name="ExampleLUNA16Nodules",
        description=(
            "EXAMPLE — LUNA16 lung nodules. A tiny/sparse-foreground case: train "
            "with nnUNetTrainerSmallStructuresMLflow (see docs/training.md)."
        ),
        structures=(StructureMapping(canonical="Nodule"),),
        source_constraints={"luna16": {"luna_root": "/in/LUNA16"}},
        tags={"region": "thorax", "modality": "CT", "dataset_license": "CC-BY-4.0"},
    ),
    key="example_luna16_nodules",
)

__all__ = ["DatasetSpec", "StructureMapping", "SPECS"]


# ──────────────────────────────────────────────────────────────────────────
# Private overlay discovery
# ──────────────────────────────────────────────────────────────────────────
# Site-specific / proprietary cohorts do NOT live in this public file. They live
# in a private overlay that registers additional DatasetSpecs at import time,
# exactly like the entries above. An overlay module just does:
#
#     from modelfactory.datasets.specs import _register, DatasetSpec, StructureMapping
#     _register(DatasetSpec(dataset_id=900, name="MyCohort", ...), key="my_cohort")
#
# Overlay location: $MFACTORY_SPECS_OVERLAY (a directory of *.py modules), else
# <repo>/overlays/private/specs/. Absent overlay → no-op. See overlays/README.md.


# Overlay modules already imported, so repeated calls don't double-register.
_LOADED_OVERLAY_FILES: set = set()


def _load_overlay_specs() -> None:
    import importlib.util
    import os
    from pathlib import Path

    dirs: list[Path] = []
    env = os.environ.get("MFACTORY_SPECS_OVERLAY")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(Path(__file__).resolve().parents[3] / "overlays" / "private" / "specs")

    seen: set[Path] = set()
    for d in dirs:
        d = d.resolve()
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_") or f.resolve() in _LOADED_OVERLAY_FILES:
                continue
            _LOADED_OVERLAY_FILES.add(f.resolve())
            mod_spec = importlib.util.spec_from_file_location(f"modelfactory._overlay_specs.{f.stem}", f)
            if mod_spec and mod_spec.loader:
                module = importlib.util.module_from_spec(mod_spec)
                mod_spec.loader.exec_module(module)


_load_overlay_specs()
