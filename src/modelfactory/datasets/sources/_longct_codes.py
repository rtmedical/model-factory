"""Class-string lookup for the Longitudinal-CT (Tübingen) source.

The on-disk masks tag each voxel with a per-patient `lesion_id` (integer 1..N).
The anatomy label comes from the per-patient CSV's `lesion_type` column, which
maps each `lesion_id` to a string like "Lymph node" / "Liver" / "Others".

This module is the single source of truth that maps the **canonical** anatomy
names we use in DatasetSpec to the **CSV strings** that appear in the published
data. Each canonical name maps to a *tuple* of CSV strings because some
canonical classes (e.g. "Other") merge multiple raw labels.

Audited 2026-05-20 against all 300 patient CSVs (4638 lesion rows total). See
`docs/longitudinal_ct_tubingen.md` for the per-class prevalence table that
drove the kept/dropped decisions.
"""
from __future__ import annotations


# canonical name → set of CSV lesion_type strings
CANONICAL_TO_CSV_TYPES: dict[str, frozenset[str]] = {
    "LymphNode":      frozenset({"Lymph node"}),
    "Lung":           frozenset({"Lung"}),
    "SoftTissueSkin": frozenset({"Soft tissue / Skin"}),
    "Liver":          frozenset({"Liver"}),
    "Other":          frozenset({"Others", "unclear"}),
    "Skeleton":       frozenset({"Skeleton"}),
    "Adrenal":        frozenset({"Adrenals"}),
    "Spleen":         frozenset({"Spleen"}),
    # The three below are too sparse (< 5 cases each) to train on. Listed for
    # completeness; not exposed in any current DatasetSpec.
    "Kidney":         frozenset({"Kidney"}),
    "Heart":          frozenset({"Heart"}),
    "CNS":            frozenset({"CNS"}),
}

# Sentinel canonical name that means "any non-zero voxel" — used by the binary
# D101 spec to fold all classes into a single foreground class.
WILDCARD_CANONICAL = "AnyMetastasis"


__all__ = ["CANONICAL_TO_CSV_TYPES", "WILDCARD_CANONICAL"]
