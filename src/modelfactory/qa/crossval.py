"""Cross-validation QA: out-of-fold routing + per-fold aggregation.

Pure, torch-free, NFS-optional helpers so they unit-test on the host without a
GPU, nnUNet, or Redis. The FastAPI orchestration (queue, predictor cache,
status writes) lives in `modelfactory.qa.api`; everything statistical and
provenance-related lives here.

The central idea: nnUNetv2 trains 5-fold cross-validation, so every training
case sits in exactly one fold's *validation* split — that fold never saw the
case during training, making its prediction **out-of-fold (OOF)** and unbiased.
`splits_final.json` records the val membership; a QA cohort case carries its
`source_case_stem` (the nnUNet case id), so the OOF fold is the fold `k` whose
`splits[k]["val"]` contains that stem. See CLAUDE.md and the QA-cohort design.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

logger = logging.getLogger("qa-crossval")


# ── out-of-fold routing ───────────────────────────────────────────────────


@dataclass(frozen=True)
class OofResolution:
    """How a cohort case maps to its unbiased fold.

    `oof_fold` is the held-out fold index when `resolvable` is True, else None.
    `reason` is a stable machine code surfaced to the UI/report so a missing
    OOF is explained rather than silently mislabelled as fold 0:
      - "oof"               — stem is in exactly one available fold's val split
      - "multiple_val"      — stem in >1 val split (pathological); first taken
      - "external"          — stem in no split at all (donated/external case)
      - "in_train_only"     — stem in splits but in no val split (pathological)
      - "oof_fold_untrained"— stem's val fold has no trained checkpoint yet
      - "no_splits"         — splits_final.json missing/unreadable/malformed
    """

    oof_fold: int | None
    resolvable: bool
    reason: str


def load_splits(splits_path: str | Path) -> list[dict] | None:
    """Read splits_final.json → list of {"train":[...], "val":[...]} or None."""
    try:
        raw = json.loads(Path(splits_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, list):
        return None
    return raw


def oof_fold_for_case(
    splits_path: str | Path,
    source_case_stem: str,
    available_folds: Sequence[int],
) -> OofResolution:
    """Resolve the out-of-fold index for `source_case_stem` (see OofResolution).

    Defensive (review R1): we do not just trust the first val match — we first
    confirm the stem is a training case at all (present in some fold's
    train∪val), distinguishing a genuine external/donated case from an
    in-training one, and we intersect the matched fold with `available_folds`
    so a not-yet-trained fold reports "no OOF" rather than a bogus index.
    """
    splits = load_splits(splits_path)
    if not splits:
        return OofResolution(None, False, "no_splits")

    val_sets = [set(s.get("val", []) or []) for s in splits]
    train_sets = [set(s.get("train", []) or []) for s in splits]
    known: set[str] = set().union(*val_sets, *train_sets) if splits else set()

    if source_case_stem not in known:
        return OofResolution(None, False, "external")

    in_val = [i for i, v in enumerate(val_sets) if source_case_stem in v]
    if not in_val:
        return OofResolution(None, False, "in_train_only")

    candidate = in_val[0]
    reason = "oof"
    if len(in_val) > 1:
        logger.warning(
            "case stem %s found in %d val splits (folds %s); using first",
            source_case_stem, len(in_val), in_val,
        )
        reason = "multiple_val"
    if candidate not in set(available_folds):
        return OofResolution(None, False, "oof_fold_untrained")
    return OofResolution(candidate, True, reason)


# ── per-case aggregation across folds ──────────────────────────────────────


def _pstd(values: Sequence[float]) -> float | None:
    """Population std; 0.0 for a single sample, None for empty."""
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def aggregate_cv(entries: Iterable[dict], oof_fold: int | None) -> dict:
    """Aggregate per-fold results into the comparison/report payload.

    `entries` are the per-fold + ensemble result dicts the orchestrator builds:
        {kind: "fold"|"ensemble", fold: int|None, is_oof: bool,
         mean_fg_dice: float|None, metrics: list[LabelMetricOut-dict]|None, ...}

    The per-label spread (mean/std/min/max) is computed over the SINGLE folds
    only (the ensemble is reference, not a sample). Everything here is derived
    from numbers already in hand — zero extra compute. `headline_mean_fg_dice`
    is the honest OOF-fold score, or the cross-fold mean when there is no OOF
    (external case / untrained fold).
    """
    entries = list(entries)
    fold_entries = [
        e for e in entries if e.get("kind") == "fold" and e.get("metrics")
    ]
    ensemble = next((e for e in entries if e.get("kind") == "ensemble"), None)

    by_label: dict[int, dict[str, Any]] = {}
    for e in fold_entries:
        is_oof = bool(e.get("is_oof"))
        for m in e["metrics"]:
            lab = int(m["label"])
            slot = by_label.setdefault(
                lab,
                {
                    "label": lab,
                    "label_name": m.get("label_name", f"label_{lab}"),
                    "_dice": [],
                    "_hd95": [],
                    "oof_dice": None,
                    "oof_hd95_mm": None,
                },
            )
            if m.get("dice") is not None:
                slot["_dice"].append(float(m["dice"]))
            if m.get("hd95_mm") is not None:
                slot["_hd95"].append(float(m["hd95_mm"]))
            if is_oof:
                slot["oof_dice"] = m.get("dice")
                slot["oof_hd95_mm"] = m.get("hd95_mm")

    per_label: list[dict] = []
    for lab in sorted(by_label):
        s = by_label[lab]
        dice, hd = s["_dice"], s["_hd95"]
        per_label.append({
            "label": lab,
            "label_name": s["label_name"],
            "dice_mean": float(mean(dice)) if dice else None,
            "dice_std": _pstd(dice),
            "dice_min": float(min(dice)) if dice else None,
            "dice_max": float(max(dice)) if dice else None,
            "hd95_mean_mm": float(mean(hd)) if hd else None,
            "hd95_std_mm": _pstd(hd),
            "oof_dice": s["oof_dice"],
            "oof_hd95_mm": s["oof_hd95_mm"],
        })

    fold_mean_fg_dice = {
        int(e["fold"]): e.get("mean_fg_dice") for e in fold_entries
    }
    fold_means = [v for v in fold_mean_fg_dice.values() if v is not None]
    cross_fold_mean = float(mean(fold_means)) if fold_means else None
    cross_fold_std = _pstd(fold_means)
    ensemble_mean = ensemble.get("mean_fg_dice") if ensemble else None

    if oof_fold is not None and fold_mean_fg_dice.get(oof_fold) is not None:
        headline = fold_mean_fg_dice.get(oof_fold)
        headline_kind = "oof"
    else:
        headline = cross_fold_mean
        headline_kind = "cross_fold_mean"

    return {
        "per_label": per_label,
        "fold_mean_fg_dice": fold_mean_fg_dice,
        "cross_fold_mean": cross_fold_mean,
        "cross_fold_std": cross_fold_std,
        "ensemble_mean_fg_dice": ensemble_mean,
        "headline_mean_fg_dice": headline,
        "headline_kind": headline_kind,
    }


# ── model-level rollup across cohort cases ─────────────────────────────────


def build_model_report(
    model_id: str,
    dataset_name: str,
    cv_runs: Iterable[dict],
    compatible_case_ids: Sequence[str],
) -> dict:
    """Aggregate completed per-case CV runs into a model-level rollup.

    Read-only over whatever cv.json runs exist (the API never triggers runs for
    a rollup). The HONEST headline averages each case's out-of-fold score (the
    aggregate's `headline_mean_fg_dice`); the per-fold columns are the biased
    all-folds comparison and are labelled as such in the report. Cohort cases
    with no CV run are listed `pending` so coverage gaps are visible.
    """
    runs = [r for r in cv_runs if r.get("status") == "done" and r.get("aggregate")]

    # Keep the most recent run per case.
    by_case: dict[str, dict] = {}
    for r in runs:
        cid = r.get("case_id")
        if not cid:
            continue
        prev = by_case.get(cid)
        if prev is None or r.get("updated_at", "") > prev.get("updated_at", ""):
            by_case[cid] = r

    cases: list[dict] = []
    headlines: list[float] = []
    fold_vals: dict[int, list[float]] = defaultdict(list)
    label_slots: dict[int, dict[str, Any]] = {}
    n_with_oof = 0

    for cid, r in by_case.items():
        agg = r["aggregate"]
        headline = agg.get("headline_mean_fg_dice")
        oof_fold = r.get("oof_fold")
        if r.get("oof_resolvable") and oof_fold is not None:
            n_with_oof += 1
        cases.append({
            "case_id": cid,
            "source_case_stem": r.get("source_case_stem"),
            "oof_fold": oof_fold,
            "oof_resolvable": bool(r.get("oof_resolvable")),
            "headline_mean_fg_dice": headline,
            "headline_kind": agg.get("headline_kind"),
            "ensemble_mean_fg_dice": agg.get("ensemble_mean_fg_dice"),
            "cv_run_id": r.get("cv_run_id"),
            "status": "done",
        })
        if headline is not None:
            headlines.append(headline)
        for fk, v in (agg.get("fold_mean_fg_dice") or {}).items():
            if v is not None:
                fold_vals[int(fk)].append(v)
        for pl in agg.get("per_label") or []:
            lab = int(pl["label"])
            slot = label_slots.setdefault(
                lab,
                {"label": lab, "label_name": pl.get("label_name", f"label_{lab}"),
                 "oof": [], "fold_mean": []},
            )
            if pl.get("oof_dice") is not None:
                slot["oof"].append(float(pl["oof_dice"]))
            if pl.get("dice_mean") is not None:
                slot["fold_mean"].append(float(pl["dice_mean"]))

    done_ids = set(by_case)
    for cid in compatible_case_ids:
        if cid not in done_ids:
            cases.append({
                "case_id": cid, "source_case_stem": None, "oof_fold": None,
                "oof_resolvable": False, "headline_mean_fg_dice": None,
                "headline_kind": None, "ensemble_mean_fg_dice": None,
                "cv_run_id": None, "status": "pending",
            })

    per_fold = [
        {
            "fold": k,
            "mean": float(mean(fold_vals[k])) if fold_vals[k] else None,
            "std": _pstd(fold_vals[k]),
            "n_cases": len(fold_vals[k]),
        }
        for k in sorted(fold_vals)
    ]

    per_label = []
    for lab in sorted(label_slots):
        s = label_slots[lab]
        oof, fm = s["oof"], s["fold_mean"]
        per_label.append({
            "label": lab, "label_name": s["label_name"],
            "oof_mean": float(mean(oof)) if oof else None, "oof_std": _pstd(oof),
            "fold_mean": float(mean(fm)) if fm else None, "n_cases": len(oof),
        })
    per_label.sort(key=lambda p: p["oof_mean"] if p["oof_mean"] is not None else 2.0)

    done_cases = [c for c in cases if c["status"] == "done"]
    ranked = sorted(
        [c for c in done_cases if c["headline_mean_fg_dice"] is not None],
        key=lambda c: c["headline_mean_fg_dice"],
    )

    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        "n_cases_total": len(compatible_case_ids) if compatible_case_ids else len(done_cases),
        "n_cases_with_cv": len(done_cases),
        "n_with_oof": n_with_oof,
        "honest_mean_fg_dice": float(mean(headlines)) if headlines else None,
        "honest_std": _pstd(headlines),
        "per_fold": per_fold,
        "per_label": per_label,
        "best_cases": list(reversed(ranked))[:5],
        "worst_cases": ranked[:10],
        "cases": sorted(cases, key=lambda c: c["case_id"]),
    }
