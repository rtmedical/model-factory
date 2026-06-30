"""Render the factory dashboard from `metrics.jsonl` files.

The renderer is intentionally MLflow-free: it reads the JSONL sidecars
written by :class:`modelfactory.trainers.mlflow_trainer.nnUNetTrainerMLflow`
directly off the shared NFS root. This means it works even when the
MLflow service is down or the cluster DNS is being weird (see the
`cluster_dns_fqdn_hijack` memory).

Schema reminder (one line per record):
  - `val`   rows: ts, phase, epoch, val_loss, mean_fg_dice, ema_fg_dice,
                  per_class_dice {canonical → float|NaN}
  - `train` rows: ts, phase, epoch, train_loss, lr, gpu_mem_mb, epoch_time_s
  - `event` rows: ts, event=run_start, run_id, dataset, ...

Usage:
    python -m modelfactory.dashboard.render \
        --output docs/factory_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2

from modelfactory.datasets.specs import SPECS, DatasetSpec

# ──────────────────────────────────────────────────────────────────────────
# What goes on the dashboard. Order is left→right, top→bottom in the grid.
# ──────────────────────────────────────────────────────────────────────────

DASHBOARD_KEYS: tuple[str, ...] = (
    # Brain-MR fanout (8 datasets, ResEncUNetXL, fold 0)
    "brain_mr_core_oar",            # 045
    "brain_mr_posterior_fossa",     # 056
    "brain_mr_ventricular_system",  # 057
    "brain_mr_basal_ganglia",       # 058
    "brain_mr_diencephalon",        # 059
    "brain_mr_medial_temporal",     # 060
    "brain_mr_cortical_lobes_4pair",  # 061
    "brain_mr_limbic_cortical",     # 062
    # SegRap CT specialists (6 datasets, started in two waves)
    "segrap_optic_pathway_ct",      # 083
    "segrap_eyes_ct",               # 084
    "segrap_mid_ear_bony_ct",       # 086
    "segrap_glands_ct",             # 087
    "segrap_aerodigestive_ct",      # 090
    "segrap_mandible_tmj_ct",       # 091
    # Body CT pack (added 2026-05-16: pelvis prostate + MSD pancreas + LUNA16)
    "pelvis_male_prostate",         # 023
    "pancreas_msd_tumor",           # 032
    "luna16_nodules",               # 036
    # H&N + pelvis specialists (added 2026-05-17/18, all on L plans)
    "hn_pddca_glands",              # 013
    "pelvis_seminal_ves",           # 051
    "optic_nerves_hn_ct",           # 054
    # RT-OAR specialists not covered by TotalSegmentator (added 2026-05-18)
    "pelvis_penile_bulb",           # 050
    "segrap_larynx_ct",             # 089
    # Brain-MR generalist (added 2026-05-19): 34-class union of D056-D062
    "brain_mr_fullbrain_generalist",  # 063
    # SegRap CT specialists wave 2 (added 2026-05-19): RT-OAR-focused complements to TS
    "segrap_inner_ear_ct",          # 085
    "segrap_medial_temporal_ct",    # 092
    "segrap_brainstem_spine_ct",    # 093
    # Larynx subparts companion to D089 (added 2026-05-19): Glottic + Supraglot
    "segrap_larynx_subparts_ct",    # 095
    # Melanoma whole-body lesion campaign (added 2026-05-20, Longitudinal-CT v2)
    "melanoma_mets_multi",          # 100  generalist multi-class
    "melanoma_any_binary",          # 101  generalist binary
    "melanoma_lymph_node",          # 102  lymph A/B exp arm
    "melanoma_lymph_node_default",  # 106  lymph A/B baseline arm
)

# Region & modality, derived from the dataset.tags when present and overridden
# here for the SegRap pack (whose tags currently say region=head_neck — true,
# but the dashboard groups by anatomical adjacency, putting eyes/optic with
# the brain pack).
REGION_OVERRIDES = {
    "segrap_optic_pathway_ct": "head_neck",
    "segrap_eyes_ct": "head_neck",
    "segrap_mid_ear_bony_ct": "head_neck",
    "segrap_glands_ct": "head_neck",
    "segrap_aerodigestive_ct": "head_neck",
    "segrap_mandible_tmj_ct": "head_neck",
    "pelvis_male_prostate": "pelvis",
    "pancreas_msd_tumor": "abdomen",
    "luna16_nodules": "thorax",
    "hn_pddca_glands": "head_neck",
    "pelvis_seminal_ves": "pelvis",
    "optic_nerves_hn_ct": "head_neck",
    "pelvis_penile_bulb": "pelvis",
    "segrap_larynx_ct": "head_neck",
    "segrap_inner_ear_ct": "head_neck",
    "segrap_medial_temporal_ct": "head_neck",
    "segrap_brainstem_spine_ct": "head_neck",
    "segrap_larynx_subparts_ct": "head_neck",
    "melanoma_mets_multi": "whole_body",
    "melanoma_any_binary": "whole_body",
    "melanoma_lymph_node": "whole_body",
    "melanoma_lymph_node_default": "whole_body",
}

# nnUNet default epoch budget per fold. The trainer can override this with
# `num_epochs`, but the standard ResEnc trainers used here all keep 1000.
EXPECTED_EPOCHS_PER_FOLD = 1000

# Stop-candidate scoring:
#   plateau_strength * remaining_frac
# where plateau_strength is in [0,1] (1 = dice not moving), remaining_frac is
# in [0,1] (1 = 0 epochs elapsed). High scores = "stuck and lots left to go"
# = best candidate to stop and reclaim its MIG slice for another campaign.
STOP_SLOPE_PLATEAU_THRESH = 0.0005   # dice/epoch — at-or-below = fully plateaued
STOP_SLOPE_HEALTHY_THRESH = 0.0050   # dice/epoch — at-or-above = healthy improvement
STOP_RECENT_VAL_WINDOW = 20          # last N val rows to compute the slope

RESULTS_ROOT_DEFAULT = Path("/data/model-factory-nfs/results")
DEFAULT_TRAINER_DIR_GLOB = "nnUNetTrainer*MLflow__nnUNetResEncUNet*Plans*__3d_fullres"
DEFAULT_FOLD = 0
LIVE_WINDOW_SECONDS = 10 * 60  # mtime within 10 min → "training"
SPARK_MAX_POINTS = 60


# ──────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class StructureRow:
    canonical: str
    dice: float | None  # None if NaN / missing
    is_nan: bool = False


@dataclass
class SparkPoint:
    epoch: int
    dice: float


@dataclass
class DatasetCardView:
    spec_key: str
    dataset_id: int
    folder: str
    name: str
    description: str
    region: str
    modality: str
    quality_tier: str  # "gold" | "silver" | "bronze"
    classes: tuple[StructureRow, ...]
    # Live numbers from the latest val row (None if no val rows yet)
    last_epoch: int | None
    mean_fg_dice: float | None
    ema_fg_dice: float | None
    val_loss: float | None
    # Latest train row
    train_loss: float | None
    gpu_mem_mb: float | None
    epoch_time_s: float | None
    # Sparkline + freshness
    spark_points: tuple[SparkPoint, ...]
    metrics_mtime: float | None  # epoch seconds; None if file missing
    first_event_ts: float | None  # epoch seconds of the first row in metrics.jsonl
    has_final_checkpoint: bool  # checkpoint_final.pth present → cleanup done
    status: str  # "training" | "complete" | "stalled" | "not_started"
    # Stop-candidate ranking (None for non-live cards)
    dice_slope: float | None       # dice change per epoch over last N val rows
    stop_score: float | None       # plateau_strength * remaining_frac, in [0,1]
    stop_rationale: str | None     # one-line human-readable explanation
    wall_hours: float | None       # hours since first metrics row written

    @property
    def status_label(self) -> str:
        return {
            "training": "live",
            "complete": "fold 0 done",
            "stalled": "paused",
            "not_started": "queued",
        }.get(self.status, self.status)


@dataclass
class StructureTableRow:
    canonical: str
    dataset_id: int
    dataset_name: str
    dice: float | None
    is_nan: bool
    epoch: int | None
    val_loss: float | None


@dataclass
class DashboardContext:
    rendered_at_utc: str
    snapshot_iso: str
    node: str
    cluster_blurb: str
    logo_b64: str
    kpis: list[dict[str, str]]
    cards_live: list[DatasetCardView]
    cards_complete: list[DatasetCardView]
    cards_paused: list[DatasetCardView]
    structure_table: list[StructureTableRow]
    stop_candidates: list[DatasetCardView] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def cards(self) -> list[DatasetCardView]:
        """All cards (live first, then complete, then paused) — used by aggregations."""
        return [*self.cards_live, *self.cards_complete, *self.cards_paused]


# ──────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────


def _find_metrics_jsonl(
    results_root: Path, folder: str, fold: int = DEFAULT_FOLD
) -> Path | None:
    """Find the freshest `metrics.jsonl` under `<results_root>/<folder>/<trainer_dir>/fold_<n>/`."""
    base = results_root / folder
    if not base.is_dir():
        return None
    candidates = sorted(
        base.glob(f"{DEFAULT_TRAINER_DIR_GLOB}/fold_{fold}/metrics.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _is_nan(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.lower() in ("nan", "inf", "-inf"):
        return True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def _decimate(points: list[SparkPoint], cap: int) -> list[SparkPoint]:
    if len(points) <= cap:
        return points
    step = len(points) / cap
    out: list[SparkPoint] = []
    i = 0.0
    while int(i) < len(points) and len(out) < cap:
        out.append(points[int(i)])
        i += step
    if out[-1].epoch != points[-1].epoch:
        out.append(points[-1])
    return out


def _build_card_view(spec_key: str, spec: DatasetSpec, results_root: Path) -> DatasetCardView:
    jsonl = _find_metrics_jsonl(results_root, spec.folder)
    rows: list[dict[str, Any]] = []
    mtime: float | None = None
    first_ts: float | None = None
    if jsonl is not None and jsonl.is_file():
        rows = _load_rows(jsonl)
        try:
            mtime = jsonl.stat().st_mtime
        except OSError:
            mtime = None
        for r in rows:
            ts = _safe_float(r.get("ts"))
            if ts is not None:
                first_ts = ts
                break

    val_rows = [r for r in rows if r.get("phase") == "val"]
    train_rows = [r for r in rows if r.get("phase") == "train"]

    last_val = val_rows[-1] if val_rows else None
    last_train = train_rows[-1] if train_rows else None

    # Per-class dice from the last val row (NaN-safe).
    classes: list[StructureRow] = []
    pcd = (last_val or {}).get("per_class_dice", {}) or {}
    for sm in spec.structures:
        canonical = sm.canonical
        raw = pcd.get(canonical)
        if _is_nan(raw):
            classes.append(StructureRow(canonical=canonical, dice=None, is_nan=True))
        else:
            classes.append(StructureRow(canonical=canonical, dice=float(raw), is_nan=False))

    # Sparkline points.
    spark_raw: list[SparkPoint] = []
    for r in val_rows:
        d = _safe_float(r.get("mean_fg_dice"))
        ep = r.get("epoch")
        if d is None or not isinstance(ep, (int, float)):
            continue
        spark_raw.append(SparkPoint(epoch=int(ep), dice=float(d)))
    spark = tuple(_decimate(spark_raw, SPARK_MAX_POINTS))

    # Detect cleanup-phase completion: nnUNet writes `checkpoint_final.pth`
    # at the end of `run_training()` after the last epoch + post-train
    # validation. Its presence means fold 0 is fully wrapped (model + val
    # predictions on disk, ready for ensembling).
    has_final_checkpoint = False
    if jsonl is not None and jsonl.parent.is_dir():
        has_final_checkpoint = (jsonl.parent / "checkpoint_final.pth").is_file()

    # Status determination.
    now = time.time()
    if mtime is None or last_val is None:
        status = "not_started"
    elif now - mtime <= LIVE_WINDOW_SECONDS:
        status = "training"
    elif has_final_checkpoint:
        status = "complete"
    else:
        status = "stalled"

    # Stop-candidate scoring: only meaningful when training is live and has
    # enough val points to estimate a slope. Otherwise leave as None so the
    # template can hide the card from the stop panel.
    dice_slope: float | None = None
    stop_score: float | None = None
    stop_rationale: str | None = None
    wall_hours: float | None = None
    if first_ts is not None and mtime is not None:
        wall_hours = max(0.0, (mtime - first_ts) / 3600.0)
    if status == "training" and len(val_rows) >= 2:
        # Compute dice slope over the last N val points (or all if fewer).
        recent = val_rows[-STOP_RECENT_VAL_WINDOW:]
        ds = [(_safe_float(r.get("epoch")), _safe_float(r.get("mean_fg_dice"))) for r in recent]
        ds = [(e, d) for e, d in ds if e is not None and d is not None]
        if len(ds) >= 2:
            e0, d0 = ds[0]
            e1, d1 = ds[-1]
            de = e1 - e0
            dice_slope = (d1 - d0) / de if de > 0 else 0.0
            # Plateau strength: 1.0 if slope ≤ plateau thresh, 0.0 if ≥ healthy thresh
            span = STOP_SLOPE_HEALTHY_THRESH - STOP_SLOPE_PLATEAU_THRESH
            plateau_strength = max(
                0.0,
                min(1.0, (STOP_SLOPE_HEALTHY_THRESH - max(0.0, dice_slope)) / span),
            )
            last_ep = int(_safe_float(last_val.get("epoch")) or 0)
            remaining_frac = max(0.0, 1.0 - last_ep / EXPECTED_EPOCHS_PER_FOLD)
            stop_score = plateau_strength * remaining_frac
            # Rationale string: pick the dominant factor.
            if remaining_frac < 0.1:
                stop_rationale = f"only {int((1 - remaining_frac) * 100)}% remaining — let it finish"
            elif plateau_strength >= 0.8 and remaining_frac >= 0.5:
                stop_rationale = (
                    f"plateaued (slope ≈ {dice_slope:+.4f}/ep over last {len(ds)} val rows) "
                    f"with {int(remaining_frac * 100)}% of epochs still to burn"
                )
            elif plateau_strength >= 0.5:
                stop_rationale = f"slowing down (slope ≈ {dice_slope:+.4f}/ep) · {int(remaining_frac * 100)}% remaining"
            else:
                stop_rationale = f"still improving (slope ≈ {dice_slope:+.4f}/ep) — keep running"

    # Region: prefer override, else spec.tags["region"], else "unknown".
    region = REGION_OVERRIDES.get(spec_key) or spec.tags.get("region", "unknown")
    modality = spec.tags.get("modality") or (
        "MR-T1" if "Brain_MR" in spec.name else ("CT" if "_CT" in spec.name else "MR/CT")
    )
    quality_tier = spec.tags.get("label_quality", "silver")

    return DatasetCardView(
        spec_key=spec_key,
        dataset_id=spec.dataset_id,
        folder=spec.folder,
        name=spec.name,
        description=spec.description,
        region=region,
        modality=modality,
        quality_tier=quality_tier,
        classes=tuple(classes),
        last_epoch=(last_val or {}).get("epoch"),
        mean_fg_dice=_safe_float((last_val or {}).get("mean_fg_dice")),
        ema_fg_dice=_safe_float((last_val or {}).get("ema_fg_dice")),
        val_loss=_safe_float((last_val or {}).get("val_loss")),
        train_loss=_safe_float((last_train or {}).get("train_loss")),
        gpu_mem_mb=_safe_float((last_train or {}).get("gpu_mem_mb")),
        epoch_time_s=_safe_float((last_train or {}).get("epoch_time_s")),
        spark_points=spark,
        metrics_mtime=mtime,
        first_event_ts=first_ts,
        has_final_checkpoint=has_final_checkpoint,
        status=status,
        dice_slope=dice_slope,
        stop_score=stop_score,
        stop_rationale=stop_rationale,
        wall_hours=wall_hours,
    )


# ──────────────────────────────────────────────────────────────────────────
# Aggregations for hero / KPI / table
# ──────────────────────────────────────────────────────────────────────────


def _build_kpis(cards: list[DatasetCardView]) -> list[dict[str, str]]:
    active = [c for c in cards if c.status == "training"]
    with_val = [c for c in cards if c.mean_fg_dice is not None]
    best = max((c.mean_fg_dice for c in with_val), default=None)
    epoch_times = [c.epoch_time_s for c in cards if c.epoch_time_s is not None]
    above_50 = [c for c in cards if (c.last_epoch or 0) >= 50]
    max_epoch = max((c.last_epoch or 0 for c in cards), default=0)

    def _fmt_dice(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "—"

    def _fmt_time(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f} s"
        m = secs / 60
        return f"{m:.1f} min"

    return [
        {"label": "Active trainings", "value": str(len(active)), "hint": f"{len(cards)} datasets tracked"},
        {"label": "MIG slices live", "value": "10 × 3g.40gb", "hint": "GPUs 3–7, ResEncUNetXL"},
        {"label": "Best fg dice", "value": _fmt_dice(best), "hint": "across all live runs"},
        {"label": "Furthest epoch", "value": str(max_epoch), "hint": f"{len(above_50)}/{len(cards)} past ep 50"},
        {
            "label": "Median epoch time",
            "value": _fmt_time(statistics.median(epoch_times)) if epoch_times else "—",
            "hint": "fold 0, 3d_fullres",
        },
        {"label": "Ray workers", "value": "10", "hint": "factory-ray, KubeRay 1.6.1"},
    ]


def _build_structure_table(cards: list[DatasetCardView]) -> list[StructureTableRow]:
    rows: list[StructureTableRow] = []
    for c in cards:
        for sr in c.classes:
            rows.append(
                StructureTableRow(
                    canonical=sr.canonical,
                    dataset_id=c.dataset_id,
                    dataset_name=c.name,
                    dice=sr.dice,
                    is_nan=sr.is_nan,
                    epoch=c.last_epoch,
                    val_loss=c.val_loss,
                )
            )
    # Sort: real values descending, NaN at the bottom.
    rows.sort(key=lambda r: (-1 if r.dice is None else 0, -(r.dice or 0.0)))
    return rows


def _read_logo_b64() -> str:
    p = Path(__file__).parent / "assets" / "rtm_logo.b64"
    try:
        return p.read_text(encoding="ascii").strip()
    except OSError:
        return ""


# ──────────────────────────────────────────────────────────────────────────
# Sparkline path generation (template helper)
# ──────────────────────────────────────────────────────────────────────────


def _spark_path(points: list[SparkPoint], width: int = 160, height: int = 36, pad: int = 3) -> dict[str, Any]:
    """Return {'d': SVG path d, 'last_x': float, 'last_y': float, 'last_label': str}.

    Used as a Jinja2 global so the template can call it inline.
    """
    if not points:
        return {"d": "", "last_x": 0, "last_y": 0, "last_label": "—"}
    if len(points) == 1:
        x = width / 2
        y = height / 2
        return {
            "d": f"M{x:.1f},{y:.1f}",
            "last_x": x,
            "last_y": y,
            "last_label": f"ep {points[0].epoch} · {points[0].dice:.3f}",
        }
    dices = [p.dice for p in points]
    epochs = [p.epoch for p in points]
    dmin, dmax = min(dices), max(dices)
    span = dmax - dmin if dmax > dmin else 1e-6
    emin, emax = min(epochs), max(epochs)
    espan = emax - emin if emax > emin else 1
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad

    def _x(ep: int) -> float:
        return pad + (ep - emin) / espan * usable_w

    def _y(d: float) -> float:
        # Higher dice = visually higher = smaller y.
        return pad + (1 - (d - dmin) / span) * usable_h

    parts: list[str] = []
    for i, p in enumerate(points):
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd}{_x(p.epoch):.1f},{_y(p.dice):.1f}")
    last = points[-1]
    return {
        "d": " ".join(parts),
        "last_x": _x(last.epoch),
        "last_y": _y(last.dice),
        "last_label": f"ep {last.epoch} · {last.dice:.3f}",
    }


# ──────────────────────────────────────────────────────────────────────────
# Top-level render entry point
# ──────────────────────────────────────────────────────────────────────────


def build_context(results_root: Path = RESULTS_ROOT_DEFAULT) -> DashboardContext:
    cards: list[DatasetCardView] = []
    notes: list[str] = []
    for key in DASHBOARD_KEYS:
        spec = SPECS.get(key)
        if spec is None:
            notes.append(f"spec key '{key}' not in registry — skipped")
            continue
        try:
            cards.append(_build_card_view(key, spec, results_root))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"card '{key}' failed to render: {exc!r}")

    # Split into three groups: live (training right now), complete (fold 0
    # cleanup done — checkpoint_final.pth on disk), paused (stalled or never
    # started). Within each group, sort by mean_fg_dice descending so the
    # best results surface at the top.
    def _sort_key(c: DatasetCardView) -> tuple[int, float]:
        d = c.mean_fg_dice if c.mean_fg_dice is not None else -1.0
        return (0 if c.mean_fg_dice is not None else 1, -d)

    cards_live = sorted([c for c in cards if c.status == "training"], key=_sort_key)
    cards_complete = sorted([c for c in cards if c.status == "complete"], key=_sort_key)
    cards_paused = sorted(
        [c for c in cards if c.status not in ("training", "complete")],
        key=_sort_key,
    )

    # Stop-candidates panel: live cards with a computed stop_score, ranked
    # high → low. Hide cards whose rationale says "still improving" so the
    # panel surfaces only campaigns worth stopping.
    stop_candidates = [
        c for c in cards_live
        if c.stop_score is not None and (c.stop_rationale or "").find("still improving") < 0
    ]
    stop_candidates.sort(key=lambda c: -(c.stop_score or 0.0))

    now = datetime.now(tz=timezone.utc)
    snapshot_iso = now.strftime("%Y-%m-%d %H:%M UTC")
    return DashboardContext(
        rendered_at_utc=now.isoformat(timespec="seconds"),
        snapshot_iso=snapshot_iso,
        node=os.environ.get("MFACTORY_DASHBOARD_NODE", ""),
        cluster_blurb=os.environ.get(
            "MFACTORY_DASHBOARD_BLURB",
            "KubeRay · nnU-Net v2 · MLflow · Kueue",
        ),
        logo_b64=_read_logo_b64(),
        kpis=_build_kpis(cards),
        cards_live=cards_live,
        cards_complete=cards_complete,
        cards_paused=cards_paused,
        structure_table=_build_structure_table(cards),
        stop_candidates=stop_candidates,
        notes=notes,
    )


def render_html(ctx: DashboardContext) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=jinja2.select_autoescape(("html", "xml")),
        trim_blocks=False,
        lstrip_blocks=False,
    )
    env.globals["spark_path"] = _spark_path
    env.filters["dice"] = lambda v: ("—" if v is None else f"{v:.3f}")
    env.filters["loss"] = lambda v: ("—" if v is None else f"{v:.3f}")
    env.filters["epoch_time"] = lambda v: ("—" if v is None else f"{v:.0f}s")
    env.filters["pct"] = lambda v: ("0" if v is None else f"{max(0.0, min(1.0, v)) * 100:.1f}")
    env.filters["mem"] = lambda v: ("—" if v is None else f"{v / 1024:.1f} GiB")
    env.filters["hours"] = lambda v: ("—" if v is None else (f"{v:.0f} h" if v >= 1 else f"{v * 60:.0f} min"))
    env.filters["slope"] = lambda v: ("—" if v is None else f"{v:+.4f}/ep")
    env.filters["score"] = lambda v: ("—" if v is None else f"{v:.2f}")
    template = env.get_template("factory_dashboard.html.j2")
    return template.render(ctx=ctx)


def render_to_file(output: Path, results_root: Path = RESULTS_ROOT_DEFAULT) -> Path:
    ctx = build_context(results_root)
    html = render_html(ctx)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


# ──────────────────────────────────────────────────────────────────────────
# CLI (also reachable via `modelfactory dashboard render`)
# ──────────────────────────────────────────────────────────────────────────


def _argparse_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render the model-factory dashboard.")
    p.add_argument("--output", type=Path, default=Path("docs/factory_dashboard.html"))
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT_DEFAULT)
    args = p.parse_args(argv)
    out = render_to_file(args.output, args.results_root)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(_argparse_main())
