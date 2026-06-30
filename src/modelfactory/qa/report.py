"""Cross-validation QA reports — self-contained HTML + CSV.

Pure rendering: takes the cv.json manifest (per-case) or the build_model_report
output (rollup) and returns a string. No torch, no GPU, no template files — the
HTML is a single self-contained document (inline CSS + inline SVG, system font
stack, no CDN/JS) that prints cleanly to PDF from the browser. CSV uses the
stdlib. Everything here unit-tests on the host.

Tier thresholds mirror the frontend MetricsBlock / lib/dice.ts (>=.8 ok,
>=.6 good, >=.4 warn, <.4 fail) so the report and the UI agree.
"""

from __future__ import annotations

import csv
import html
import io
from typing import Any

VERSION = "0.7.2"
FAIL_THRESHOLD = 0.4

# Precomputed so it can be interpolated without a backslash inside an f-string
# expression (forbidden before Python 3.12; ruff targets py310).
_STAR = '<span class="star">★</span>'

# Tier palette (matches the frontend tokens' resolved colours closely enough
# for a standalone document that can't read CSS vars).
_TIER_COLOR = {
    "ok": "#2f9e6b",
    "good": "#3b7fc4",
    "warn": "#d49a2b",
    "fail": "#d1495b",
    "none": "#9aa0a6",
}


def _tier(d: float | None) -> str:
    if d is None:
        return "none"
    if d >= 0.8:
        return "ok"
    if d >= 0.6:
        return "good"
    if d >= FAIL_THRESHOLD:
        return "warn"
    return "fail"


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _fmt(v: float | None, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _fmt_mm(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{round(v)} mm" if v >= 100 else f"{v:.1f} mm"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _bar(value: float | None, w: int = 160, h: int = 9) -> str:
    """Inline-SVG horizontal dice bar, tier-coloured."""
    if value is None:
        return '<span class="muted">—</span>'
    tier = _tier(value)
    fill = _clamp01(value) * w
    return (
        f'<svg class="bar" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<rect width="{w}" height="{h}" rx="3" fill="#e7e9ec"/>'
        f'<rect width="{fill:.1f}" height="{h}" rx="3" fill="{_TIER_COLOR[tier]}"/>'
        f"</svg>"
    )


def _range_bar(
    mn: float | None, mx: float | None, mean: float | None, oof: float | None,
    w: int = 180, h: int = 12,
) -> str:
    """Inline-SVG range bar: min–max span + mean tick + OOF diamond."""
    if mean is None:
        return '<span class="muted">—</span>'
    tier = _tier(mean)
    lo = _clamp01(mn if mn is not None else mean) * w
    hi = _clamp01(mx if mx is not None else mean) * w
    mid = _clamp01(mean) * w
    parts = [
        f'<svg class="range" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect y="{h/2-1:.0f}" width="{w}" height="2" rx="1" fill="#e7e9ec"/>',
        f'<rect x="{lo:.1f}" y="{h/2-3:.0f}" width="{max(0,hi-lo):.1f}" height="6" rx="3" '
        f'fill="{_TIER_COLOR[tier]}" fill-opacity="0.55"/>',
        f'<rect x="{mid-1:.1f}" y="1" width="2" height="{h-2}" fill="{_TIER_COLOR[tier]}"/>',
    ]
    if oof is not None:
        ox = _clamp01(oof) * w
        parts.append(
            f'<rect x="{ox-2.5:.1f}" y="{h/2-2.5:.0f}" width="5" height="5" '
            f'transform="rotate(45 {ox:.1f} {h/2:.0f})" fill="#2f9e6b" stroke="#fff"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _label_metrics(entry: dict) -> list[dict]:
    return entry.get("metrics") or []


def _failed_count(entry: dict) -> int:
    n = 0
    for m in _label_metrics(entry):
        d = m.get("dice")
        if (d is None and (m.get("n_voxels_gt") or 0) > 0) or (
            d is not None and d < FAIL_THRESHOLD
        ):
            n += 1
    return n


def _worst_label(entry: dict) -> tuple[str, float] | None:
    worst: tuple[str, float] | None = None
    for m in _label_metrics(entry):
        d = m.get("dice")
        if d is None:
            continue
        if worst is None or d < worst[1]:
            worst = (m.get("label_name", f"label_{m.get('label')}"), d)
    return worst


def _hd95_max(entry: dict) -> float | None:
    vals = [m.get("hd95_mm") for m in _label_metrics(entry) if m.get("hd95_mm") is not None]
    return max(vals) if vals else None


# ── shared HTML chrome ─────────────────────────────────────────────────────

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: #1d2125; background: #f6f7f9; margin: 0; padding: 24px;
  font-size: 13px; line-height: 1.45;
}
.wrap { max-width: 960px; margin: 0 auto; }
header.rpt { display: flex; justify-content: space-between; align-items: flex-start;
  gap: 16px; border-bottom: 2px solid #e7e9ec; padding-bottom: 12px; margin-bottom: 18px; }
h1 { font-size: 19px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.12em; color: #6b7178;
  margin: 22px 0 8px; }
.meta { color: #6b7178; font-size: 11.5px; }
.meta b { color: #1d2125; font-weight: 600; }
.pill { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 11px;
  font-weight: 600; }
.pill.oof { background: #e4f3ec; color: #1f7a4d; }
.pill.noof { background: #fbeede; color: #8a5a12; }
.section { background: #fff; border: 1px solid #e7e9ec; border-radius: 10px; padding: 14px 16px;
  margin-bottom: 14px; }
.hero { display: flex; align-items: baseline; gap: 14px; }
.hero .num { font-size: 40px; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
.hero .cap { color: #6b7178; font-size: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #eef0f2; }
th { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7178; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.oof td { background: #f1faf5; }
tr.oof td:first-child { box-shadow: inset 3px 0 0 #2f9e6b; }
tr.ensemble td { border-top: 2px solid #e7e9ec; font-style: italic; color: #444; }
.star { color: #2f9e6b; }
.muted { color: #9aa0a6; }
.tier-ok { color: #2f9e6b; } .tier-good { color: #3b7fc4; }
.tier-warn { color: #b9831f; } .tier-fail { color: #d1495b; }
.callout { background: #fff7f8; border: 1px solid #f2d6da; border-radius: 8px; padding: 10px 12px; }
.coverage { background: #eef3fb; border: 1px solid #d6e2f2; border-radius: 8px; padding: 8px 12px;
  font-size: 12px; color: #2b3a4a; margin-bottom: 14px; }
.group-honest { background: #f1faf5; } .group-cmp { background: #f7f8fa; }
footer.rpt { color: #9aa0a6; font-size: 10.5px; margin-top: 18px; border-top: 1px solid #e7e9ec; padding-top: 10px; }
.btn { display: inline-block; padding: 6px 12px; border: 1px solid #cdd2d7; border-radius: 6px;
  background: #fff; color: #1d2125; font-size: 12px; cursor: pointer; text-decoration: none; }
@page { size: A4; margin: 14mm; }
@media print {
  body { background: #fff; padding: 0; }
  .no-print { display: none !important; }
  .section, .coverage, .callout { break-inside: avoid; page-break-inside: avoid; }
  tr { break-inside: avoid; }
  h1, h2, .hero { break-after: avoid; }
  a[href]::after { content: ""; }
}
"""


def _doc(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
        f"<body><div class=\"wrap\">{body}</div></body></html>"
    )


def _print_button() -> str:
    return (
        '<a class="btn no-print" href="#" onclick="window.print();return false;">'
        "Print / Save as PDF</a>"
    )


# ── per-case report ─────────────────────────────────────────────────────────


def render_case_html(cv: dict) -> str:
    agg = cv.get("aggregate") or {}
    entries = cv.get("entries") or []
    folds = [e for e in entries if e.get("kind") == "fold"]
    ensemble = next((e for e in entries if e.get("kind") == "ensemble"), None)
    oof_fold = cv.get("oof_fold")
    headline = agg.get("headline_mean_fg_dice")
    is_oof = agg.get("headline_kind") == "oof" and oof_fold is not None

    oof_pill = (
        f'<span class="pill oof">Out-of-fold (unbiased): fold {oof_fold}</span>'
        if is_oof
        else '<span class="pill noof">No OOF — external case · every fold unbiased</span>'
        if cv.get("oof_reason") == "external"
        else '<span class="pill noof">No held-out fold available</span>'
    )

    stale = (
        '<div class="callout no-print" style="margin-bottom:14px">'
        "Ground truth changed since this run — metrics may be stale. Re-run cross-validation.</div>"
        if cv.get("stale")
        else ""
    )

    # honest hero
    hero = (
        f'<div class="section"><h2>Honest cross-validation score</h2>'
        f'<div class="hero"><div class="num tier-{_tier(headline)}">{_fmt(headline)}</div>'
        f'<div class="cap">{"out-of-fold · fold " + str(oof_fold) if is_oof else "cross-fold mean (no held-out fold)"}'
        f' · σ folds {_fmt(agg.get("cross_fold_std"))} · ensemble {_fmt(agg.get("ensemble_mean_fg_dice"))}</div>'
        f"</div></div>"
    )

    # per-fold table
    rows = []
    for e in folds:
        worst = _worst_label(e)
        cls = "oof" if e.get("is_oof") else ""
        rows.append(
            f'<tr class="{cls}"><td>fold {e.get("fold")}</td>'
            f'<td>{_STAR if e.get("is_oof") else ""}</td>'
            f'<td class="num tier-{_tier(e.get("mean_fg_dice"))}">{_fmt(e.get("mean_fg_dice"))}</td>'
            f'<td>{_bar(e.get("mean_fg_dice"))}</td>'
            f'<td class="num">{_failed_count(e)}</td>'
            f'<td>{_esc(worst[0]) + " " + _fmt(worst[1], 2) if worst else "—"}</td>'
            f'<td class="num">{_fmt_mm(_hd95_max(e))}</td></tr>'
        )
    if ensemble:
        worst = _worst_label(ensemble)
        rows.append(
            f'<tr class="ensemble"><td>ensemble</td><td></td>'
            f'<td class="num tier-{_tier(ensemble.get("mean_fg_dice"))}">{_fmt(ensemble.get("mean_fg_dice"))}</td>'
            f'<td>{_bar(ensemble.get("mean_fg_dice"))}</td>'
            f'<td class="num">{_failed_count(ensemble)}</td>'
            f'<td>{_esc(worst[0]) + " " + _fmt(worst[1], 2) if worst else "—"}</td>'
            f'<td class="num">{_fmt_mm(_hd95_max(ensemble))}</td></tr>'
        )
    fold_table = (
        '<div class="section"><h2>Per-fold comparison</h2><table>'
        "<thead><tr><th>fold</th><th>oof</th><th class=\"num\">mean dice</th><th></th>"
        "<th class=\"num\">failed</th><th>worst label</th><th class=\"num\">hd95 max</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )

    # inter-fold agreement
    per_label = sorted(
        agg.get("per_label") or [],
        key=lambda p: p["dice_mean"] if p.get("dice_mean") is not None else 2.0,
    )
    agr_rows = []
    for p in per_label:
        agr_rows.append(
            f"<tr><td>{_esc(p.get('label_name'))}</td>"
            f'<td class="num tier-{_tier(p.get("dice_mean"))}">{_fmt(p.get("dice_mean"), 2)}</td>'
            f'<td class="num">±{_fmt(p.get("dice_std"), 2)}</td>'
            f"<td>{_range_bar(p.get('dice_min'), p.get('dice_max'), p.get('dice_mean'), p.get('oof_dice'))}</td>"
            f'<td class="num">{_fmt(p.get("oof_dice"), 2)}</td></tr>'
        )
    agreement = (
        '<div class="section"><h2>Inter-fold agreement (per label)</h2>'
        '<table><thead><tr><th>label</th><th class="num">mean</th><th class="num">σ</th>'
        '<th>min–max · mean ▏ · ◆ oof</th><th class="num">oof dice</th></tr></thead>'
        f"<tbody>{''.join(agr_rows)}</tbody></table></div>"
        if per_label else ""
    )

    # worst labels callout (by OOF dice when present, else cross-fold mean)
    def _wl_key(p: dict) -> float:
        v = p.get("oof_dice")
        if v is None:
            v = p.get("dice_mean")
        return v if v is not None else 2.0

    worst5 = sorted(per_label, key=_wl_key)[:5]
    wl_items = "".join(
        f"<li><b>{_esc(p.get('label_name'))}</b> — "
        f"{'oof ' + _fmt(p.get('oof_dice'), 3) if p.get('oof_dice') is not None else 'mean ' + _fmt(p.get('dice_mean'), 3)}"
        f" (σ {_fmt(p.get('dice_std'), 2)})</li>"
        for p in worst5
    )
    callout = (
        f'<div class="section"><h2>Worst labels</h2><div class="callout"><ul style="margin:0;padding-left:18px">{wl_items}</ul></div></div>'
        if worst5 else ""
    )

    pred_ids = ", ".join(
        f"{('fold ' + str(e.get('fold'))) if e.get('kind') == 'fold' else 'ensemble'}={_esc(e.get('prediction_id') or '—')}"
        for e in entries
    )
    footer = (
        '<footer class="rpt">'
        f"HD95 policy: {_esc(cv.get('compute_hd95'))} · run updated {_esc(cv.get('updated_at'))} · "
        f"predictions: {pred_ids} · model-qa {VERSION}"
        "</footer>"
    )

    header = (
        '<header class="rpt"><div>'
        f"<h1>Cross-Validation QA — {_esc(cv.get('dataset_name'))}</h1>"
        f'<div class="meta">model <b>{_esc(cv.get("model_id"))}</b><br>'
        f'case <b>{_esc(cv.get("case_id"))}</b> · stem {_esc(cv.get("source_case_stem"))} · '
        f'GT {"original" if cv.get("gt_revision") in (None, 0) else "rev " + str(cv.get("gt_revision"))}</div>'
        f'<div style="margin-top:6px">{oof_pill}</div></div>'
        f"<div>{_print_button()}</div></header>"
    )

    return _doc(
        f"Cross-Validation QA — {cv.get('case_id')}",
        header + stale + hero + fold_table + agreement + callout + footer,
    )


def render_case_csv(cv: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "model_id", "dataset_name", "case_id", "source_case_stem", "region",
        "gt_revision", "oof_fold", "row_kind", "fold", "is_oof",
        "label", "label_name", "dice", "hd95_mm", "n_voxels_gt", "n_voxels_pred",
        "mean_fold_dice",
    ])

    def _c(v: Any) -> Any:
        return "" if v is None else v

    base = [
        cv.get("model_id"), cv.get("dataset_name"), cv.get("case_id"),
        cv.get("source_case_stem"), cv.get("region"),
        _c(cv.get("gt_revision")), _c(cv.get("oof_fold")),
    ]
    for e in cv.get("entries") or []:
        kind = e.get("kind")
        fold = e.get("fold")
        for m in e.get("metrics") or []:
            w.writerow([
                *base, kind, _c(fold), e.get("is_oof"),
                m.get("label"), m.get("label_name"),
                _c(m.get("dice")), _c(m.get("hd95_mm")),
                m.get("n_voxels_gt"), m.get("n_voxels_pred"),
                _c(e.get("mean_fg_dice")),
            ])
    return buf.getvalue()


# ── model-level rollup ──────────────────────────────────────────────────────


def render_rollup_html(report: dict) -> str:
    honest = report.get("honest_mean_fg_dice")
    coverage = (
        f'<div class="coverage">CV runs: <b>{report.get("n_cases_with_cv")}</b> / '
        f'{report.get("n_cases_total")} compatible cohort cases · with out-of-fold: '
        f'<b>{report.get("n_with_oof")}</b> · the honest score below averages each case\'s '
        f"out-of-fold fold.</div>"
    )
    hero = (
        '<div class="section"><h2>Honest cross-validation score</h2>'
        f'<div class="hero"><div class="num tier-{_tier(honest)}">{_fmt(honest)}</div>'
        f'<div class="cap">mean out-of-fold dice across {report.get("n_with_oof")} cases · '
        f'σ {_fmt(report.get("honest_std"))}</div></div></div>'
    )

    # per-fold (biased comparison)
    pf_rows = "".join(
        f'<tr><td>fold {f.get("fold")}</td>'
        f'<td class="num tier-{_tier(f.get("mean"))}">{_fmt(f.get("mean"))}</td>'
        f'<td>{_bar(f.get("mean"))}</td>'
        f'<td class="num">±{_fmt(f.get("std"))}</td>'
        f'<td class="num">{f.get("n_cases")}</td></tr>'
        for f in report.get("per_fold") or []
    )
    per_fold = (
        '<div class="section"><h2>Per-fold mean (biased — comparison only)</h2>'
        '<p class="meta">Each fold averaged over every case, mixing in-training and '
        'held-out cases; for the honest number use the score above.</p>'
        '<table><thead><tr><th>fold</th><th class="num">mean dice</th><th></th>'
        '<th class="num">±σ</th><th class="num">n cases</th></tr></thead>'
        f"<tbody>{pf_rows}</tbody></table></div>"
    )

    # per-label scoreboard
    pl_rows = "".join(
        f"<tr><td>{_esc(p.get('label_name'))}</td>"
        f'<td class="num">{p.get("n_cases")}</td>'
        f'<td class="num tier-{_tier(p.get("oof_mean"))}">{_fmt(p.get("oof_mean"), 3)}</td>'
        f'<td>{_bar(p.get("oof_mean"))}</td>'
        f'<td class="num">±{_fmt(p.get("oof_std"), 3)}</td>'
        f'<td class="num muted">{_fmt(p.get("fold_mean"), 3)}</td></tr>'
        for p in report.get("per_label") or []
    )
    per_label = (
        '<div class="section"><h2>Per-label out-of-fold dice (across cases)</h2>'
        '<table><thead><tr><th>label</th><th class="num">n</th>'
        '<th class="num group-honest">OOF mean</th><th></th><th class="num">±σ</th>'
        '<th class="num group-cmp">fold-mean (cmp)</th></tr></thead>'
        f"<tbody>{pl_rows}</tbody></table></div>"
        if report.get("per_label") else ""
    )

    def _case_rows(cases: list[dict]) -> str:
        out = []
        for c in cases:
            out.append(
                f"<tr><td>{_esc(c.get('case_id'))}</td>"
                f"<td>{_esc(c.get('source_case_stem'))}</td>"
                f'<td class="num">{"—" if c.get("oof_fold") is None else c.get("oof_fold")}</td>'
                f'<td class="num tier-{_tier(c.get("headline_mean_fg_dice"))}">{_fmt(c.get("headline_mean_fg_dice"))}</td></tr>'
            )
        return "".join(out)

    worst = report.get("worst_cases") or []
    best = report.get("best_cases") or []
    cases_tbl = (
        '<div class="section"><h2>Worst cases (by honest dice)</h2>'
        '<table><thead><tr><th>case</th><th>stem</th><th class="num">oof fold</th>'
        '<th class="num">honest dice</th></tr></thead>'
        f"<tbody>{_case_rows(worst)}</tbody></table>"
        '<h2 style="margin-top:16px">Best cases</h2>'
        '<table><thead><tr><th>case</th><th>stem</th><th class="num">oof fold</th>'
        '<th class="num">honest dice</th></tr></thead>'
        f"<tbody>{_case_rows(best)}</tbody></table></div>"
        if (worst or best) else ""
    )

    header = (
        '<header class="rpt"><div>'
        f"<h1>Cross-Validation Rollup — {_esc(report.get('dataset_name'))}</h1>"
        f'<div class="meta">model <b>{_esc(report.get("model_id"))}</b></div></div>'
        f"<div>{_print_button()}</div></header>"
    )
    footer = f'<footer class="rpt">model-qa {VERSION}</footer>'

    return _doc(
        f"Cross-Validation Rollup — {report.get('model_id')}",
        header + coverage + hero + per_fold + per_label + cases_tbl + footer,
    )


def render_rollup_csv(report: dict) -> str:
    buf = io.StringIO()
    buf.write(f"# model_id,{report.get('model_id')}\n")
    buf.write(f"# dataset_name,{report.get('dataset_name')}\n")
    buf.write(
        f"# n_cases_total,{report.get('n_cases_total')},n_cases_with_cv,"
        f"{report.get('n_cases_with_cv')},n_with_oof,{report.get('n_with_oof')}\n"
    )
    buf.write(
        f"# honest_mean_fg_dice,{report.get('honest_mean_fg_dice')},"
        f"honest_std,{report.get('honest_std')}\n"
    )
    w = csv.writer(buf)
    w.writerow([
        "section", "key", "label_or_fold", "n_cases",
        "mean", "std", "extra",
    ])
    for f in report.get("per_fold") or []:
        w.writerow(["per_fold", "fold", f.get("fold"), f.get("n_cases"),
                    "" if f.get("mean") is None else f.get("mean"),
                    "" if f.get("std") is None else f.get("std"), ""])
    for p in report.get("per_label") or []:
        w.writerow(["per_label_oof", p.get("label_name"), p.get("label"), p.get("n_cases"),
                    "" if p.get("oof_mean") is None else p.get("oof_mean"),
                    "" if p.get("oof_std") is None else p.get("oof_std"),
                    "" if p.get("fold_mean") is None else p.get("fold_mean")])
    for c in report.get("cases") or []:
        w.writerow(["case", c.get("case_id"), c.get("oof_fold"), "",
                    "" if c.get("headline_mean_fg_dice") is None else c.get("headline_mean_fg_dice"),
                    "", c.get("status")])
    return buf.getvalue()
