"""Surface the lowest-dice validation cases for a finished training run.

After nnUNetv2 finishes a fold, it writes `validation/summary.json` containing
per-case dice scores. This module re-reads that file, ranks cases by dice (or
loss), and writes a parquet artifact alongside the run so we can mine failure
modes across many runs in pandas.

Designed to be invoked from `nnUNetTrainerMLflow.on_train_end` via an opt-in
flag, OR offline against an existing MLflow run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_summary(summary_path: Path) -> pd.DataFrame:
    """Parse nnUNetv2's validation/summary.json into one row per case."""
    raw = json.loads(summary_path.read_text())
    rows = []
    for case, payload in raw.get("metric_per_case", {}).items():
        # Each `payload` has structure { "1": { "Dice": 0.83, "FP": ... }, "2": {...} }
        for label, metrics in payload.items():
            rows.append({
                "case": case,
                "label": label,
                "dice": metrics.get("Dice"),
                "fp": metrics.get("FP"),
                "fn": metrics.get("FN"),
                "tp": metrics.get("TP"),
                "tn": metrics.get("TN"),
            })
    return pd.DataFrame(rows)


def write_failure_parquet(
    nnunet_results_dir: Path,
    out_parquet: Path,
    decile: float = 0.10,
) -> pd.DataFrame:
    """Build a worst-decile-cases parquet from one fold's results dir."""
    summary = nnunet_results_dir / "validation" / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    df = _read_summary(summary)
    # Aggregate to one row per case by averaging dice across labels (simple, defensible)
    per_case = df.groupby("case", as_index=False).agg(mean_dice=("dice", "mean"))
    cutoff = per_case["mean_dice"].quantile(decile)
    failures = per_case[per_case["mean_dice"] <= cutoff].sort_values("mean_dice")
    failures.to_parquet(out_parquet, index=False)
    return failures
