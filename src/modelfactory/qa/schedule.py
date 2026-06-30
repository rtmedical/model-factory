"""SQLite-backed persistence for the *future-trainings pipeline*.

A planned training is a (dataset, fold, trainer, plans) that we *intend* to
submit but haven't yet. The live week-Gantt on the QA home dashboard renders
only folds that are physically training (filesystem heuristic over
``results/``); this store is the missing other half — the queue of what's
coming next — so the calendar can show planned bars alongside the live ones.

Design mirrors ``qa/verdicts.py`` 1:1: one table, one file, one writer (the
qa-viewer pod), on the shared NFS mount so the queue survives pod restarts.
Schema grows via additive ``ALTER TABLE`` in ``_ensure_schema``.

This module stores *intent*; the wall-clock projection (when each queued fold
is likely to start/finish) is a pure function, :func:`project_schedule`, fed
the live-training finish times by the API layer — the store itself never
reaches into the cluster.
"""

from __future__ import annotations

import datetime as dt
import heapq
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Lifecycle of a queued training. `submitted` is set by reconcile() once the
# real fold appears live/done in results/ (so the calendar never double-draws
# it); `cancelled` is a soft delete kept for audit.
PLANNED_STATUSES = ("planned", "submitted", "cancelled")

# Per-plan fold-0 wall-time priors (hours), used when a queued item has no
# measured duration yet. ResEnc-L ~3 days; the HighRes small-structures plan
# is heavier per epoch, so budget more. Refined per-dataset by the API layer
# from any already-completed fold of the same dataset.
_DURATION_HOURS_HIGHRES = 96.0
_DURATION_HOURS_DEFAULT = 72.0


def default_duration_hours(plans: str) -> float:
    """Fold wall-time prior keyed off the plans identifier."""
    return _DURATION_HOURS_HIGHRES if "HighRes" in plans else _DURATION_HOURS_DEFAULT


@dataclass
class PlannedTraining:
    id: str
    dataset_key: str          # SPECS key, e.g. pelvis_clinical_neural
    dataset_name: str         # Dataset127_... — so the calendar's displayDataset reuses unchanged
    fold: int
    trainer: str
    plans: str
    priority: int             # higher = earlier
    status: str               # one of PLANNED_STATUSES
    est_duration_hours: float | None
    submitted_by: str
    notes: str
    created_at: str           # ISO-8601 UTC
    # ── projected (NOT persisted) — filled by project_schedule() ──
    scheduled_start: str | None = field(default=None)  # ISO-8601 UTC
    est_finish: str | None = field(default=None)        # ISO-8601 UTC
    eta_seconds: float | None = field(default=None)


def _key(dataset_key: str, fold: int) -> str:
    return f"{dataset_key}::{fold}"


class ScheduleStore:
    """Thread-safe wrapper around the planned_trainings table."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            self._ensure_schema(conn)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS planned_trainings (
                id                 TEXT PRIMARY KEY,
                dataset_key        TEXT NOT NULL,
                dataset_name       TEXT NOT NULL,
                fold               INTEGER NOT NULL,
                trainer            TEXT NOT NULL,
                plans              TEXT NOT NULL,
                priority           INTEGER NOT NULL DEFAULT 0,
                status             TEXT NOT NULL DEFAULT 'planned'
                                     CHECK (status IN ('planned','submitted','cancelled')),
                est_duration_hours REAL,
                submitted_by       TEXT NOT NULL DEFAULT '',
                notes              TEXT NOT NULL DEFAULT '',
                created_at         TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS planned_status_idx ON planned_trainings(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS planned_prio_idx "
            "ON planned_trainings(priority DESC, created_at ASC)"
        )
        # One open queue entry per (dataset, fold): re-adding the same fold
        # updates priority/notes rather than duplicating a bar on the calendar.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS planned_dataset_fold_idx "
            "ON planned_trainings(dataset_key, fold)"
        )

    def add(
        self,
        *,
        dataset_key: str,
        dataset_name: str,
        fold: int,
        trainer: str,
        plans: str,
        priority: int = 0,
        est_duration_hours: float | None = None,
        submitted_by: str = "",
        notes: str = "",
    ) -> PlannedTraining:
        """Enqueue a fold (idempotent on (dataset_key, fold) — upserts an open row)."""
        created = dt.datetime.now(dt.timezone.utc).isoformat()
        new_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planned_trainings
                    (id, dataset_key, dataset_name, fold, trainer, plans, priority,
                     status, est_duration_hours, submitted_by, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?)
                ON CONFLICT(dataset_key, fold) DO UPDATE SET
                    dataset_name       = excluded.dataset_name,
                    trainer            = excluded.trainer,
                    plans              = excluded.plans,
                    priority           = excluded.priority,
                    est_duration_hours = excluded.est_duration_hours,
                    submitted_by       = excluded.submitted_by,
                    notes              = excluded.notes,
                    -- re-open a previously submitted/cancelled fold when re-queued
                    status             = 'planned'
                """,
                (new_id, dataset_key, dataset_name, fold, trainer, plans, priority,
                 est_duration_hours, submitted_by, notes, created),
            )
            row = conn.execute(
                "SELECT * FROM planned_trainings WHERE dataset_key = ? AND fold = ?",
                (dataset_key, fold),
            ).fetchone()
        return _row_to_planned(row)

    def list_all(self, *, status: str | None = "planned") -> list[PlannedTraining]:
        """Queue rows, highest priority first. status=None returns every row."""
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM planned_trainings "
                    "ORDER BY priority DESC, created_at ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM planned_trainings WHERE status = ? "
                    "ORDER BY priority DESC, created_at ASC",
                    (status,),
                ).fetchall()
        return [_row_to_planned(r) for r in rows]

    def get(self, planned_id: str) -> PlannedTraining | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM planned_trainings WHERE id = ?", (planned_id,)
            ).fetchone()
        return _row_to_planned(row) if row else None

    def update(
        self,
        planned_id: str,
        *,
        priority: int | None = None,
        notes: str | None = None,
        status: str | None = None,
        est_duration_hours: float | None = None,
    ) -> PlannedTraining | None:
        sets: list[str] = []
        vals: list[object] = []
        if priority is not None:
            sets.append("priority = ?"); vals.append(priority)
        if notes is not None:
            sets.append("notes = ?"); vals.append(notes)
        if status is not None:
            if status not in PLANNED_STATUSES:
                raise ValueError(f"bad status: {status}")
            sets.append("status = ?"); vals.append(status)
        if est_duration_hours is not None:
            sets.append("est_duration_hours = ?"); vals.append(est_duration_hours)
        if not sets:
            return self.get(planned_id)
        vals.append(planned_id)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE planned_trainings SET {', '.join(sets)} WHERE id = ?", vals
            )
        return self.get(planned_id)

    def delete(self, planned_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM planned_trainings WHERE id = ?", (planned_id,)
            )
        return (cur.rowcount or 0) > 0

    def reconcile(self, live_keys: set[str]) -> int:
        """Flip queued folds that have since gone live/done to `submitted`.

        `live_keys` is the set of ``<id>::<fold>`` strings currently present
        (training or done) in results/, where ``<id>`` is the dataset_name
        (what results/ is keyed by) — though dataset_key is also accepted, so
        callers can pass either form. Keeps the calendar from drawing a
        planned bar on top of the real one. Returns rows changed.
        """
        rows = self.list_all(status="planned")
        to_close = [
            r.id for r in rows
            if _key(r.dataset_key, r.fold) in live_keys
            or _key(r.dataset_name, r.fold) in live_keys
        ]
        if not to_close:
            return 0
        with self._lock, self._connect() as conn:
            conn.executemany(
                "UPDATE planned_trainings SET status = 'submitted' WHERE id = ?",
                [(i,) for i in to_close],
            )
        return len(to_close)


def _row_to_planned(r: sqlite3.Row) -> PlannedTraining:
    return PlannedTraining(
        id=r["id"],
        dataset_key=r["dataset_key"],
        dataset_name=r["dataset_name"],
        fold=int(r["fold"]),
        trainer=r["trainer"],
        plans=r["plans"],
        priority=int(r["priority"] or 0),
        status=r["status"],
        est_duration_hours=r["est_duration_hours"],
        submitted_by=r["submitted_by"] or "",
        notes=r["notes"] or "",
        created_at=r["created_at"],
    )


# ── ETA projection ─────────────────────────────────────────────────────────


def _ms_to_iso(ms: float) -> str:
    return dt.datetime.fromtimestamp(ms / 1000.0, dt.timezone.utc).isoformat()


def project_schedule(
    planned: list[PlannedTraining],
    *,
    running_finish_ms: list[float],
    slots: int,
    now_ms: float,
) -> list[PlannedTraining]:
    """Greedy slot simulation → fill scheduled_start/est_finish/eta_seconds.

    Models `slots` parallel training slots. The currently-training folds
    occupy the soonest-freeing slots until their live `est_finish`; the rest
    are free now. Each queued item (already ordered by priority, then age)
    takes the next slot to open, runs for its duration prior, and frees that
    slot again. Pure + deterministic given `now_ms` — no wall-clock reads —
    so it's unit-testable and resume-safe. This is a *visualization estimate*:
    the qa-viewer can't see live MIG/Kueue state, so `slots` is configured.
    """
    slots = max(1, slots)
    # Seed each slot's next-free time: busy slots free at their running finish
    # (never before now), idle slots free now. Cap to the soonest `slots`
    # free-times if more folds report training than we have slots.
    free_times = sorted(max(float(f), now_ms) for f in running_finish_ms)
    while len(free_times) < slots:
        free_times.append(now_ms)
    free_times = free_times[:slots]
    heapq.heapify(free_times)

    out: list[PlannedTraining] = []
    for p in planned:
        slot_free = heapq.heappop(free_times)
        start = max(slot_free, now_ms)
        dur_ms = (p.est_duration_hours or default_duration_hours(p.plans)) * 3_600_000.0
        finish = start + dur_ms
        heapq.heappush(free_times, finish)
        p.scheduled_start = _ms_to_iso(start)
        p.est_finish = _ms_to_iso(finish)
        p.eta_seconds = max(0.0, (finish - now_ms) / 1000.0)
        out.append(p)
    return out
