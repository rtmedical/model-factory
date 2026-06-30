"""SQLite-backed persistence for reviewer corrections to cohort GT labels.

Reviewer edits land at
``/factory/qa-cohort/<region>/<case>/label_corrected_v{N}.nii.gz`` —
the original ``label_groundtruth.nii.gz`` is **never overwritten** so the
dataset-as-released stays auditable. This module owns the index that
points at the active revision per case.

Concurrency: SQLite WAL handles concurrent readers. A partial unique
index pins "at most one active revision per case" at the DB layer, so
the activate operation is a single transaction that flips one row to
``superseded`` and one row to ``active`` atomically.

This is intentionally a sibling of ``verdicts.py`` (and shares its
SQLite file): the two write paths are orthogonal and we don't want
the verdicts lock serializing reviewer paint strokes.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

GT_STATUS_ACTIVE = "active"
GT_STATUS_SUPERSEDED = "superseded"
GT_STATUS_VALUES = (GT_STATUS_ACTIVE, GT_STATUS_SUPERSEDED)


@dataclass
class GroundTruthRevision:
    id: int
    region: str
    case_id: str  # cohort-local "<region>/<case>" full id
    revision: int
    path: str  # cohort-relative path
    base_prediction_id: str | None
    reviewer: str
    notes: str
    status: str
    created_at: str  # ISO-8601 UTC


class GroundTruthStore:
    """Thread-safe wrapper around the ``gt_corrections`` table.

    Reuses the verdicts SQLite file so a single shared NFS DB persists
    both stores. Schema is created on first construction; subsequent
    runs are no-ops via ``IF NOT EXISTS``.
    """

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
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS gt_corrections (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                region              TEXT NOT NULL,
                case_id             TEXT NOT NULL,
                revision            INTEGER NOT NULL,
                path                TEXT NOT NULL,
                base_prediction_id  TEXT,
                reviewer            TEXT NOT NULL DEFAULT '',
                notes               TEXT NOT NULL DEFAULT '',
                status              TEXT NOT NULL
                                    CHECK (status IN ('{GT_STATUS_ACTIVE}','{GT_STATUS_SUPERSEDED}')),
                created_at          TEXT NOT NULL,
                UNIQUE (region, case_id, revision)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS gt_corr_case_idx "
            "ON gt_corrections(region, case_id)"
        )
        # At most one active revision per case — enforced at the DB layer
        # so activate() can rely on the invariant instead of locking.
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS gt_corr_active_idx "
            f"ON gt_corrections(region, case_id) WHERE status='{GT_STATUS_ACTIVE}'"
        )

    # ── reads ────────────────────────────────────────────────────────

    def list_for_case(self, region: str, case_id: str) -> list[GroundTruthRevision]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM gt_corrections "
                "WHERE region=? AND case_id=? "
                "ORDER BY revision DESC",
                (region, case_id),
            ).fetchall()
        return [_row_to_revision(r) for r in rows]

    def get_active(self, region: str, case_id: str) -> GroundTruthRevision | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM gt_corrections "
                f"WHERE region=? AND case_id=? AND status='{GT_STATUS_ACTIVE}' "
                f"LIMIT 1",
                (region, case_id),
            ).fetchone()
        return _row_to_revision(row) if row else None

    def get_by_id(self, revision_id: int) -> GroundTruthRevision | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM gt_corrections WHERE id=?", (revision_id,),
            ).fetchone()
        return _row_to_revision(row) if row else None

    def next_revision_number(self, region: str, case_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(revision), 0) AS last "
                "FROM gt_corrections WHERE region=? AND case_id=?",
                (region, case_id),
            ).fetchone()
        return int((row["last"] if row else 0) or 0) + 1

    # ── writes ───────────────────────────────────────────────────────

    def record_active(
        self,
        *,
        region: str,
        case_id: str,
        revision: int,
        path: str,
        base_prediction_id: str | None,
        reviewer: str,
        notes: str,
    ) -> GroundTruthRevision:
        """Insert a new revision and mark it active in one transaction.

        The previous active row (if any) flips to ``superseded``. The
        partial unique index guarantees no two rows can be active for the
        same case — if the index would be violated, ``INSERT`` raises and
        the whole transaction rolls back.
        """
        created = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    f"UPDATE gt_corrections SET status='{GT_STATUS_SUPERSEDED}' "
                    f"WHERE region=? AND case_id=? AND status='{GT_STATUS_ACTIVE}'",
                    (region, case_id),
                )
                cur = conn.execute(
                    f"INSERT INTO gt_corrections "
                    f"(region, case_id, revision, path, base_prediction_id, "
                    f" reviewer, notes, status, created_at) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, '{GT_STATUS_ACTIVE}', ?)",
                    (region, case_id, revision, path, base_prediction_id,
                     reviewer, notes, created),
                )
                row_id = int(cur.lastrowid or 0)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return GroundTruthRevision(
            id=row_id,
            region=region,
            case_id=case_id,
            revision=revision,
            path=path,
            base_prediction_id=base_prediction_id,
            reviewer=reviewer,
            notes=notes,
            status=GT_STATUS_ACTIVE,
            created_at=created,
        )

    def activate(self, revision_id: int) -> GroundTruthRevision:
        """Promote an older revision to active. Idempotent if already active."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM gt_corrections WHERE id=?", (revision_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"no revision {revision_id}")
            if row["status"] == GT_STATUS_ACTIVE:
                return _row_to_revision(row)

            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    f"UPDATE gt_corrections SET status='{GT_STATUS_SUPERSEDED}' "
                    f"WHERE region=? AND case_id=? AND status='{GT_STATUS_ACTIVE}'",
                    (row["region"], row["case_id"]),
                )
                conn.execute(
                    f"UPDATE gt_corrections SET status='{GT_STATUS_ACTIVE}' "
                    f"WHERE id=?",
                    (revision_id,),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

            new_row = conn.execute(
                "SELECT * FROM gt_corrections WHERE id=?", (revision_id,),
            ).fetchone()
        return _row_to_revision(new_row)


def _row_to_revision(row: sqlite3.Row | None) -> GroundTruthRevision:
    assert row is not None
    return GroundTruthRevision(
        id=int(row["id"]),
        region=row["region"],
        case_id=row["case_id"],
        revision=int(row["revision"]),
        path=row["path"],
        base_prediction_id=row["base_prediction_id"],
        reviewer=row["reviewer"] or "",
        notes=row["notes"] or "",
        status=row["status"],
        created_at=row["created_at"],
    )
