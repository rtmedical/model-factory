"""SQLite-backed persistence for QA verdicts.

One table, one file, one writer (the qa-viewer pod). The DB lives on the
shared NFS mount so verdicts survive pod restarts.

Schema is intentionally tiny: PK + timestamps + the four fields a clinician
actually writes during review (verdict, notes, reviewer) + the lineage of
the prediction they reviewed (model_id, case_id, prediction_id, fold_choice,
dice). New columns can be added cheaply via `ALTER TABLE` in
`_ensure_schema`.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

VERDICT_VALUES = ("accept", "reject", "needs_review")

# Structured reject-reason taxonomy. Lets rejections roll up into an
# actionable "what to fix in the next model" signal instead of freeform
# prose. Keep in sync with REJECT_REASONS in web/lib/api.ts (same
# contract as PALETTE ↔ themes.py).
REJECT_REASONS = (
    "over_segmentation",
    "misses_small_structures",
    "wrong_anatomy",
    "boundary_errors",
    "false_positives",
    "other",
)

# Model-level approval is DERIVED from the per-case verdict tallies — it is
# not a separately-stored sign-off. One rule, used by `summary()` and mirrored
# in the frontend only for display.
ApprovalStatus = str  # "approved" | "rejected" | "pending"


def approval_status_for(
    accept: int, reject: int, needs_review: int, total: int
) -> ApprovalStatus:
    """Roll case verdicts up to a single model decision.

    all reviewed cases accepted -> approved; any reject -> rejected;
    nothing reviewed / unresolved needs-review / mixed -> pending.
    """
    if total <= 0:
        return "pending"
    if reject > 0:
        return "rejected"
    if needs_review > 0:
        return "pending"
    if accept > 0:
        return "approved"
    return "pending"


@dataclass
class Verdict:
    id: int
    prediction_id: str
    model_id: str
    case_id: str
    verdict: str
    notes: str
    reviewer: str
    fold_choice: str
    mean_dice: float | None
    created_at: str  # ISO-8601 UTC
    # Only meaningful when verdict == "reject"; one of REJECT_REASONS or "".
    reject_reason: str = ""


@dataclass
class VerdictSummary:
    model_id: str
    accept: int
    reject: int
    needs_review: int
    total: int
    last_at: str | None
    last_verdict: str | None
    # Derived model decision (see approval_status_for) + a per-reason
    # breakdown of the rejects, so the catalog/sign-off can show *why*.
    approval_status: ApprovalStatus = "pending"
    reject_reasons: dict[str, int] | None = None


class VerdictStore:
    """Thread-safe wrapper around a single SQLite file."""

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
        # WAL is friendlier on NFS than the default rollback journal.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verdicts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                model_id      TEXT NOT NULL,
                case_id       TEXT NOT NULL,
                verdict       TEXT NOT NULL CHECK (verdict IN ('accept','reject','needs_review')),
                notes         TEXT NOT NULL DEFAULT '',
                reviewer      TEXT NOT NULL DEFAULT '',
                fold_choice   TEXT NOT NULL DEFAULT 'best',
                mean_dice     REAL,
                created_at    TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS verdicts_model_idx ON verdicts(model_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS verdicts_case_idx ON verdicts(case_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS verdicts_created_idx ON verdicts(created_at DESC)"
        )
        # Cheap forward-migration: add the reject_reason column on existing
        # databases that predate the structured-reason taxonomy.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(verdicts)")}
        if "reject_reason" not in cols:
            conn.execute(
                "ALTER TABLE verdicts ADD COLUMN reject_reason TEXT NOT NULL DEFAULT ''"
            )

    def record(
        self,
        *,
        prediction_id: str,
        model_id: str,
        case_id: str,
        verdict: str,
        notes: str = "",
        reviewer: str = "",
        fold_choice: str = "best",
        mean_dice: float | None = None,
        reject_reason: str = "",
    ) -> Verdict:
        if verdict not in VERDICT_VALUES:
            raise ValueError(f"bad verdict: {verdict}")
        # Only retain a reason on a reject; ignore it otherwise so accept/
        # needs_review rows stay clean.
        reason = reject_reason if verdict == "reject" else ""
        created = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO verdicts (prediction_id, model_id, case_id, verdict,
                                      notes, reviewer, fold_choice, mean_dice,
                                      reject_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (prediction_id, model_id, case_id, verdict,
                 notes, reviewer, fold_choice, mean_dice, reason, created),
            )
            row_id = cur.lastrowid
        return Verdict(
            id=row_id or 0,
            prediction_id=prediction_id,
            model_id=model_id,
            case_id=case_id,
            verdict=verdict,
            notes=notes,
            reviewer=reviewer,
            fold_choice=fold_choice,
            mean_dice=mean_dice,
            created_at=created,
            reject_reason=reason,
        )

    def list_for_model(self, model_id: str, limit: int = 100) -> list[Verdict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM verdicts WHERE model_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (model_id, limit),
            ).fetchall()
        return [_row_to_verdict(r) for r in rows]

    def list_for_case(self, model_id: str, case_id: str) -> list[Verdict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM verdicts WHERE model_id = ? AND case_id = ?
                ORDER BY created_at DESC
                """,
                (model_id, case_id),
            ).fetchall()
        return [_row_to_verdict(r) for r in rows]

    def summary(self) -> list[VerdictSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    model_id,
                    SUM(verdict = 'accept')        AS accept,
                    SUM(verdict = 'reject')        AS reject,
                    SUM(verdict = 'needs_review')  AS needs_review,
                    COUNT(*)                        AS total,
                    MAX(created_at)                 AS last_at
                FROM verdicts
                GROUP BY model_id
                ORDER BY last_at DESC
                """
            ).fetchall()
            # Need the last verdict per model — one extra correlated lookup
            # is fine at this table size.
            last_map: dict[str, str] = {}
            for r in rows:
                lv = conn.execute(
                    "SELECT verdict FROM verdicts WHERE model_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (r["model_id"],),
                ).fetchone()
                if lv:
                    last_map[r["model_id"]] = lv["verdict"]
            # Per-model reject-reason breakdown in one grouped pass.
            reason_map: dict[str, dict[str, int]] = {}
            for rr in conn.execute(
                """
                SELECT model_id, reject_reason, COUNT(*) AS n
                FROM verdicts
                WHERE verdict = 'reject' AND reject_reason != ''
                GROUP BY model_id, reject_reason
                """
            ):
                reason_map.setdefault(rr["model_id"], {})[rr["reject_reason"]] = int(
                    rr["n"] or 0
                )
        out: list[VerdictSummary] = []
        for r in rows:
            accept = int(r["accept"] or 0)
            reject = int(r["reject"] or 0)
            needs_review = int(r["needs_review"] or 0)
            total = int(r["total"] or 0)
            out.append(
                VerdictSummary(
                    model_id=r["model_id"],
                    accept=accept,
                    reject=reject,
                    needs_review=needs_review,
                    total=total,
                    last_at=r["last_at"],
                    last_verdict=last_map.get(r["model_id"]),
                    approval_status=approval_status_for(
                        accept, reject, needs_review, total
                    ),
                    reject_reasons=reason_map.get(r["model_id"]) or None,
                )
            )
        return out


def _row_to_verdict(r: sqlite3.Row) -> Verdict:
    return Verdict(
        id=r["id"],
        prediction_id=r["prediction_id"],
        model_id=r["model_id"],
        case_id=r["case_id"],
        verdict=r["verdict"],
        notes=r["notes"] or "",
        reviewer=r["reviewer"] or "",
        fold_choice=r["fold_choice"] or "best",
        mean_dice=r["mean_dice"],
        created_at=r["created_at"],
        reject_reason=(r["reject_reason"] if "reject_reason" in r.keys() else "") or "",
    )
