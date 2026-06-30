"""SQLite-backed per-model card color overrides.

The QA catalog renders one card per trained model. Each card derives a
default swatch from the model's lifecycle status (training/done/failed)
and verdict mix (accept/review/reject) — see ``services/qa-viewer/web/
components/catalog/ModelCard.tsx``. Reviewers can override that with a
named swatch from a curated palette; the override is stored here.

We keep the palette as a closed set of named keys (rather than free-form
hex) so a future palette tweak in CSS doesn't strand stored values.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# The canonical 10-swatch palette. Must stay in sync with the
# `--card-<name>-{bg,fg,ring}` CSS variables in
# ``services/qa-viewer/web/app/globals.css``.
PALETTE = (
    "slate",
    "sky",
    "indigo",
    "violet",
    "fuchsia",
    "rose",
    "amber",
    "lime",
    "emerald",
    "teal",
)


@dataclass
class ModelTheme:
    model_id: str
    color_key: str
    updated_by: str
    updated_at: str  # ISO-8601 UTC


class ModelThemeStore:
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
            """
            CREATE TABLE IF NOT EXISTS model_themes (
                model_id    TEXT PRIMARY KEY,
                color_key   TEXT NOT NULL,
                updated_by  TEXT NOT NULL DEFAULT '',
                updated_at  TEXT NOT NULL
            )
            """
        )

    def all(self) -> dict[str, ModelTheme]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM model_themes").fetchall()
        return {
            r["model_id"]: ModelTheme(
                model_id=r["model_id"],
                color_key=r["color_key"],
                updated_by=r["updated_by"] or "",
                updated_at=r["updated_at"],
            )
            for r in rows
        }

    def set(self, model_id: str, color_key: str, updated_by: str = "") -> ModelTheme:
        if color_key not in PALETTE:
            raise ValueError(
                f"color_key must be one of {PALETTE}; got {color_key!r}"
            )
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_themes (model_id, color_key, updated_by, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    color_key = excluded.color_key,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (model_id, color_key, updated_by, now),
            )
        return ModelTheme(
            model_id=model_id,
            color_key=color_key,
            updated_by=updated_by,
            updated_at=now,
        )

    def delete(self, model_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM model_themes WHERE model_id=?", (model_id,),
            )
            return cur.rowcount > 0
