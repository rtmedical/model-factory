"""Redis pointer-cache for QA inference results.

Three classes of value are stored (Redis is a pointer-table; the actual
bytes live on NFS under `/factory/qa-cohort/predictions/{id}/`):

- `infer::{plan_hash}::{model_id}::{folds_csv}::{case_id}` →
  `prediction_id`. Lets a re-click on the same (model, case, folds)
  triple return the existing prediction instead of re-running.
- `metrics::{plan_hash}::{model_id}::{folds_csv}::{case_id}::v{gt_rev}` →
  JSON blob (the `metrics.json` payload). GT-revision-keyed so a
  groundtruth correction invalidates metrics but not the seg.
- `meshes::{plan_hash}::{model_id}::{folds_csv}::{case_id}` →
  `prediction_id`. Same prediction_id whose mesh/*.vtp files are
  already on disk.

Connection is lazy; every accessor returns None / no-ops if the Redis
URL is unset or unreachable. The viewer must keep working when Redis is
down — the cache is opportunistic, not load-bearing.

Plan hash is computed via `modelfactory.qa.preprocess._plan_hash`, so
the cache invalidates automatically when plans.json or
dataset_fingerprint.json changes upstream.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default TTL: 30 days. The cache is reproducible from on-disk seg files,
# so even an aggressive expiry only costs one extra inference run.
_DEFAULT_TTL_SECONDS = 30 * 24 * 3600


class _RedisHandle:
    """Lazy redis-py wrapper, safe to call when Redis is unreachable.

    Single global instance owned by `_get_handle()`. First failed call
    flips `_disabled` so we don't pay a connection timeout per request
    when the sidecar is down — a fresh `_get_handle()` re-checks the URL
    so an admin can revive the cache by setting QA_REDIS_URL and bouncing
    the pod.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._client: Any | None = None
        self._disabled = False
        self._lock = threading.Lock()

    def _client_or_none(self) -> Any | None:
        if self._disabled:
            return None
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import redis  # local import: optional dep
                self._client = redis.Redis.from_url(
                    self.url,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                    decode_responses=True,
                )
                # Probe immediately so we don't discover a dead Redis on
                # the first hot-path call.
                self._client.ping()
                logger.info("qa cache: connected to %s", self.url)
                return self._client
            except Exception as exc:  # noqa: BLE001
                logger.warning("qa cache: Redis unreachable (%s) — disabled", exc)
                self._disabled = True
                self._client = None
                return None

    def get(self, key: str) -> str | None:
        c = self._client_or_none()
        if c is None:
            return None
        try:
            return c.get(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qa cache: GET %s failed: %s", key, exc)
            return None

    def set(self, key: str, value: str, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        c = self._client_or_none()
        if c is None:
            return
        try:
            c.set(key, value, ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qa cache: SET %s failed: %s", key, exc)

    def delete(self, key: str) -> None:
        c = self._client_or_none()
        if c is None:
            return
        try:
            c.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qa cache: DEL %s failed: %s", key, exc)


_handle: _RedisHandle | None = None
_handle_lock = threading.Lock()


def _get_handle() -> _RedisHandle | None:
    global _handle
    if _handle is not None:
        return _handle
    url = os.environ.get("QA_REDIS_URL", "").strip()
    if not url:
        return None
    with _handle_lock:
        if _handle is None:
            _handle = _RedisHandle(url)
    return _handle


# ── key builders ───────────────────────────────────────────────────────


def _folds_csv(folds: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(f) for f in folds)


def _infer_key(plan_hash: str, model_id: str, folds, case_id: str) -> str:
    return f"infer::{plan_hash}::{model_id}::{_folds_csv(folds)}::{case_id}"


def _metrics_key(
    plan_hash: str, model_id: str, folds, case_id: str, gt_revision: int | None
) -> str:
    rev = "none" if gt_revision is None else f"v{int(gt_revision)}"
    return f"metrics::{plan_hash}::{model_id}::{_folds_csv(folds)}::{case_id}::{rev}"


def _meshes_key(plan_hash: str, model_id: str, folds, case_id: str) -> str:
    return f"meshes::{plan_hash}::{model_id}::{_folds_csv(folds)}::{case_id}"


# ── public accessors ──────────────────────────────────────────────────


def get_inference(
    plan_hash: str,
    model_id: str,
    folds: tuple[int, ...],
    case_id: str,
) -> str | None:
    """Return the cached `prediction_id` for this (model, folds, case), if any."""
    h = _get_handle()
    if h is None:
        return None
    return h.get(_infer_key(plan_hash, model_id, folds, case_id))


def set_inference(
    plan_hash: str,
    model_id: str,
    folds: tuple[int, ...],
    case_id: str,
    prediction_id: str,
) -> None:
    h = _get_handle()
    if h is None:
        return
    h.set(_infer_key(plan_hash, model_id, folds, case_id), prediction_id)


def get_metrics(
    plan_hash: str,
    model_id: str,
    folds: tuple[int, ...],
    case_id: str,
    gt_revision: int | None,
) -> list[dict] | None:
    """Return the cached metrics list, or None."""
    h = _get_handle()
    if h is None:
        return None
    raw = h.get(_metrics_key(plan_hash, model_id, folds, case_id, gt_revision))
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


def set_metrics(
    plan_hash: str,
    model_id: str,
    folds: tuple[int, ...],
    case_id: str,
    gt_revision: int | None,
    metrics: list[dict],
) -> None:
    h = _get_handle()
    if h is None:
        return
    h.set(
        _metrics_key(plan_hash, model_id, folds, case_id, gt_revision),
        json.dumps(metrics),
    )


def get_meshes(
    plan_hash: str,
    model_id: str,
    folds: tuple[int, ...],
    case_id: str,
) -> str | None:
    """Return the prediction_id whose mesh/*.vtp files are on disk, or None."""
    h = _get_handle()
    if h is None:
        return None
    return h.get(_meshes_key(plan_hash, model_id, folds, case_id))


def set_meshes(
    plan_hash: str,
    model_id: str,
    folds: tuple[int, ...],
    case_id: str,
    prediction_id: str,
) -> None:
    h = _get_handle()
    if h is None:
        return
    h.set(_meshes_key(plan_hash, model_id, folds, case_id), prediction_id)


def invalidate_inference(
    plan_hash: str,
    model_id: str,
    folds: tuple[int, ...],
    case_id: str,
) -> None:
    h = _get_handle()
    if h is None:
        return
    h.delete(_infer_key(plan_hash, model_id, folds, case_id))


def plan_hash_for_model(model_dir: Path) -> str:
    """Forward to the existing helper in qa.preprocess so callers in
    api.py don't need a second import."""
    from modelfactory.qa.preprocess import _plan_hash
    return _plan_hash(model_dir)
