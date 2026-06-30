"""In-memory queue for QA-viewer prediction submissions.

Before 0.6.0, `api.py` serialised inference behind a single `asyncio.Lock`
named `_predict_lock`. Two concurrent reviewers would both call POST
/api/predict, both get a `prediction_id` back, both see their UIs flip
to "running" — and then the second one would silently block for 5
minutes while the first inference held the GPU. No queue depth, no ETA,
no "you're position 2" feedback.

`PredictQueue` keeps that single lock (single GPU, single CUDA context)
but layers a position-aware queue around it so the API surface can
report:

  - how many predictions are currently in flight,
  - the requesting prediction's 0-based position in line,
  - an ETA derived from a rolling history of recent runs for the same
    model_id.

Submission order matches FIFO insertion order — the actual lock
acquisition order is whatever asyncio's scheduler gives us, which is
also FIFO in practice but isn't guaranteed by spec. Close enough for
"approximately position N" UX.

State is fully in-memory: a pod restart wipes the queue, which is fine
because a pod restart also drops the GPU and CUDA context — anything
queued at restart time was going to fail anyway.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections import deque
from dataclasses import dataclass


@dataclass
class InFlight:
    """One entry in the queue. Mutated in-place as the prediction advances."""

    prediction_id: str
    model_id: str
    case_id: str
    reviewer: str
    submitted_at: str  # ISO-8601 UTC
    started_at: str | None = None  # set when GPU lock acquired
    eta_s: float | None = None  # estimated remaining time at submission

    @property
    def state(self) -> str:
        return "running" if self.started_at else "queued"


class PredictQueue:
    """In-memory FIFO queue of in-flight predictions.

    Single GPU lock (`gpu_lock`) is the actual serialisation primitive;
    every entry waits its turn on this lock in `_run_predict_background`.
    The queue layer just provides observability so a second user knows
    they're waiting and roughly how long.
    """

    def __init__(self, history_size: int = 5, default_eta_s: float = 60.0) -> None:
        self._gpu_lock = asyncio.Lock()
        self._in_flight: dict[str, InFlight] = {}
        self._order: list[str] = []
        self._history: dict[str, deque[float]] = {}
        self._history_size = history_size
        self._default_eta_s = default_eta_s
        self._mu = asyncio.Lock()  # protects _in_flight + _order

    @property
    def gpu_lock(self) -> asyncio.Lock:
        """The actual lock that serialises GPU work. Acquire it for the
        duration of the inference call."""
        return self._gpu_lock

    async def submit(
        self,
        prediction_id: str,
        model_id: str,
        case_id: str,
        reviewer: str = "",
    ) -> InFlight:
        """Register a new prediction in the queue and return its entry."""
        async with self._mu:
            entry = InFlight(
                prediction_id=prediction_id,
                model_id=model_id,
                case_id=case_id,
                reviewer=reviewer,
                submitted_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                eta_s=self._eta_for_model_locked(model_id),
            )
            self._in_flight[prediction_id] = entry
            self._order.append(prediction_id)
            return entry

    async def mark_started(self, prediction_id: str) -> None:
        """Mark the moment the GPU lock was acquired and inference began."""
        async with self._mu:
            entry = self._in_flight.get(prediction_id)
            if entry is None:
                return
            entry.started_at = dt.datetime.now(dt.timezone.utc).isoformat()

    async def remove(
        self,
        prediction_id: str,
        elapsed_s: float | None = None,
    ) -> None:
        """Remove the entry from the queue once the prediction terminates.

        If `elapsed_s` is provided, it feeds the per-model rolling history
        used by `eta_for_model`.
        """
        async with self._mu:
            entry = self._in_flight.pop(prediction_id, None)
            try:
                self._order.remove(prediction_id)
            except ValueError:
                pass
            if entry is not None and elapsed_s is not None and elapsed_s > 0:
                hist = self._history.setdefault(
                    entry.model_id, deque(maxlen=self._history_size)
                )
                hist.append(float(elapsed_s))

    def position(self, prediction_id: str) -> int | None:
        """0-based queue position, or None if not currently queued."""
        try:
            return self._order.index(prediction_id)
        except ValueError:
            return None

    def depth(self) -> int:
        return len(self._order)

    def in_flight(self) -> list[InFlight]:
        return [
            self._in_flight[pid] for pid in self._order if pid in self._in_flight
        ]

    def eta_for_model(self, model_id: str) -> float:
        """Mean of the rolling history; falls back to the default."""
        return self._eta_for_model_locked(model_id)

    def _eta_for_model_locked(self, model_id: str) -> float:
        hist = self._history.get(model_id)
        if hist:
            return sum(hist) / len(hist)
        return self._default_eta_s


# Module-level singleton — api.py imports this directly so the queue is
# shared across all request handlers in the worker process.
queue = PredictQueue()
