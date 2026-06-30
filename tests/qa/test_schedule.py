"""Host-runnable tests for the future-trainings scheduler.

Pure + fast: the SQLite store and the ETA projection have no GPU / cluster /
Redis dependency, so they're exercised directly against a temp DB and with an
injected `now_ms` (project_schedule never reads the wall clock).
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

from modelfactory.qa.schedule import (
    ScheduleStore,
    default_duration_hours,
    project_schedule,
)

H = 3_600_000.0  # one hour in ms


def _store() -> ScheduleStore:
    return ScheduleStore(Path(tempfile.mkdtemp(prefix="qa-sched-")) / "qa.sqlite")


def test_duration_prior_by_plan():
    assert default_duration_hours("nnUNetResEncUNetLPlans") == 72.0
    assert default_duration_hours("nnUNetResEncUNetLPlans_HighRes") == 96.0


def test_add_is_idempotent_per_dataset_fold():
    s = _store()
    a = s.add(dataset_key="thorax_clinical_breast_l",
              dataset_name="Dataset129_Thorax_Clinical_Breast_L", fold=1,
              trainer="nnUNetTrainerMLflow", plans="nnUNetResEncUNetLPlans",
              priority=50, notes="v1")
    a2 = s.add(dataset_key="thorax_clinical_breast_l",
               dataset_name="Dataset129_Thorax_Clinical_Breast_L", fold=1,
               trainer="nnUNetTrainerMLflow", plans="nnUNetResEncUNetLPlans",
               priority=60, notes="v2")
    assert a.id == a2.id
    rows = s.list_all()
    assert len(rows) == 1
    assert rows[0].priority == 60 and rows[0].notes == "v2"


def test_list_orders_by_priority_then_age():
    s = _store()
    s.add(dataset_key="d_lo", dataset_name="D1", fold=0, trainer="t",
          plans="nnUNetResEncUNetLPlans", priority=10)
    s.add(dataset_key="d_hi", dataset_name="D2", fold=0, trainer="t",
          plans="nnUNetResEncUNetLPlans", priority=90)
    rows = s.list_all()
    assert [r.dataset_key for r in rows] == ["d_hi", "d_lo"]


def test_update_and_delete():
    s = _store()
    p = s.add(dataset_key="d", dataset_name="D", fold=0, trainer="t",
              plans="nnUNetResEncUNetLPlans")
    s.update(p.id, priority=5, notes="hi")
    assert s.get(p.id).priority == 5
    assert s.delete(p.id) is True
    assert s.get(p.id) is None


def test_projection_free_slots_start_now():
    s = _store()
    s.add(dataset_key="a", dataset_name="A", fold=0, trainer="t",
          plans="nnUNetResEncUNetLPlans", priority=1)
    s.add(dataset_key="b", dataset_name="B", fold=0, trainer="t",
          plans="nnUNetResEncUNetLPlans", priority=2)
    proj = project_schedule(s.list_all(), running_finish_ms=[1 * H, 2 * H],
                            slots=10, now_ms=0.0)
    # plenty of free slots -> everything starts at now (epoch 0)
    assert all(p.scheduled_start == "1970-01-01T00:00:00+00:00" for p in proj)
    # 72h prior -> finish at +72h
    assert all(abs(p.eta_seconds - 72 * 3600) < 1 for p in proj)


def test_projection_serializes_under_contention():
    s = _store()
    s.add(dataset_key="a", dataset_name="A", fold=0, trainer="t",
          plans="nnUNetResEncUNetLPlans", priority=2)
    s.add(dataset_key="b", dataset_name="B", fold=0, trainer="t",
          plans="nnUNetResEncUNetLPlans", priority=1)
    # one slot, one running fold that finishes in 5h
    proj = project_schedule(s.list_all(), running_finish_ms=[5 * H], slots=1, now_ms=0.0)
    starts = [dt.datetime.fromisoformat(p.scheduled_start) for p in proj]
    five_h = dt.datetime.fromtimestamp(5 * H / 1000, dt.UTC)
    assert starts[0] >= five_h                 # waits for the running fold
    assert starts[1] >= starts[0]              # second waits behind first
    # higher-priority item ('a') scheduled first
    assert proj[0].dataset_key == "a"


def test_reconcile_closes_live_folds():
    s = _store()
    s.add(dataset_key="live", dataset_name="L", fold=0, trainer="t",
          plans="nnUNetResEncUNetLPlans")
    s.add(dataset_key="still_queued", dataset_name="Q", fold=0, trainer="t",
          plans="nnUNetResEncUNetLPlans")
    closed = s.reconcile({"live::0"})
    assert closed == 1
    planned = s.list_all(status="planned")
    assert [p.dataset_key for p in planned] == ["still_queued"]
    assert len(s.list_all(status="submitted")) == 1
