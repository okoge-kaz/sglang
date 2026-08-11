from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats


def _make_weight_updater(monkeypatch, *, applied_version=10):
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr("torch.distributed.barrier", lambda **_kwargs: None)

    runner = MagicMock()
    worker = SimpleNamespace(iter_runners=lambda: [("", runner)])
    scheduler = SimpleNamespace(
        applied_weight_version=applied_version,
        pending_weight_version=None,
    )
    manager = SchedulerWeightUpdaterManager(
        tp_worker=worker,
        draft_worker=None,
        tp_cpu_group=object(),
        memory_saver_adapter=MagicMock(),
        flush_cache=MagicMock(return_value=True),
        is_fully_idle=MagicMock(return_value=True),
        scheduler=scheduler,
    )
    return manager, scheduler, runner


def test_weight_version_commits_only_after_end(monkeypatch):
    manager, scheduler, runner = _make_weight_updater(monkeypatch)

    begin = manager.begin_weight_update(
        BeginWeightUpdateReqInput(weight_version="11")
    )
    assert begin.success
    assert scheduler.applied_weight_version == 10
    assert scheduler.pending_weight_version == 11

    end = manager.end_weight_update(EndWeightUpdateReqInput())
    assert end.success
    assert scheduler.applied_weight_version == 11
    assert scheduler.pending_weight_version is None
    runner.end_weight_update.assert_called_once_with(run_post_load=True)


def test_failed_finalize_does_not_advance_applied_version(monkeypatch):
    manager, scheduler, runner = _make_weight_updater(monkeypatch)
    assert manager.begin_weight_update(
        BeginWeightUpdateReqInput(weight_version="11")
    ).success
    runner.end_weight_update.side_effect = RuntimeError("finalize failed")

    end = manager.end_weight_update(EndWeightUpdateReqInput())
    assert not end.success
    assert scheduler.applied_weight_version == 10
    assert scheduler.pending_weight_version is None


def test_forward_stamp_tracks_prefill_and_mixed_versions():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.applied_weight_version = 10
    stats = SchedulerReqTimeStats()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        reqs=[SimpleNamespace(time_stats=stats)],
    )

    scheduler.stamp_forward_weight_version(batch)
    scheduler.applied_weight_version = 11
    batch.forward_mode = SimpleNamespace(is_extend=lambda: False)
    scheduler.stamp_forward_weight_version(batch)

    assert stats.first_prefill_weight_version == 10
    assert stats.min_forward_weight_version == 10
    assert stats.max_forward_weight_version == 11
    assert stats.last_forward_weight_version == 11


def test_policy_version_stats_serialize_when_metrics_are_disabled():
    stats = SchedulerReqTimeStats(
        first_prefill_weight_version=10,
        min_forward_weight_version=10,
        max_forward_weight_version=11,
        last_forward_weight_version=11,
    )
    stats.enable_metrics = False

    assert stats.__getstate__() == {
        "first_prefill_weight_version": 10,
        "min_forward_weight_version": 10,
        "max_forward_weight_version": 11,
        "last_forward_weight_version": 11,
    }
