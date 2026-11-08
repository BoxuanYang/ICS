from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.expert_cache import ExpertEvictionRequired


def test_normal_overlap_launches_next_batch_before_retiring_previous():
    scheduler = object.__new__(Scheduler)
    actions = []
    scheduler.engine = NS(expert_memory=None, stream=NS(wait_stream=lambda _: None))
    scheduler.stream = None
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.prefill_manager = NS(runnable=True)
    scheduler.decode_manager = NS(runnable=False)
    scheduler.receive_msg = lambda **_: []
    scheduler._select_next_batch = lambda: object()
    scheduler._prepare_batch = lambda _: actions.append("prepare") or "prepared"
    scheduler._forward = lambda _: actions.append("launch") or "output"
    scheduler._process_last_data = lambda _: actions.append("retire")
    assert scheduler.overlap_loop("previous") == ("prepared", "output")
    assert actions == ["prepare", "launch", "retire"]


@pytest.mark.parametrize("needs_eviction", [False, True])
def test_slow_path_retires_once_before_retry_and_launch(needs_eviction, monkeypatch):
    import minisgl.scheduler.scheduler as module
    monkeypatch.setattr(module.logger, "info_rank0", lambda *args: None)
    scheduler = object.__new__(Scheduler)
    actions = []
    scheduler.engine = NS(expert_memory=None, stream=NS(wait_stream=lambda _: None))
    scheduler.stream = None
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.prefill_manager = NS(runnable=True)
    scheduler.decode_manager = NS(runnable=False)
    scheduler.receive_msg = lambda **_: []
    scheduler.cache_manager = NS(
        poll_restorations=lambda: actions.append("poll"),
        evict_for_batch=lambda _: actions.append("evict"),
    )
    batch = NS(reqs=[object()], is_decode=False)
    scheduler._select_next_batch = lambda: batch
    attempts = 0

    def prepare(_):
        nonlocal attempts
        attempts += 1
        actions.append("prepare")
        if attempts == 1 or (needs_eviction and attempts == 2):
            raise ExpertEvictionRequired(7)
        return "prepared"

    def retire(data, **kwargs):
        if data is not None:
            assert not kwargs.get("flush_restorations", True)
            actions.append("retire")

    scheduler._prepare_batch = prepare
    scheduler._process_last_data = retire
    scheduler._forward = lambda _: actions.append("launch") or "output"
    result = scheduler.overlap_loop("previous")
    assert result == ("prepared", "output")
    expected = ["prepare", "retire", "poll", "prepare"]
    if needs_eviction:
        expected += ["evict", "prepare"]
    assert actions == expected + ["launch"]


def test_deferred_prefill_preserves_chunk_and_returns_new_admission_resources():
    from minisgl.scheduler.prefill import PrefillManager
    actions = []
    manager = PrefillManager(
        NS(unlock=lambda handle: actions.append(("unlock", handle))),
        NS(free=lambda table: actions.append(("free", table))), NS())
    old_chunk = object()
    first, second = NS(uid=1, chunked_req=None), NS(uid=2, chunked_req=None)
    pending = [first, second]
    manager._selection_snapshot = (pending, [(first, old_chunk), (second, None)])
    manager.pending_list = []
    manager.defer_selected_batch(NS(reqs=[
        NS(uid=1, table_idx=3, cache_handle="old"),
        NS(uid=2, table_idx=4, cache_handle="new")]))
    assert manager.pending_list is pending
    assert first.chunked_req is old_chunk
    assert second.chunked_req is None
    assert actions == [("unlock", "new"), ("free", 4)]


