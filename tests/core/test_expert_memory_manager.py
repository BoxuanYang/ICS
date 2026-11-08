from __future__ import annotations

import pytest
from minisgl.kvcache.unified import TemporaryKVRegionState, UnifiedKVAllocator
from minisgl.moe.memory import ExpertMemoryManager
from minisgl.moe.offload import (
    ExpertEvictionProfile,
    ExpertOffloadController,
    ExpertProfileEntry,
    ExpertResidencyState,
)
from minisgl.moe.storage import ExpertStorage, ExpertStorageCatalog


class FakeTensor:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes

    def numel(self) -> int:
        return self.nbytes // 2

    def element_size(self) -> int:
        return 2


class RecordingBackend:
    def __init__(self) -> None:
        self.events = []
        self.fail_restore = False

    def route_to_cpu(self, key) -> None:
        self.events.append(("cpu", key))

    def restore_weights(self, key) -> None:
        self.events.append(("copy", key))
        if self.fail_restore:
            raise RuntimeError("copy failed")

    def route_to_gpu(self, key) -> None:
        self.events.append(("gpu", key))


def _manager() -> ExpertMemoryManager:
    profile = ExpertEvictionProfile(
        [
            ExpertProfileEntry(0, 0, 0.01),
            ExpertProfileEntry(0, 1, 0.02),
        ]
    )
    catalog = ExpertStorageCatalog(
        [
            ExpertStorage(0, 0, 0.01, FakeTensor(32), FakeTensor(16)),
            ExpertStorage(0, 1, 0.02, FakeTensor(16), FakeTensor(16)),
        ]
    )
    allocator = UnifiedKVAllocator(persistent_capacity_blocks=1, block_nbytes=16)
    catalog.create_and_register_regions(block_nbytes=16, allocator=allocator)
    return ExpertMemoryManager(
        ExpertOffloadController(profile), catalog, allocator, RecordingBackend()
    )


def test_execution_switch_precedes_storage_reuse() -> None:
    manager = _manager()
    assert manager.available_gpu_capacity_blocks == 5
    assert [entry.key for entry in manager.plan_evictions(4).experts] == [(0, 0)]
    manager.begin_eviction((0, 0))
    assert manager.backend.events == [("cpu", (0, 0))]
    assert manager.allocator.temporary_available_blocks == 0
    manager.complete_eviction((0, 0))
    assert manager.allocator.temporary_available_blocks == 3
    assert manager.available_gpu_capacity_blocks == 2


def test_transition_barrier_orders_eviction_and_restoration() -> None:
    manager = _manager()
    events = manager.backend.events
    manager.transition_barrier = lambda: events.append(("barrier", None))

    manager.begin_eviction((0, 0))
    manager.complete_eviction((0, 0))
    manager.allocator.release(manager.allocator.allocate(3))
    manager.flush_pending_restorations()

    assert events == [
        ("barrier", None),
        ("cpu", (0, 0)),
        ("copy", (0, 0)),
        ("gpu", (0, 0)),
    ]


def test_each_empty_expert_restores_after_allocation_phase() -> None:
    manager = _manager()
    for key in ((0, 0), (0, 1)):
        manager.begin_eviction(key)
        manager.complete_eviction(key)
    blocks = manager.allocator.allocate(6)
    regions0 = manager._regions((0, 0))
    regions1 = manager._regions((0, 1))
    manager.allocator.release(
        [b for b in blocks if b.region_id in {r.region_id for r in regions0}]
    )
    assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.CPU_RESIDENT
    assert manager.controller.residency.state((0, 1)) is ExpertResidencyState.CPU_RESIDENT
    assert all(region.state is TemporaryKVRegionState.ACTIVE for region in regions0)
    manager.flush_pending_restorations()
    assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.GPU_RESIDENT
    assert manager.controller.residency.state((0, 1)) is ExpertResidencyState.CPU_RESIDENT

    manager.allocator.release(
        [b for b in blocks if b.region_id in {r.region_id for r in regions1}]
    )
    manager.flush_pending_restorations()
    assert manager.backend.events[-4:] == [
        ("copy", (0, 0)),
        ("gpu", (0, 0)),
        ("copy", (0, 1)),
        ("gpu", (0, 1)),
    ]
    assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.GPU_RESIDENT
    assert manager.controller.residency.state((0, 1)) is ExpertResidencyState.GPU_RESIDENT
    assert manager.available_gpu_capacity_blocks == 5
    assert all(
        region.state is TemporaryKVRegionState.INACTIVE
        for region in (*regions0, *regions1)
    )
    assert manager.allocator.available_blocks == 0  # Persistent pool is still full.


def test_never_used_sibling_does_not_prevent_eager_restore() -> None:
    manager = _manager()
    manager.begin_eviction((0, 0))
    manager.complete_eviction((0, 0))
    blocks = manager.allocator.allocate(2)  # Down region stays unused.
    manager.allocator.release(blocks[1:])
    manager.flush_pending_restorations()
    assert manager.backend.events[-2:] == [("copy", (0, 0)), ("gpu", (0, 0))]


def test_failed_copy_keeps_expert_on_cpu_and_regions_unavailable() -> None:
    manager = _manager()
    manager.begin_eviction((0, 0))
    manager.complete_eviction((0, 0))
    blocks = manager.allocator.allocate(4)
    manager.backend.fail_restore = True
    with pytest.raises(RuntimeError, match="copy failed"):
        manager.allocator.release(blocks)
        manager.flush_pending_restorations()
    assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.RESTORING
    assert manager.backend.events[-1] == ("copy", (0, 0))
    assert manager.allocator.temporary_available_blocks == 0


def test_same_expert_can_be_evicted_again_after_restoration() -> None:
    manager = _manager()
    for _ in range(2):
        manager.begin_eviction((0, 0))
        manager.complete_eviction((0, 0))
        manager.allocator.release(manager.allocator.allocate(4))
        manager.flush_pending_restorations()
        assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.GPU_RESIDENT
    assert (
        manager.backend.events
        == [
            ("cpu", (0, 0)),
            ("copy", (0, 0)),
            ("gpu", (0, 0)),
        ]
        * 2
    )


def test_reservation_prevents_fetch_until_final_release() -> None:
    manager = _manager()
    manager.begin_eviction((0, 0))
    manager.complete_eviction((0, 0))
    manager.allocator.release(manager.allocator.allocate(4))
    reserved = manager.allocator.allocate(4)

    assert manager.flush_pending_restorations() == 0
    assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.CPU_RESIDENT
    manager.allocator.release(reserved)
    assert manager.flush_pending_restorations() == 1
    assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.GPU_RESIDENT


def test_admission_window_limits_promised_expert_capacity() -> None:
    manager = _manager()
    manager.max_offloaded_experts = 1
    assert manager.admission_eviction_capacity_blocks == 3
    assert [entry.key for entry in manager.plan_evictions(5).experts] == [(0, 0)]

    manager.begin_eviction((0, 0))
    manager.complete_eviction((0, 0))
    assert manager.admission_eviction_capacity_blocks == 0
    assert manager.plan_evictions(5).experts == ()


def test_fetch_completion_requires_all_ranks_and_excludes_kv_allocations():
    manager = _manager()
    manager.begin_eviction((0, 0))
    manager.complete_eviction((0, 0))
    blocks = manager.allocator.allocate(4)
    class Event:
        ready = False
        def query(self):
            return self.ready
    event = Event()
    manager.backend.restore_weights = lambda _: event
    manager.allocator.release(blocks[1:])
    manager.flush_pending_restorations()
    assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.RESTORING
    assert manager.allocator.available_blocks == 0
    with pytest.raises(MemoryError):
        manager.allocator.allocate(1)
    assert manager.poll_restorations() == 0
    event.ready = True
    manager.backend.agree_fetch_completion = lambda ready: [False]
    assert manager.poll_restorations() == 0
    manager.backend.agree_fetch_completion = lambda ready: ready
    assert manager.poll_restorations() == 1
    assert manager.controller.residency.state((0, 0)) is ExpertResidencyState.GPU_RESIDENT
