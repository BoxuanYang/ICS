from types import SimpleNamespace

import pytest
import torch
from minisgl.kvcache.unified import TemporaryKVRegion, UnifiedKVAllocator
from minisgl.moe.memory import ExpertMemoryManager
from minisgl.moe.offload import ExpertEvictionProfile, ExpertOffloadController, ExpertProfileEntry
from minisgl.moe.storage import ExpertStorage, ExpertStorageCatalog
from minisgl.scheduler.expert_cache import ExpertCacheManager, ExpertEvictionRequired


@pytest.mark.parametrize("cache_type", ["naive", "radix"])
def test_admission_eviction_release_and_reuse(monkeypatch, cache_type):
    import minisgl.core as core
    import minisgl.distributed.info as info
    from minisgl.distributed import set_tp_info

    monkeypatch.setattr(info, "_TP_INFO", None)
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))
    set_tp_info(0, 1)
    allocator = UnifiedKVAllocator(2, 16)
    gate = TemporaryKVRegion("gate", (0, 0), 32, 16)
    down = TemporaryKVRegion("down", (0, 0), 16, 16)
    allocator.register_temporary_region(gate)
    allocator.register_temporary_region(down)
    catalog = ExpertStorageCatalog([ExpertStorage(0, 0, 0.1, None, None, gate, down)])
    events = []
    backend = SimpleNamespace(
        route_to_cpu=lambda key: events.append("cpu"),
        restore_weights=lambda key: events.append("copy"),
        route_to_gpu=lambda key: events.append("gpu"),
    )
    profile = ExpertEvictionProfile([ExpertProfileEntry(0, 0, 0.1)])
    manager = ExpertMemoryManager(ExpertOffloadController(profile), catalog, allocator, backend)
    starts = {"persistent": 0, "gate": 3, "down": 5}
    pool = SimpleNamespace(
        allocator=allocator, slot_id=lambda b: starts[b.region_id] + b.local_block_id
    )
    cache = ExpertCacheManager(pool, manager, torch.zeros(1, 8, dtype=torch.int32), cache_type)
    assert cache.available_size == 5
    assert events == []  # Merely inspecting admission capacity must not evict.
    with pytest.raises(ExpertEvictionRequired):
        cache._allocate(5)
    assert events == []
    cache.evict_for_batch(5)
    blocks = cache._allocate(5)
    assert events == ["cpu"]
    assert set(blocks.tolist()) == {0, 1, 3, 4, 5}  # Slot 2 is the dummy page.
    with pytest.raises(ExpertEvictionRequired):
        cache._allocate(1)
    if cache_type == "radix":
        cache.prefix_cache.insert_prefix(torch.arange(3, dtype=torch.int32), blocks[2:])
        cache._free(blocks[:2])
        assert events == ["cpu"]  # Prefix ownership keeps the expert on CPU.
        cache.check_integrity()
        cache._release(cache.prefix_cache.evict(3))
        cache.flush_pending_restorations()
    else:
        with cache.lazy_free_region():
            cache._free(blocks[:3])
            cache._free(blocks[3:])
    assert events == ["cpu", "copy", "gpu"]
    cache.check_integrity()
    cache.evict_for_batch(5)
    cache._free(cache._allocate(5))
    cache.flush_pending_restorations()
    assert events == ["cpu", "copy", "gpu"] * 2


def test_finished_request_does_not_pin_expert_storage_in_radix_cache(monkeypatch):
    import minisgl.core as core
    import minisgl.distributed.info as info
    from minisgl.distributed import set_tp_info

    monkeypatch.setattr(info, "_TP_INFO", None)
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))
    set_tp_info(0, 1)
    allocator = UnifiedKVAllocator(2, 16)
    gate = TemporaryKVRegion("gate", (0, 0), 32, 16)
    down = TemporaryKVRegion("down", (0, 0), 16, 16)
    allocator.register_temporary_region(gate)
    allocator.register_temporary_region(down)
    catalog = ExpertStorageCatalog([ExpertStorage(0, 0, 0.1, None, None, gate, down)])
    events = []
    backend = SimpleNamespace(
        route_to_cpu=lambda key: events.append("cpu"),
        restore_weights=lambda key: events.append("copy"),
        route_to_gpu=lambda key: events.append("gpu"),
    )
    profile = ExpertEvictionProfile([ExpertProfileEntry(0, 0, 0.1)])
    manager = ExpertMemoryManager(
        ExpertOffloadController(profile), catalog, allocator, backend
    )
    starts = {"persistent": 0, "gate": 3, "down": 5}
    pool = SimpleNamespace(
        allocator=allocator,
        slot_id=lambda block: starts[block.region_id] + block.local_block_id,
    )
    page_table = torch.zeros(1, 8, dtype=torch.int32)
    cache = ExpertCacheManager(pool, manager, page_table, "radix")
    root_handle = cache.prefix_cache.match_prefix(
        torch.empty(0, dtype=torch.int32)
    ).cuda_handle
    req = SimpleNamespace(
        uid=7,
        table_idx=0,
        cached_len=0,
        device_len=5,
        cache_handle=root_handle,
        input_ids=torch.arange(5, dtype=torch.int32),
    )

    cache.evict_for_batch(5)
    cache.allocate_paged([req])
    req.cached_len = 5
    cache.cache_req(req, finished=False)
    assert events == ["cpu"]
    assert cache.prefix_cache.size_info.total_size == 0
    cache.cache_req(req, finished=True)
    cache.flush_pending_restorations()

    assert events == ["cpu", "copy", "gpu"]
    assert cache.prefix_cache.size_info.total_size == 0
    assert not cache.live
    cache.check_integrity()


def test_temporary_kv_stays_in_place_when_persistent_slots_are_released(monkeypatch):
    import minisgl.core as core
    import minisgl.distributed.info as info
    from minisgl.distributed import set_tp_info

    monkeypatch.setattr(info, "_TP_INFO", None)
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))
    set_tp_info(0, 1)
    allocator = UnifiedKVAllocator(2, 16)
    gate = TemporaryKVRegion("gate", (0, 0), 32, 16)
    down = TemporaryKVRegion("down", (0, 0), 16, 16)
    allocator.register_temporary_region(gate)
    allocator.register_temporary_region(down)
    catalog = ExpertStorageCatalog([ExpertStorage(0, 0, 0.1, None, None, gate, down)])
    events = []
    backend = SimpleNamespace(
        route_to_cpu=lambda key: events.append("cpu"),
        restore_weights=lambda key: events.append("copy"),
        route_to_gpu=lambda key: events.append("gpu"),
    )
    manager = ExpertMemoryManager(
        ExpertOffloadController(
            ExpertEvictionProfile([ExpertProfileEntry(0, 0, 0.1)])
        ),
        catalog,
        allocator,
        backend,
    )
    starts = {"persistent": 0, "gate": 3, "down": 5}
    copies = []
    pool = SimpleNamespace(
        allocator=allocator,
        slot_id=lambda block: starts[block.region_id] + block.local_block_id,
        copy_slots=lambda sources, destinations: copies.append(
            (sources.tolist(), destinations.tolist())
        ),
    )
    page_table = torch.zeros(1, 8, dtype=torch.int32)
    cache = ExpertCacheManager(pool, manager, page_table, "radix")
    handle = cache.prefix_cache.match_prefix(
        torch.empty(0, dtype=torch.int32)
    ).cuda_handle
    req = SimpleNamespace(
        uid=9,
        table_idx=0,
        cached_len=0,
        device_len=5,
        cache_handle=handle,
        input_ids=torch.arange(5, dtype=torch.int32),
    )

    cache.evict_for_batch(5)
    cache.allocate_paged([req])
    assert events == ["cpu"]
    assert set(cache.temporary_locations) == {3, 4, 5}
    assert cache.uses_temporary(req)

    cache._release(torch.tensor([0, 1], dtype=torch.int32))
    cache.flush_pending_restorations()
    assert copies == []
    assert page_table[0, 2:5].tolist() == [3, 4, 5]
    assert set(cache.temporary_locations) == {3, 4, 5}
    assert cache.temporary_counts == {0: 3}
    assert events == ["cpu"]

    cache._release(torch.tensor([3, 4, 5], dtype=torch.int32))
    cache.flush_pending_restorations()
    assert copies == []
    assert not cache.temporary_locations
    assert not cache.uses_temporary(req)
    assert events == ["cpu", "copy", "gpu"]


def test_prefix_eviction_does_not_migrate_into_slots_needed_by_allocation(monkeypatch):
    import minisgl.core as core
    import minisgl.distributed.info as info
    from minisgl.distributed import set_tp_info

    monkeypatch.setattr(info, "_TP_INFO", None)
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))
    set_tp_info(0, 1)
    allocator = UnifiedKVAllocator(2, 16)
    gate = TemporaryKVRegion("gate", (0, 0), 32, 16)
    down = TemporaryKVRegion("down", (0, 0), 16, 16)
    allocator.register_temporary_region(gate)
    allocator.register_temporary_region(down)
    catalog = ExpertStorageCatalog([ExpertStorage(0, 0, 0.1, None, None, gate, down)])
    backend = SimpleNamespace(
        route_to_cpu=lambda key: None,
        restore_weights=lambda key: None,
        route_to_gpu=lambda key: None,
    )
    manager = ExpertMemoryManager(
        ExpertOffloadController(ExpertEvictionProfile([ExpertProfileEntry(0, 0, 0.1)])),
        catalog,
        allocator,
        backend,
    )
    copies = []
    pool = SimpleNamespace(
        allocator=allocator,
        slot_id=lambda block: {"persistent": 0, "gate": 3, "down": 5}[block.region_id]
        + block.local_block_id,
        copy_slots=lambda sources, destinations: copies.append((sources, destinations)),
    )
    cache = ExpertCacheManager(pool, manager, torch.zeros(1, 8, dtype=torch.int32), "radix")
    cache.evict_for_batch(5)
    blocks = cache._allocate(5)
    cache.prefix_cache.insert_prefix(
        torch.tensor([11, 12], dtype=torch.int32), blocks[:2]
    )
    cache.temporary_locations[blocks[2].item()] = (0, 2)
    cache.page_table[0, 2] = blocks[2]

    allocated = cache._allocate(1)
    assert allocated.item() in (0, 1)
    assert copies == []
    assert blocks[2].item() in cache.temporary_locations
    assert manager.available_gpu_capacity_blocks == 0
