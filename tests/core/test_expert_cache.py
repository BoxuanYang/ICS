from types import SimpleNamespace

import pytest
import torch
from minisgl.kvcache.unified import TemporaryKVRegion, UnifiedKVAllocator
from minisgl.moe.memory import ExpertMemoryManager
from minisgl.moe.offload import ExpertEvictionProfile, ExpertOffloadController, ExpertProfileEntry
from minisgl.moe.storage import ExpertStorage, ExpertStorageCatalog
from minisgl.scheduler.expert_cache import ExpertCacheManager


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
    blocks = cache._allocate(5)
    assert events == ["cpu"]
    assert set(blocks.tolist()) == {0, 1, 3, 4, 5}  # Slot 2 is the dummy page.
    with pytest.raises(MemoryError):
        cache._allocate(1)
    if cache_type == "radix":
        cache.prefix_cache.insert_prefix(torch.arange(3, dtype=torch.int32), blocks[2:])
        cache._free(blocks[:2])
        assert events == ["cpu"]  # Prefix ownership keeps the expert on CPU.
        cache.check_integrity()
        cache._release(cache.prefix_cache.evict(3))
    else:
        with cache.lazy_free_region():
            cache._free(blocks[:3])
            cache._free(blocks[3:])
    assert events == ["cpu", "copy", "gpu"]
    cache.check_integrity()
    cache._free(cache._allocate(5))
    assert events == ["cpu", "copy", "gpu"] * 2
