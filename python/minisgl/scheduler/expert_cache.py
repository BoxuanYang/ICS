"""Existing prefix-cache semantics with expert-backed token allocations."""

from contextlib import contextmanager

import torch
from minisgl.moe.offload import ExpertResidencyState
from minisgl.utils import init_logger

from .cache import CacheManager

logger = init_logger(__name__)


class ExpertCacheManager(CacheManager):
    def __init__(self, pool, manager, page_table, cache_type):
        super().__init__(pool.allocator.persistent_capacity_blocks, 1, page_table, cache_type)
        self.pool = pool
        self.manager = manager
        self.allocator = pool.allocator
        self.live = {}  # Global slot -> allocation reference (including its generation).

    @property
    def available_size(self):
        available = self.allocator.available_blocks + self.prefix_cache.size_info.evictable_size
        controller = self.manager.controller
        if not controller.check_stop_moe_eviction():
            available += sum(
                capacity
                for key, capacity in self.manager.capacity_blocks.items()
                if controller.residency.state(key) is ExpertResidencyState.GPU_RESIDENT
            )
        # Admission reserves future decode space; actual eviction occurs in _allocate.
        return available

    def _allocate(self, needed_pages):
        missing = needed_pages - self.allocator.available_blocks
        if missing > 0:
            evictable = self.prefix_cache.size_info.evictable_size
            if evictable:
                self._release(self.prefix_cache.evict(min(missing, evictable)))
        while needed_pages > self.allocator.available_blocks:
            plan = self.manager.plan_evictions(needed_pages)
            if not plan.experts:
                raise MemoryError("KV demand exceeds the allowed expert candidates")
            key = plan.experts[0].key
            # This mode runs between completed batches on the engine stream.
            # route_to_cpu happens before storage is exposed for KV writes.
            self.manager.begin_eviction(key)
            self.manager.complete_eviction(key)
            logger.info_rank0("KV-MoE evict layer=%d expert=%d", *key)
        blocks = self.allocator.allocate(needed_pages)
        ids = [self.pool.slot_id(block) for block in blocks]
        self.live.update(zip(ids, blocks))
        return torch.tensor(ids, dtype=torch.int32, device=self.device)

    def _release(self, indices):
        ids = indices.tolist()
        if not ids:
            return
        blocks = [self.live[slot] for slot in ids]
        self.allocator.release(blocks)  # Eager fetch-back runs inside this release.
        for slot in ids:
            del self.live[slot]

    def _free(self, indices):
        self._release(indices)

    @contextmanager
    def lazy_free_region(self):
        # Batch releases so an expert shared by several requests restores once.
        pending = []
        self._free = pending.append
        try:
            yield
        finally:
            del self._free
            if pending:
                self._release(torch.cat(pending))

    def check_integrity(self):
        self.prefix_cache.check_integrity()
        if len(self.live) != self.prefix_cache.size_info.total_size:
            raise RuntimeError("Live KV slots do not match idle prefix-cache ownership")
