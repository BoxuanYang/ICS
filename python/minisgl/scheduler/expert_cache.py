"""Existing prefix-cache semantics with expert-backed token allocations."""

from contextlib import contextmanager

import torch
from minisgl.utils import div_ceil, init_logger

from .cache import CacheManager, _write_page_table

logger = init_logger(__name__)


class ExpertEvictionRequired(RuntimeError):
    """The selected batch fits only after one or more new expert evictions."""

    def __init__(self, needed_pages: int):
        super().__init__(f"Batch needs {needed_pages} KV pages and requires expert eviction")
        self.needed_pages = needed_pages


class ExpertCapacityExhausted(RuntimeError):
    """The configured expert prefix cannot provide enough KV capacity."""


class ExpertCacheManager(CacheManager):
    def __init__(self, pool, manager, page_table, cache_type):
        super().__init__(pool.allocator.persistent_capacity_blocks, 1, page_table, cache_type)
        self.pool = pool
        self.manager = manager
        self.allocator = pool.allocator
        self.live = {}  # Global slot -> allocation reference (including its generation).
        # Temporary slots are never inserted into the radix cache, so each has
        # exactly one page-table owner and remains in place until final release.
        self.temporary_locations: dict[int, tuple[int, int]] = {}
        self.temporary_counts: dict[int, int] = {}
        # Requests that touch expert-backed storage must release that storage at
        # completion instead of leaving it owned by the radix prefix cache.
        self.uncached_reqs: set[int] = set()

    @property
    def available_size(self):
        available = self.allocator.available_blocks + self.prefix_cache.size_info.evictable_size
        controller = self.manager.controller
        if not controller.check_stop_moe_eviction():
            available += self.manager.admission_eviction_capacity_blocks
        # Admission reserves future decode space; actual eviction occurs in _allocate.
        return available

    def _allocate(self, needed_pages):
        # The normal overlap path may reclaim idle radix entries, but it must
        # never synchronize the engine or evict a new expert.  Escalate that
        # decision to Scheduler.overlap_loop, which can first retire the
        # previous batch and recompute the shortfall.
        available_with_radix = (
            self.allocator.available_blocks
            + self.prefix_cache.size_info.evictable_size
        )
        if needed_pages > available_with_radix:
            raise ExpertEvictionRequired(needed_pages)
        missing = needed_pages - self.allocator.available_blocks
        if missing > 0:
            self._release(self.prefix_cache.evict(missing))
        blocks = self.allocator.allocate(needed_pages)
        self._last_allocation_temporary = tuple(
            block.region_id != self.allocator.PERSISTENT_REGION_ID for block in blocks
        )
        ids = [self.pool.slot_id(block) for block in blocks]
        self._last_allocation_ids = tuple(ids)
        self.live.update(zip(ids, blocks))
        return torch.tensor(ids, dtype=torch.int32, device=self.device)

    def evict_for_batch(self, needed_pages: int) -> int:
        """Slow path: reclaim completed work, then evict the coldest prefix."""
        missing = needed_pages - self.allocator.available_blocks
        if missing > 0:
            evictable = self.prefix_cache.size_info.evictable_size
            if evictable:
                self._release(self.prefix_cache.evict(min(missing, evictable)))
        if needed_pages <= self.allocator.available_blocks:
            return 0

        # Only the slow path may wait for copies. Finish FETCHING before
        # rebuilding the coldest-first plan so a colder fetching expert is
        # never skipped in favour of a warmer GPU-resident candidate.
        for event in self.manager.fetch_events.values():
            event.synchronize()
        self.manager.poll_restorations()
        plan = self.manager.plan_evictions(needed_pages)
        if plan.experts:
            self.manager.evict_many(tuple(entry.key for entry in plan.experts))
            for entry in plan.experts:
                logger.info_rank0(
                    "KV-MoE evict layer=%d expert=%d frequency=%.9f",
                    entry.layer_id,
                    entry.expert_id,
                    entry.activation_frequency,
                )
        if plan.remaining_shortfall_blocks:
            raise ExpertCapacityExhausted(
                f"Batch needs {needed_pages} KV pages; configured expert prefix "
                f"still lacks {plan.remaining_shortfall_blocks} pages"
            )
        return len(plan.experts)

    def fit_decode_reqs(self, reqs):
        """Return the largest ordered decode subset that fits current space."""
        budget = (
            self.allocator.available_blocks
            + self.prefix_cache.size_info.evictable_size
        )
        selected = []
        used = 0
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            cost = max(0, last_page - first_page)
            if used + cost > budget:
                continue
            selected.append(req)
            used += cost
        return selected

    def allocate_paged(self, reqs) -> None:
        needed_pages = 0
        allocation_info = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req, req.table_idx, first_page, last_page))
        if needed_pages == 0:
            self.manager.flush_pending_restorations()
            return

        allocated = self._page_to_token(self._allocate(needed_pages))
        flags = self._last_allocation_temporary
        ids = self._last_allocation_ids
        offset = 0
        for req, table_idx, first_page, last_page in allocation_info:
            count = last_page - first_page
            if any(flags[offset : offset + count]):
                self.uncached_reqs.add(req.uid)
            for index in range(count):
                allocation_index = offset + index
                if flags[allocation_index]:
                    self.temporary_locations[ids[allocation_index]] = (
                        table_idx,
                        first_page + index,
                    )
                    self.temporary_counts[table_idx] = (
                        self.temporary_counts.get(table_idx, 0) + 1
                    )
            offset += count
        _write_page_table(
            self.page_table,
            allocated,
            [(table_idx, first, last) for _, table_idx, first, last in allocation_info],
            self.page_size,
        )
        # Releases enqueue restoration candidates.  Only fetch experts that the
        # just-completed allocation did not reuse.
        self.manager.flush_pending_restorations()

    def _release(self, indices):
        ids = indices.tolist()
        if not ids:
            return
        blocks = [self.live[slot] for slot in ids]
        for slot in ids:
            location = self.temporary_locations.pop(slot, None)
            if location is not None:
                self._decrement_temporary_count(location[0])
        self.allocator.release(blocks)  # Empty experts become deferred candidates.
        for slot in ids:
            del self.live[slot]

    def _free(self, indices):
        self._release(indices)

    def cache_req(self, req, *, finished: bool) -> None:
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        if req.uid not in self.uncached_reqs:
            return super().cache_req(req, finished=finished)

        if not finished:
            # Keep the prefix handle that was matched before admission locked.
            # All newly allocated tail slots remain owned directly by the req.
            return

        old_handle = req.cache_handle
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len :])
        self.uncached_reqs.discard(req.uid)

    def uses_temporary(self, req) -> bool:
        return self.temporary_counts.get(req.table_idx, 0) > 0

    def _decrement_temporary_count(self, table_idx: int) -> None:
        remaining = self.temporary_counts[table_idx] - 1
        if remaining:
            self.temporary_counts[table_idx] = remaining
        else:
            del self.temporary_counts[table_idx]

    @contextmanager
    def lazy_free_region(self, *, flush_restorations: bool = True):
        # Batch releases so an expert shared by several requests restores once.
        pending = []
        self._free = pending.append
        try:
            yield
        finally:
            del self._free
            if pending:
                self._release(torch.cat(pending))
            if flush_restorations:
                self.manager.flush_pending_restorations()

    def flush_pending_restorations(self) -> None:
        self.manager.flush_pending_restorations()

    def poll_restorations(self) -> int:
        return self.manager.poll_restorations()

    def check_integrity(self):
        self.prefix_cache.check_integrity()
        if len(self.live) != self.prefix_cache.size_info.total_size:
            raise RuntimeError("Live KV slots do not match idle prefix-cache ownership")
