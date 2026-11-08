from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Protocol

from minisgl.kvcache.unified import TemporaryKVRegion, TemporaryKVRegionState, UnifiedKVAllocator

from .offload import (
    EvictionPlan,
    ExpertKey,
    ExpertOffloadController,
    ExpertResidencyState,
    KVPressureSnapshot,
)
from .storage import ExpertStorageCatalog


class ExpertMemoryBackend(Protocol):
    """Execution/transfer boundary. No CUDA or KT work is simulated here."""

    def route_to_cpu(self, key: ExpertKey) -> None:
        """Update the expert mask so subsequent forwards use KT."""
        ...

    def restore_weights(self, key: ExpertKey) -> Any:
        """Launch both weight copies and return an event-like completion token."""
        ...

    def route_to_gpu(self, key: ExpertKey) -> None: ...


class ExpertMemoryManager:
    """Coordinates expert residency with ownership of its two KV regions."""

    def __init__(
        self,
        controller: ExpertOffloadController,
        storage_catalog: ExpertStorageCatalog,
        allocator: UnifiedKVAllocator,
        backend: ExpertMemoryBackend,
        transition_barrier: Callable[[], None] | None = None,
        max_offloaded_experts: int | None = None,
        restore_headroom_blocks: int = 0,
        restore_stability_flushes: int = 1,
    ) -> None:
        catalog_keys = frozenset(storage.key for storage in storage_catalog)
        if catalog_keys != controller.profile.allowed_keys:
            missing = controller.profile.allowed_keys - catalog_keys
            extra = catalog_keys - controller.profile.allowed_keys
            raise ValueError(
                "Storage catalog and eviction profile differ: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        self.controller = controller
        self.storage_catalog = storage_catalog
        self.allocator = allocator
        self.backend = backend
        self.transition_barrier = transition_barrier
        if max_offloaded_experts is not None and max_offloaded_experts < 1:
            raise ValueError("max_offloaded_experts must be positive")
        if restore_headroom_blocks < 0:
            raise ValueError("restore_headroom_blocks must be non-negative")
        if restore_stability_flushes < 1:
            raise ValueError("restore_stability_flushes must be positive")
        self.max_offloaded_experts = max_offloaded_experts
        self.restore_headroom_blocks = restore_headroom_blocks
        self.restore_stability_flushes = restore_stability_flushes
        self._restore_headroom_streak = 0
        self.pending_restorations: set[ExpertKey] = set()
        self.fetch_events: dict[ExpertKey, Any] = {}
        for storage in storage_catalog:
            for region in self._regions(storage.key):
                try:
                    registered = allocator.region(region.region_id)
                except KeyError as exc:
                    raise ValueError(
                        f"KV region {region.region_id} for expert {storage.key} is not registered"
                    ) from exc
                if registered is not region:
                    raise ValueError(
                        f"KV region id {region.region_id} is registered to a different object"
                    )
        if allocator.on_expert_release is not None:
            raise ValueError("Allocator already has an expert release owner")
        allocator.on_expert_release = self._restore_if_empty
        self.available_gpu_capacity_blocks = sum(self.capacity_blocks.values())

    @property
    def admission_eviction_capacity_blocks(self) -> int:
        """GPU storage that admission may safely promise to the next batch.

        Advertising every candidate at once lets prefill admit an enormous burst
        and forces hundreds of experts onto the CPU before the scheduler can
        observe the resulting execution cost.  Keep a bounded sliding window;
        restored experts reopen the window later.
        """
        active = self.controller.residency.keys_in(
            ExpertResidencyState.EVICTING,
            ExpertResidencyState.CPU_RESIDENT,
            ExpertResidencyState.RESTORING,
        )
        remaining = (
            len(self.controller.profile.entries)
            if self.max_offloaded_experts is None
            else max(0, self.max_offloaded_experts - len(active))
        )
        if remaining == 0:
            return 0
        capacity = 0
        for entry in self.controller.profile.entries:
            if self.controller.residency.state(entry.key) is not ExpertResidencyState.GPU_RESIDENT:
                continue
            capacity += self.capacity_blocks[entry.key]
            remaining -= 1
            if remaining == 0:
                break
        return capacity

    @cached_property
    def capacity_blocks(self) -> dict[ExpertKey, int]:
        result: dict[ExpertKey, int] = {}
        for storage in self.storage_catalog:
            gate_up, down = self._regions(storage.key)
            result[storage.key] = gate_up.capacity_blocks + down.capacity_blocks
        return result

    def plan_evictions(self, upcoming_demand_blocks: int) -> EvictionPlan:
        pressure = KVPressureSnapshot(
            persistent_available_blocks=self.allocator.persistent_available_blocks,
            temporary_available_blocks=self.allocator.temporary_available_blocks,
            upcoming_demand_blocks=upcoming_demand_blocks,
        )
        active = self.controller.residency.keys_in(
            ExpertResidencyState.EVICTING,
            ExpertResidencyState.CPU_RESIDENT,
            ExpertResidencyState.RESTORING,
        )
        remaining = (
            len(self.controller.profile.entries)
            if self.max_offloaded_experts is None
            else max(0, self.max_offloaded_experts - len(active))
        )
        limited_capacity: dict[ExpertKey, int] = {}
        for entry in self.controller.profile.entries:
            if remaining == 0:
                break
            if self.controller.residency.state(entry.key) is not ExpertResidencyState.GPU_RESIDENT:
                continue
            limited_capacity[entry.key] = self.capacity_blocks[entry.key]
            remaining -= 1
        return self.controller.plan_evictions(pressure, limited_capacity)

    def begin_eviction(self, key: ExpertKey) -> None:
        self.pending_restorations.discard(key)
        gate_up, down = self._regions(key)
        self._require_region_states(
            key,
            (gate_up, down),
            TemporaryKVRegionState.INACTIVE,
        )
        if self.transition_barrier is not None:
            self.transition_barrier()
        self.controller.begin_eviction(key)
        # Switch subsequent execution immediately. Storage reuse is a separate
        # operation: the caller must order its first KV write after old readers.
        self.backend.route_to_cpu(key)

    def evict_many(self, keys: tuple[ExpertKey, ...]) -> None:
        """Synchronize once, then expose a JSON-ordered eviction prefix."""
        if not keys:
            return
        if self.transition_barrier is not None:
            self.transition_barrier()
        barrier = self.transition_barrier
        self.transition_barrier = None
        try:
            for key in keys:
                self.begin_eviction(key)
                self.complete_eviction(key)
        finally:
            self.transition_barrier = barrier

    def complete_eviction(self, key: ExpertKey) -> None:
        """Expose storage after old weight reads are ordered before KV writes."""
        if self.controller.residency.state(key) is not ExpertResidencyState.EVICTING:
            raise RuntimeError(f"Expert {key} is not being evicted")
        gate_up, down = self._regions(key)
        self._require_region_states(
            key,
            (gate_up, down),
            TemporaryKVRegionState.INACTIVE,
        )
        gate_up.activate()
        down.activate()
        self.controller.complete_eviction(key)
        self.available_gpu_capacity_blocks -= self.capacity_blocks[key]
        # Also cover evictions followed by a deferred/empty batch. Allocation
        # removes eligibility via live_blocks before the next flush.
        self.pending_restorations.add(key)

    def _restore_if_empty(self, key: ExpertKey) -> None:
        """Mark an empty expert for restoration after the allocation phase."""
        gate_up, down = self._regions(key)
        if (
            self.controller.residency.state(key) is ExpertResidencyState.CPU_RESIDENT
            and gate_up.live_blocks == 0
            and down.live_blocks == 0
        ):
            self.pending_restorations.add(key)

    def flush_pending_restorations(self) -> int:
        """Launch asynchronous fetches that stayed unused by the next batch.

        Release callbacks only enqueue work.  The scheduler flushes after it has
        allocated the next token/batch, so a buffer needed immediately is reused
        instead of being fetched and evicted again.
        """
        pending = sorted(self.pending_restorations)
        self.pending_restorations.clear()
        launched = 0
        for key in pending:
            gate_up, down = self._regions(key)
            if (
                self.controller.residency.state(key) is ExpertResidencyState.CPU_RESIDENT
                and gate_up.live_blocks == 0
                and down.live_blocks == 0
            ):
                self._restore_one(key)
                launched += 1
        return launched

    def _restore_one(self, key: ExpertKey) -> None:
        gate_up, down = self._regions(key)
        self._require_region_states(
            key,
            (gate_up, down),
            TemporaryKVRegionState.ACTIVE,
        )
        # The expert remains CPU-routed until the event completes. RECLAIMED
        # regions are unavailable to the allocator throughout the copy.
        gate_up.reclaim()
        down.reclaim()
        self.controller.begin_restoring(key)
        event = self.backend.restore_weights(key)
        if event is None:
            # Lightweight test backends may implement an immediate copy.
            self.complete_restoration(key)
        else:
            self.fetch_events[key] = event

    def poll_restorations(self) -> int:
        """Commit completed copies at a scheduler batch boundary."""
        keys = sorted(self.fetch_events)
        if not keys:
            return 0
        ready = [self.fetch_events[key].query() for key in keys]
        agree = getattr(self.backend, "agree_fetch_completion", None)
        if agree is not None:
            ready = agree(ready)
        completed = [key for key, done in zip(keys, ready) if done]
        for key in completed:
            del self.fetch_events[key]
            self.complete_restoration(key)
        return len(completed)

    def complete_restoration(self, key: ExpertKey) -> None:
        if self.controller.residency.state(key) is not ExpertResidencyState.RESTORING:
            raise RuntimeError(f"Expert {key} weights are not being restored")
        gate_up, down = self._regions(key)
        self._require_region_states(
            key,
            (gate_up, down),
            TemporaryKVRegionState.RECLAIMED,
        )
        self.backend.route_to_gpu(key)
        gate_up.deactivate()
        down.deactivate()
        self.controller.complete_restoration(key)
        self.available_gpu_capacity_blocks += self.capacity_blocks[key]

    def _regions(self, key: ExpertKey) -> tuple[TemporaryKVRegion, TemporaryKVRegion]:
        storage = self.storage_catalog[key]
        if storage.gate_up_region is None or storage.down_region is None:
            raise RuntimeError(f"Temporary KV regions have not been created for expert {key}")
        return storage.gate_up_region, storage.down_region

    @staticmethod
    def _require_region_states(
        key: ExpertKey,
        regions: tuple[TemporaryKVRegion, TemporaryKVRegion],
        expected: TemporaryKVRegionState,
    ) -> None:
        actual = tuple(region.state for region in regions)
        if any(state is not expected for state in actual):
            names = tuple(state.name for state in actual)
            raise RuntimeError(f"Expert {key} regions must both be {expected.name}, found {names}")
