from __future__ import annotations

from typing import Callable, Protocol

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

    def restore_weights(self, key: ExpertKey) -> None:
        """Copy both original weights back, ordered after KV accesses.

        This synchronous reference interface returns only when the copy is done.
        Residency must remain CPU-side until then.
        """
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

    @property
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
        return self.controller.plan_evictions(pressure, self.capacity_blocks)

    def begin_eviction(self, key: ExpertKey) -> None:
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

    def _restore_if_empty(self, key: ExpertKey) -> None:
        """Eager restoration on final release, independent of global pressure."""
        if self.controller.residency.state(key) is not ExpertResidencyState.CPU_RESIDENT:
            return
        gate_up, down = self._regions(key)
        if gate_up.live_blocks or down.live_blocks:
            return
        self._require_region_states(
            key,
            (gate_up, down),
            TemporaryKVRegionState.ACTIVE,
        )
        if self.transition_barrier is not None:
            self.transition_barrier()
        gate_up.reclaim()
        down.reclaim()
        self.controller.begin_restoring(key)
        # Failure leaves both regions unavailable and execution on CPU.
        self.backend.restore_weights(key)
        self.complete_restoration(key)

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
