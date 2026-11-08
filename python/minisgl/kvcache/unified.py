from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterable, TypeAlias

ExpertKey: TypeAlias = tuple[int, int]
RegionId: TypeAlias = str


class TemporaryKVRegionState(Enum):
    INACTIVE = auto()
    ACTIVE = auto()
    RECLAIMED = auto()


@dataclass(frozen=True, slots=True)
class KVBlockRef:
    region_id: RegionId
    local_block_id: int
    byte_offset: int
    block_nbytes: int
    allocation_id: int


class TemporaryKVRegion:
    """Logical block allocator over one contiguous expert tensor slice."""

    def __init__(
        self,
        region_id: RegionId,
        owner_expert: ExpertKey | None,
        storage_nbytes: int,
        block_nbytes: int,
        *,
        persistent: bool = False,
        storage_view: object | None = None,
    ) -> None:
        if not region_id:
            raise ValueError("region_id must not be empty")
        if isinstance(storage_nbytes, bool) or not isinstance(storage_nbytes, int):
            raise TypeError("storage_nbytes must be an integer")
        if isinstance(block_nbytes, bool) or not isinstance(block_nbytes, int):
            raise TypeError("block_nbytes must be an integer")
        if storage_nbytes < 0:
            raise ValueError("storage_nbytes must be non-negative")
        if block_nbytes <= 0:
            raise ValueError("block_nbytes must be positive")
        if persistent and owner_expert is not None:
            raise ValueError("A persistent region cannot be owned by an expert")
        if not persistent and owner_expert is None:
            raise ValueError("A temporary region must identify its owner expert")

        self.region_id = region_id
        self.owner_expert = owner_expert
        self.storage_nbytes = storage_nbytes
        self.block_nbytes = block_nbytes
        self.capacity_blocks = storage_nbytes // block_nbytes
        self.tail_nbytes = storage_nbytes % block_nbytes
        self.persistent = persistent
        self.storage_view = storage_view
        self.state = (
            TemporaryKVRegionState.ACTIVE if persistent else TemporaryKVRegionState.INACTIVE
        )
        self._free_ids = deque(range(self.capacity_blocks))
        self._live_ids: dict[int, int] = {}
        self._allocation_serial = 0

    @property
    def free_blocks(self) -> int:
        return len(self._free_ids) if self.state is TemporaryKVRegionState.ACTIVE else 0

    @property
    def live_blocks(self) -> int:
        return len(self._live_ids)

    def activate(self) -> None:
        if self.persistent:
            raise RuntimeError("The persistent KV region is always active")
        if self.state not in {
            TemporaryKVRegionState.INACTIVE,
        }:
            raise RuntimeError(f"Cannot activate region {self.region_id} from {self.state.name}")
        self.state = TemporaryKVRegionState.ACTIVE

    def reclaim(self) -> None:
        """Reserve an empty region for weight restoration, after GPU access is ordered."""
        if self.persistent:
            raise RuntimeError("The persistent KV region cannot be reclaimed")
        if self.state is not TemporaryKVRegionState.ACTIVE:
            raise RuntimeError(f"Cannot reclaim region {self.region_id} from {self.state.name}")
        if self.live_blocks:
            raise RuntimeError(f"Region {self.region_id} still has live KV blocks")
        self.state = TemporaryKVRegionState.RECLAIMED

    def deactivate(self) -> None:
        if self.persistent:
            raise RuntimeError("The persistent KV region cannot be deactivated")
        if self.state is not TemporaryKVRegionState.RECLAIMED:
            raise RuntimeError(
                f"Cannot return region {self.region_id} to expert storage from {self.state.name}"
            )
        self.state = TemporaryKVRegionState.INACTIVE

    def allocate(self, count: int) -> tuple[KVBlockRef, ...]:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Block count must be an integer")
        if count < 0:
            raise ValueError("Block count must be non-negative")
        if self.state is not TemporaryKVRegionState.ACTIVE:
            raise RuntimeError(f"Region {self.region_id} is not accepting allocations")
        if count > len(self._free_ids):
            raise MemoryError(
                f"Region {self.region_id} has {len(self._free_ids)} free blocks, "
                f"but {count} were requested"
            )

        blocks: list[KVBlockRef] = []
        for _ in range(count):
            local_block_id = self._free_ids.popleft()
            self._allocation_serial += 1
            self._live_ids[local_block_id] = self._allocation_serial
            blocks.append(
                KVBlockRef(
                    region_id=self.region_id,
                    local_block_id=local_block_id,
                    byte_offset=local_block_id * self.block_nbytes,
                    block_nbytes=self.block_nbytes,
                    allocation_id=self._allocation_serial,
                )
            )
        return tuple(blocks)

    def release(self, blocks: Iterable[KVBlockRef]) -> None:
        blocks = tuple(blocks)
        self.validate_release(blocks)
        for block in blocks:
            del self._live_ids[block.local_block_id]
            self._free_ids.append(block.local_block_id)

    def validate_release(self, blocks: Iterable[KVBlockRef]) -> None:
        seen: set[int] = set()
        for block in blocks:
            if block.region_id != self.region_id:
                raise ValueError(f"Block belongs to region {block.region_id}, not {self.region_id}")
            if block.local_block_id not in self._live_ids:
                raise ValueError(
                    f"Block {block.local_block_id} is not live in region {self.region_id}"
                )
            if block.local_block_id in seen:
                raise ValueError("Duplicate block in release batch")
            seen.add(block.local_block_id)
            if self._live_ids[block.local_block_id] != block.allocation_id:
                raise ValueError("Stale block from a previous allocation")
            expected_offset = block.local_block_id * self.block_nbytes
            if block.byte_offset != expected_offset or block.block_nbytes != self.block_nbytes:
                raise ValueError(f"Block metadata does not match region {self.region_id}")


class UnifiedKVAllocator:
    """Allocates logical KV blocks from persistent, then temporary regions."""

    PERSISTENT_REGION_ID = "persistent"

    def __init__(self, persistent_capacity_blocks: int, block_nbytes: int) -> None:
        if isinstance(persistent_capacity_blocks, bool) or not isinstance(
            persistent_capacity_blocks, int
        ):
            raise TypeError("persistent_capacity_blocks must be an integer")
        if persistent_capacity_blocks < 0:
            raise ValueError("persistent_capacity_blocks must be non-negative")
        self.persistent_region = TemporaryKVRegion(
            region_id=self.PERSISTENT_REGION_ID,
            owner_expert=None,
            storage_nbytes=persistent_capacity_blocks * block_nbytes,
            block_nbytes=block_nbytes,
            persistent=True,
        )
        self._temporary_regions: dict[RegionId, TemporaryKVRegion] = {}
        self.on_expert_release: Callable[[ExpertKey], None] | None = None

    @property
    def persistent_capacity_blocks(self) -> int:
        return self.persistent_region.capacity_blocks

    @property
    def persistent_available_blocks(self) -> int:
        return self.persistent_region.free_blocks

    @property
    def temporary_capacity_blocks(self) -> int:
        return sum(
            region.capacity_blocks
            for region in self._temporary_regions.values()
            if region.state is not TemporaryKVRegionState.INACTIVE
        )

    @property
    def temporary_available_blocks(self) -> int:
        return sum(region.free_blocks for region in self._temporary_regions.values())

    @property
    def available_blocks(self) -> int:
        return self.persistent_available_blocks + self.temporary_available_blocks

    def register_temporary_region(self, region: TemporaryKVRegion) -> None:
        if region.persistent:
            raise ValueError("Only temporary regions can be registered")
        if region.region_id == self.PERSISTENT_REGION_ID:
            raise ValueError(f"Region id {region.region_id!r} is reserved")
        if region.region_id in self._temporary_regions:
            raise ValueError(f"Region id already registered: {region.region_id}")
        if region.block_nbytes != self.persistent_region.block_nbytes:
            raise ValueError("All regions must use the same KV allocation unit")
        self._temporary_regions[region.region_id] = region

    def region(self, region_id: RegionId) -> TemporaryKVRegion:
        if region_id == self.PERSISTENT_REGION_ID:
            return self.persistent_region
        try:
            return self._temporary_regions[region_id]
        except KeyError as exc:
            raise KeyError(f"Unknown KV region: {region_id}") from exc

    def allocate(self, count: int) -> tuple[KVBlockRef, ...]:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Block count must be an integer")
        if count < 0:
            raise ValueError("Block count must be non-negative")
        if count > self.available_blocks:
            raise MemoryError(
                f"Unified KV allocator has {self.available_blocks} free blocks, "
                f"but {count} were requested"
            )

        remaining = count
        allocated: list[KVBlockRef] = []
        regions = (self.persistent_region, *self._temporary_regions.values())
        for region in regions:
            take = min(remaining, region.free_blocks)
            if take:
                allocated.extend(region.allocate(take))
                remaining -= take
            if remaining == 0:
                break
        return tuple(allocated)

    def allocate_persistent(self, count: int) -> tuple[KVBlockRef, ...]:
        """Allocate only from the normal KV pool.

        The expert-cache scheduler uses this to keep ordinary requests entirely
        on the contiguous FlashInfer-compatible pool.  Falling through into an
        expert region here would turn every row in a decode batch into a
        pointer-attention row.
        """
        return self.persistent_region.allocate(count)

    def allocate_temporary(self, count: int) -> tuple[KVBlockRef, ...]:
        """Allocate only from active expert-backed KV regions."""
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Block count must be an integer")
        if count < 0:
            raise ValueError("Block count must be non-negative")
        if count > self.temporary_available_blocks:
            raise MemoryError(
                f"Temporary KV regions have {self.temporary_available_blocks} "
                f"free blocks, but {count} were requested"
            )

        remaining = count
        allocated: list[KVBlockRef] = []
        for region in self._temporary_regions.values():
            take = min(remaining, region.free_blocks)
            if take:
                allocated.extend(region.allocate(take))
                remaining -= take
            if remaining == 0:
                break
        return tuple(allocated)

    def release(self, blocks: Iterable[KVBlockRef]) -> None:
        """Release final KV ownership, after ordering GPU accesses in the caller.

        Prefix-cache references must also be gone. Request completion alone is
        insufficient. Callbacks run after the whole release batch, before any
        subsequent allocation on this single-threaded allocator.
        """
        grouped: dict[RegionId, list[KVBlockRef]] = {}
        for block in blocks:
            grouped.setdefault(block.region_id, []).append(block)
        for region_id, region_blocks in grouped.items():
            self.region(region_id).validate_release(region_blocks)
        for region_id, region_blocks in grouped.items():
            self.region(region_id).release(region_blocks)
        owners = dict.fromkeys(self.region(region_id).owner_expert for region_id in grouped)
        if self.on_expert_release is not None:
            for owner in owners:
                if owner is not None:
                    self.on_expert_release(owner)
