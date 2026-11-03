from __future__ import annotations

import pytest
from minisgl.kvcache.unified import (
    TemporaryKVRegion,
    TemporaryKVRegionState,
    UnifiedKVAllocator,
)


def _temporary(region_id: str, blocks: int, block_nbytes: int = 16) -> TemporaryKVRegion:
    return TemporaryKVRegion(
        region_id=region_id,
        owner_expert=(0, int(region_id[-1])),
        storage_nbytes=blocks * block_nbytes,
        block_nbytes=block_nbytes,
    )


def test_region_capacity_uses_only_complete_blocks() -> None:
    region = TemporaryKVRegion(
        region_id="expert-0-gate-up",
        owner_expert=(0, 0),
        storage_nbytes=75,
        block_nbytes=16,
    )

    assert region.capacity_blocks == 4
    assert region.tail_nbytes == 11
    assert region.free_blocks == 0


def test_unified_allocator_prefers_persistent_then_temporary_regions() -> None:
    allocator = UnifiedKVAllocator(persistent_capacity_blocks=2, block_nbytes=16)
    first = _temporary("temporary-0", blocks=2)
    second = _temporary("temporary-1", blocks=2)
    allocator.register_temporary_region(first)
    allocator.register_temporary_region(second)
    first.activate()
    second.activate()

    blocks = allocator.allocate(5)

    assert [block.region_id for block in blocks] == [
        UnifiedKVAllocator.PERSISTENT_REGION_ID,
        UnifiedKVAllocator.PERSISTENT_REGION_ID,
        "temporary-0",
        "temporary-0",
        "temporary-1",
    ]
    assert allocator.available_blocks == 1


def test_reclaim_requires_zero_live_blocks() -> None:
    allocator = UnifiedKVAllocator(persistent_capacity_blocks=0, block_nbytes=16)
    region = _temporary("temporary-0", blocks=3)
    allocator.register_temporary_region(region)
    region.activate()
    blocks = allocator.allocate(2)

    with pytest.raises(RuntimeError, match="live KV blocks"):
        region.reclaim()
    assert region.state is TemporaryKVRegionState.ACTIVE
    assert region.free_blocks == 1

    allocator.release(blocks[:1])
    assert region.state is TemporaryKVRegionState.ACTIVE
    allocator.release(blocks[1:])
    region.reclaim()
    assert region.state is TemporaryKVRegionState.RECLAIMED
    assert region.live_blocks == 0

    region.deactivate()
    assert region.state is TemporaryKVRegionState.INACTIVE


def test_active_region_stays_allocatable_until_owner_reclaims() -> None:
    allocator = UnifiedKVAllocator(persistent_capacity_blocks=0, block_nbytes=16)
    region = _temporary("temporary-0", blocks=2)
    allocator.register_temporary_region(region)
    region.activate()
    live_block = allocator.allocate(1)

    assert region.state is TemporaryKVRegionState.ACTIVE
    assert region.free_blocks == 1
    allocator.release(live_block)
    assert region.state is TemporaryKVRegionState.ACTIVE


def test_failed_unified_allocation_is_atomic() -> None:
    allocator = UnifiedKVAllocator(persistent_capacity_blocks=1, block_nbytes=16)
    region = _temporary("temporary-0", blocks=1)
    allocator.register_temporary_region(region)
    region.activate()

    with pytest.raises(MemoryError, match="2 free blocks"):
        allocator.allocate(3)

    assert allocator.available_blocks == 2


def test_double_release_is_rejected() -> None:
    allocator = UnifiedKVAllocator(persistent_capacity_blocks=1, block_nbytes=16)
    block = allocator.allocate(1)
    allocator.release(block)

    with pytest.raises(ValueError, match="is not live"):
        allocator.release(block)


def test_stale_reference_cannot_release_a_reused_slot() -> None:
    allocator = UnifiedKVAllocator(1, 16)
    old = allocator.allocate(1)
    allocator.release(old)
    current = allocator.allocate(1)
    with pytest.raises(ValueError, match="Stale"):
        allocator.release(old)
    assert allocator.available_blocks == 0
    allocator.release(current)


def test_invalid_release_batch_does_not_partially_free_blocks() -> None:
    allocator = UnifiedKVAllocator(2, 16)
    blocks = allocator.allocate(2)
    with pytest.raises(ValueError, match="Duplicate"):
        allocator.release([blocks[0], blocks[0]])
    assert allocator.available_blocks == 0
    allocator.release(blocks)
