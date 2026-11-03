from __future__ import annotations

import pytest
from minisgl.kvcache.unified import UnifiedKVAllocator
from minisgl.moe.offload import ExpertEvictionProfile, ExpertProfileEntry
from minisgl.moe.storage import ExpertStorageCatalog


class FakeTensor:
    def __init__(self, elements: int, item_size: int = 2) -> None:
        self.elements = elements
        self.item_size = item_size

    def numel(self) -> int:
        return self.elements

    def element_size(self) -> int:
        return self.item_size


class FakeMoELayer:
    def __init__(self, layer_id: int, num_experts: int) -> None:
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.gate_up = [FakeTensor(30 + expert_id) for expert_id in range(num_experts)]
        self.down = [FakeTensor(20 + expert_id) for expert_id in range(num_experts)]

    def expert_tensor_views(self, expert_id: int) -> tuple[FakeTensor, FakeTensor]:
        if not 0 <= expert_id < self.num_experts:
            raise IndexError(expert_id)
        return self.gate_up[expert_id], self.down[expert_id]


def test_catalog_uses_profile_keys_and_tensor_slices() -> None:
    profile = ExpertEvictionProfile(
        [
            ExpertProfileEntry(0, 1, 0.01),
            ExpertProfileEntry(1, 0, 0.02),
        ]
    )

    catalog = ExpertStorageCatalog.from_layers([FakeMoELayer(0, 2), FakeMoELayer(1, 1)], profile)

    assert len(catalog) == 2
    assert catalog[(0, 1)].gate_up_tensor is not None
    assert catalog[(0, 1)].gate_up_nbytes == 62
    assert catalog[(0, 1)].down_nbytes == 42
    assert catalog[(1, 0)].activation_frequency == 0.02


def test_catalog_creates_two_regions_per_expert_and_preserves_tail() -> None:
    profile = ExpertEvictionProfile([ExpertProfileEntry(0, 0, 0.01)])
    catalog = ExpertStorageCatalog.from_layers([FakeMoELayer(0, 1)], profile)
    allocator = UnifiedKVAllocator(persistent_capacity_blocks=0, block_nbytes=16)

    catalog.create_and_register_regions(block_nbytes=16, allocator=allocator)
    storage = catalog[(0, 0)]

    assert storage.gate_up_region is not None
    assert storage.down_region is not None
    assert storage.gate_up_region.capacity_blocks == 3
    assert storage.gate_up_region.tail_nbytes == 12
    assert storage.gate_up_region.storage_view is storage.gate_up_tensor
    assert storage.down_region.capacity_blocks == 2
    assert storage.down_region.tail_nbytes == 8
    assert storage.down_region.storage_view is storage.down_tensor


def test_catalog_rejects_profile_layer_or_expert_missing_from_model() -> None:
    missing_layer = ExpertEvictionProfile([ExpertProfileEntry(3, 0, 0.01)])
    with pytest.raises(ValueError, match="missing MoE layer 3"):
        ExpertStorageCatalog.from_layers([FakeMoELayer(0, 1)], missing_layer)

    missing_expert = ExpertEvictionProfile([ExpertProfileEntry(0, 4, 0.01)])
    with pytest.raises(ValueError, match=r"missing expert \(0, 4\)"):
        ExpertStorageCatalog.from_layers([FakeMoELayer(0, 1)], missing_expert)
