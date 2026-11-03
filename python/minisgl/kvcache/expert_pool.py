"""KV address table over persistent storage and unchanged expert tensor addresses."""

from __future__ import annotations

import torch

from .mha_pool import MHAKVCache
from .unified import UnifiedKVAllocator


class ExpertKVCache(MHAKVCache):
    def __init__(self, config, num_pages: int, device, catalog):
        super().__init__(
            config.num_kv_heads,
            config.num_layers,
            config.head_dim,
            num_pages + 1,
            1,
            torch.bfloat16,
            device,
        )
        self.heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.layer_elements = self.heads * self.head_dim
        # One allocation owns a token's K and V for EVERY layer.
        self.block_nbytes = 2 * config.num_layers * self.layer_elements * 2
        self.allocator = UnifiedKVAllocator(num_pages, self.block_nbytes)
        catalog.create_and_register_regions(self.block_nbytes, self.allocator)
        self.region_start = {"persistent": 0}
        self.region_views = []  # Keep the aliases (and their original storages) alive.
        total = num_pages + 1  # Persistent dummy slot is never allocated.
        for storage in catalog:
            for region in (storage.gate_up_region, storage.down_region):
                self.region_start[region.region_id] = total
                total += region.capacity_blocks
        self.total_slots = total
        addresses = torch.empty((2, config.num_layers, total), dtype=torch.int64)
        offsets = torch.arange(num_pages + 1, dtype=torch.int64) * self.layer_elements * 2
        for layer in range(config.num_layers):
            addresses[0, layer, : num_pages + 1] = self.k_cache(layer).data_ptr() + offsets
            addresses[1, layer, : num_pages + 1] = self.v_cache(layer).data_ptr() + offsets
        for storage in catalog:
            for region in (storage.gate_up_region, storage.down_region):
                tensor = region.storage_view
                if not tensor.is_contiguous() or tensor.dtype != torch.bfloat16:
                    raise ValueError("Expert KV reuse requires contiguous BF16 slices")
                count = region.capacity_blocks
                view = tensor.view(-1)[: count * self.block_nbytes // 2].view(
                    count, 2, config.num_layers, self.heads, self.head_dim
                )
                self.region_views.append(view)
                start = self.region_start[region.region_id]
                token_offsets = torch.arange(count, dtype=torch.int64) * self.block_nbytes
                for kind in range(2):
                    for layer in range(config.num_layers):
                        addresses[kind, layer, start : start + count] = (
                            tensor.data_ptr()
                            + token_offsets
                            + (kind * config.num_layers + layer) * self.layer_elements * 2
                        )
        self.addresses = addresses.to(device)

    def slot_id(self, block) -> int:
        return self.region_start[block.region_id] + block.local_block_id

    def store_kv(self, k, v, out_loc, layer_id: int) -> None:
        from minisgl.kernel.triton.expert_attention import store_kv

        store_kv(k, v, out_loc, self.addresses[:, layer_id], self.layer_elements)
