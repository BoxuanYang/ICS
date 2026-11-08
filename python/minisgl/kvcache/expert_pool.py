"""KV address table over persistent storage and unchanged expert tensor addresses."""

from __future__ import annotations

from functools import cached_property

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .mha_pool import MHAKVCache
from .unified import UnifiedKVAllocator


class ExpertKVCache(MHAKVCache):
    def __init__(self, config, num_pages: int, device, catalog):
        heads = div_even(config.num_kv_heads, get_tp_info().size, allow_replicate=True)
        super().__init__(
            # MHAKVCache performs the TP head sharding internally.  Passing the
            # already-local count here would shard twice and make the address
            # table stride past the actual persistent KV allocation.
            config.num_kv_heads,
            config.num_layers,
            config.head_dim,
            num_pages + 1,
            1,
            torch.bfloat16,
            device,
        )
        self.heads = heads
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
        self.active_slots = torch.zeros(total, dtype=torch.bool, device=device)
        self.active_slots[:num_pages + 1] = True
        self.expert_slot_ranges = {
            storage.key: [(self.region_start[region.region_id], region.capacity_blocks)
                          for region in (storage.gate_up_region, storage.down_region)]
            for storage in catalog
        }
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
                    2, config.num_layers, count, self.heads, self.head_dim
                )
                self.region_views.append(view)
                start = self.region_start[region.region_id]
                # The allocator owns all layers of a logical token, but those
                # rows need not be adjacent in physical memory. Layer-major
                # storage keeps an attention layer's expert KV reads contiguous.
                token_offsets = torch.arange(count, dtype=torch.int64) * self.layer_elements * 2
                for kind in range(2):
                    for layer in range(config.num_layers):
                        addresses[kind, layer, start : start + count] = (
                            tensor.data_ptr()
                            + token_offsets
                            + (kind * config.num_layers + layer) * count * self.layer_elements * 2
                        )
        self.addresses = addresses.to(device)

    def slot_id(self, block) -> int:
        return self.region_start[block.region_id] + block.local_block_id

    @cached_property
    def attention_workspace(self):
        # Only the compatibility path needs a layer of transient scratch KV.
        # The indirect FlashInfer path never allocates or gathers this workspace.
        return tuple(torch.empty(
            (self.total_slots, 1, self.heads, self.head_dim),
            dtype=torch.bfloat16, device=self.device) for _ in range(2))

    def set_expert_active(self, key, active):
        for start, count in self.expert_slot_ranges[key]:
            self.active_slots[start:start + count] = active

    def gather_attention_workspace(self, layer_id):
        from minisgl.kernel.triton.expert_attention import gather_dense_kv
        k, v = self.attention_workspace
        gather_dense_kv(k, v, self.addresses[:, layer_id], self.active_slots)
        return k, v

    def store_kv(self, k, v, out_loc, layer_id: int) -> None:
        from minisgl.kernel.triton.expert_attention import store_kv

        store_kv(k, v, out_loc, self.addresses[:, layer_id], self.layer_elements)

    def copy_slots(self, sources: torch.Tensor, destinations: torch.Tensor) -> None:
        from minisgl.kernel.triton.expert_attention import copy_slots

        copy_slots(self.addresses, sources, destinations, self.layer_elements)
