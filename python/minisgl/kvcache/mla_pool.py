"""One compressed latent and one positional key per token per layer."""

import torch

from .unified import UnifiedKVAllocator


class MLAKVCache:
    def __init__(self, config, num_pages, device, dtype, catalog=None):
        self.device, self.dtype = device, dtype
        self.num_layers = config.num_layers
        self.rank = config.kv_lora_rank
        self.width = self.rank + config.rotary_config.rotary_dim
        self.rows = torch.empty(
            (self.num_layers, num_pages + 1, self.width), device=device, dtype=dtype
        )
        self.block_nbytes = self.num_layers * self.width * self.rows.element_size()
        self.total_slots = num_pages + 1
        self.addresses = None
        if catalog is None:
            return
        self.allocator = UnifiedKVAllocator(num_pages, self.block_nbytes)
        catalog.create_and_register_regions(self.block_nbytes, self.allocator)
        self.region_start = {self.allocator.PERSISTENT_REGION_ID: 0}
        parts = [self._addresses(self.rows, self.total_slots)]
        for expert in catalog:
            for region in (expert.gate_up_region, expert.down_region):
                self.region_start[region.region_id] = self.total_slots
                self.total_slots += region.capacity_blocks
                parts.append(self._addresses(region.storage_view, region.capacity_blocks))
        self.addresses = torch.cat(parts, dim=1).to(device)

    def _addresses(self, tensor, slots):
        assert tensor.is_contiguous() and tensor.dtype == self.dtype
        offsets = torch.arange(self.num_layers * slots, dtype=torch.int64, device="cpu")
        return (
            tensor.data_ptr()
            + offsets.view(self.num_layers, slots) * self.width * tensor.element_size()
        )

    @property
    def allocated_nbytes(self):
        return self.rows.nbytes

    def slot_id(self, block):
        return self.region_start[block.region_id] + block.local_block_id
