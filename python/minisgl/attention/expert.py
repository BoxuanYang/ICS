from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class ExpertMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class ExpertAttention(BaseAttnBackend):
    def __init__(self, config):
        self.pool = get_global_ctx().kv_cache
        self.kv_heads = config.num_kv_heads

    def prepare_metadata(self, batch) -> None:
        ends, offset = [], 0
        for req in batch.reqs:
            offset += req.extend_len
            ends.append(offset - 1)
        batch.attn_metadata = ExpertMetadata(
            torch.tensor(ends, dtype=torch.int32, device=self.pool.device)
        )

    def forward(self, q, k, v, layer_id, batch):
        from minisgl.kernel.triton.expert_attention import paged_attention

        self.pool.store_kv(k, v, batch.out_loc, layer_id)
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        offset = 0
        table = get_global_ctx().page_table
        for req in batch.reqs:
            length = req.extend_len
            paged_attention(
                q[offset : offset + length],
                output[offset : offset + length],
                table[req.table_idx, : req.device_len],
                self.pool.addresses[:, layer_id],
                req.cached_len,
                self.kv_heads,
            )
            offset += length
        return output

    def init_capture_graph(self, max_seq_len, bs_list):
        raise RuntimeError("Expert memory sharing runs without CUDA graphs")

    def prepare_for_capture(self, batch):
        raise RuntimeError("Expert memory sharing runs without CUDA graphs")

    def prepare_for_replay(self, batch):
        raise RuntimeError("Expert memory sharing runs without CUDA graphs")
