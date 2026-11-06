from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from minisgl.core import get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData


@dataclass
class ExpertCaptureData(BaseCaptureData):
    pass


@dataclass
class ExpertMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor
    page_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    graph_mode: bool = False
    moe_hybrid: bool = False

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class ExpertAttention(BaseAttnBackend):
    def __init__(self, config):
        self.pool = get_global_ctx().kv_cache
        self.kv_heads = config.num_kv_heads
        self.capture: ExpertCaptureData | None = None
        self.capture_bs: List[int] = []

    def prepare_metadata(self, batch) -> None:
        ends, offset = [], 0
        for req in batch.reqs:
            offset += req.extend_len
            ends.append(offset - 1)
        compact_table = None
        seq_lens = None
        if batch.is_decode:
            reqs = batch.padded_reqs
            max_len = max(req.device_len for req in reqs)
            table = get_global_ctx().page_table
            compact_table = torch.stack([table[req.table_idx, :max_len] for req in reqs])
            seq_lens = torch.tensor(
                [req.device_len for req in reqs], dtype=torch.int32, device=self.pool.device
            )
        batch.attn_metadata = ExpertMetadata(
            last_indices=torch.tensor(ends, dtype=torch.int32, device=self.pool.device),
            page_table=compact_table,
            seq_lens=seq_lens,
        )

    def forward(self, q, k, v, layer_id, batch):
        from minisgl.kernel.triton.expert_attention import decode_attention, paged_attention

        self.pool.store_kv(k, v, batch.out_loc, layer_id)
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        metadata = batch.attn_metadata
        assert isinstance(metadata, ExpertMetadata)
        if metadata.graph_mode:
            assert metadata.page_table is not None and metadata.seq_lens is not None
            decode_attention(
                q,
                output,
                metadata.page_table,
                self.pool.addresses[:, layer_id],
                metadata.seq_lens,
                self.kv_heads,
            )
            return output

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

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized"
        max_bs = max(bs_list)
        self.capture = ExpertCaptureData.create(max_bs, max_seq_len, self.pool.device)
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch) -> None:
        assert self.capture is not None and batch.size in self.capture_bs
        bs = batch.size
        batch.attn_metadata = ExpertMetadata(
            last_indices=self.capture.cu_seqlens_q[1 : bs + 1] - 1,
            page_table=self.capture.page_table[:bs],
            seq_lens=self.capture.seq_lens[:bs],
            graph_mode=True,
        )

    def prepare_for_replay(self, batch) -> None:
        assert self.capture is not None and batch.padded_size in self.capture_bs
        metadata = batch.attn_metadata
        assert isinstance(metadata, ExpertMetadata)
        assert metadata.page_table is not None and metadata.seq_lens is not None
        bs = batch.padded_size
        table_len = metadata.page_table.size(1)
        self.capture.seq_lens[:bs].copy_(metadata.seq_lens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
