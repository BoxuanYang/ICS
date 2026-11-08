from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

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


@dataclass
class RoutedExpertMetadata(BaseAttnMetadata):
    """Metadata for the fast contiguous path or the expert-address path."""

    pure_gpu: BaseAttnMetadata | None
    hybrid: ExpertMetadata | None
    attention_hybrid: bool
    moe_hybrid: bool
    hybrid_mask: torch.Tensor | None = None

    @property
    def graph_mode(self) -> bool:
        return bool(getattr(self.selected, "graph_mode", False))

    @property
    def selected(self) -> BaseAttnMetadata:
        metadata = self.hybrid if self.attention_hybrid else self.pure_gpu
        if metadata is None:
            raise RuntimeError("Selected attention metadata was not prepared")
        return metadata

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.selected.get_last_indices(bs)


class ExpertAttention(BaseAttnBackend):
    def __init__(self, config):
        self.pool = get_global_ctx().kv_cache
        self.kv_heads = self.pool.heads
        self.capture: ExpertCaptureData | None = None
        self.capture_bs: List[int] = []

    def prepare_metadata(self, batch, active_mask: list[bool] | None = None) -> None:
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
                [
                    req.device_len
                    if active_mask is None or active_mask[i]
                    else 0
                    for i, req in enumerate(reqs)
                ],
                dtype=torch.int32,
                device=self.pool.device,
            )
        batch.attn_metadata = ExpertMetadata(
            last_indices=torch.tensor(ends, dtype=torch.int32, device=self.pool.device),
            page_table=compact_table,
            seq_lens=seq_lens,
        )

    def forward(self, q, k, v, layer_id, batch):
        self.pool.store_kv(k, v, batch.out_loc, layer_id)
        return self.run_attention(q, layer_id, batch, batch.attn_metadata)

    def run_attention(self, q, layer_id, batch, metadata):
        """Run pointer-table attention after KV has already been stored."""
        from minisgl.kernel.triton.expert_attention import decode_attention, paged_attention

        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        assert isinstance(metadata, ExpertMetadata)
        if metadata.page_table is not None and metadata.seq_lens is not None:
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


class ExpertAttentionRouter(BaseAttnBackend):
    """Use FlashInfer until expert storage is actually serving KV blocks.

    Pointer-table attention is required after an expert is evicted, but paying
    its cost while every KV slot is still in the contiguous persistent pool
    makes the common path substantially slower.  The router captures both CUDA
    graph variants and selects the matching metadata/backend for each batch.
    """

    def __init__(self, config, use_hybrid: Callable[[], bool]):
        from .fi import FlashInferBackend

        self.pure_gpu = FlashInferBackend(config)
        self.hybrid = ExpertAttention(config)
        self.use_hybrid = use_hybrid
        self.capture_hybrid_mask: torch.Tensor | None = None

    @staticmethod
    def _run_with_metadata(batch, metadata, fn):
        routed = batch.attn_metadata
        batch.attn_metadata = metadata
        try:
            return fn()
        finally:
            batch.attn_metadata = routed

    def forward(self, q, k, v, layer_id, batch):
        metadata = batch.attn_metadata
        assert isinstance(metadata, RoutedExpertMetadata)
        if not metadata.attention_hybrid:
            return self._run_with_metadata(
                batch,
                metadata.pure_gpu,
                lambda: self.pure_gpu.forward(q, k, v, layer_id, batch),
            )

        if getattr(batch, "is_prefill", False) and metadata.pure_gpu is not None:
            from minisgl.kernel.triton.expert_attention import gather_kv
            pool = self.hybrid.pool
            pool.store_kv(k, v, batch.out_loc, layer_id)
            workspace = gather_kv(metadata.pure_gpu.gather_indices,
                                  pool.addresses[:, layer_id], pool.heads, pool.head_dim)
            return self.pure_gpu.run_attention(q, layer_id, metadata.pure_gpu, workspace)

        if (
            bool(getattr(batch, "is_prefill", False))
            or metadata.pure_gpu is None
            or metadata.hybrid_mask is None
        ):
            return self._run_with_metadata(
                batch,
                metadata.hybrid,
                lambda: self.hybrid.forward(q, k, v, layer_id, batch),
            )

        # Store every row exactly once through the unified address table.  Then
        # run FlashInfer for persistent rows and pointer attention only for rows
        # that actually own expert-backed KV.  The latter receive seq_len=0 in
        # the pointer kernel, so the slow indirect scan is not paid by the rest
        # of a large decode batch.
        self.hybrid.pool.store_kv(k, v, batch.out_loc, layer_id)
        pure_output = self.pure_gpu.run_attention(q, layer_id, metadata.pure_gpu)
        hybrid_output = self.hybrid.run_attention(q, layer_id, batch, metadata.hybrid)
        return torch.where(metadata.hybrid_mask, hybrid_output, pure_output)

    def prepare_metadata(self, batch) -> None:
        attention_hybrid = bool(getattr(batch, "uses_temporary_kv", False))
        moe_hybrid = self.use_hybrid()
        if attention_hybrid and getattr(batch, "is_prefill", False):
            self.hybrid.prepare_metadata(batch)
            hybrid_metadata = batch.attn_metadata
            self.pure_gpu.prepare_metadata(batch)
            pure_metadata = batch.attn_metadata
            pure_metadata.gather_indices = pure_metadata.indices
            pure_metadata.indices = torch.arange(pure_metadata.indices.numel(),
                                                 dtype=torch.int32,
                                                 device=self.hybrid.pool.device)
            hybrid_mask = None
        elif attention_hybrid and bool(getattr(batch, "is_decode", False)):
            active = [
                bool(getattr(req, "uses_temporary_kv", False))
                for req in batch.padded_reqs
            ]
            self.pure_gpu.prepare_metadata(batch, active_mask=[not value for value in active])
            pure_metadata = batch.attn_metadata
            self.hybrid.prepare_metadata(batch, active_mask=active)
            hybrid_metadata = batch.attn_metadata
            hybrid_mask = torch.tensor(
                active, dtype=torch.bool, device=self.hybrid.pool.device
            ).view(-1, 1, 1)
        else:
            backend = self.hybrid if attention_hybrid else self.pure_gpu
            backend.prepare_metadata(batch)
            prepared = batch.attn_metadata
            pure_metadata = None if attention_hybrid else prepared
            hybrid_metadata = prepared if attention_hybrid else None
            hybrid_mask = None
        batch.attn_metadata = RoutedExpertMetadata(
            pure_gpu=pure_metadata,
            hybrid=hybrid_metadata,
            attention_hybrid=attention_hybrid,
            moe_hybrid=moe_hybrid,
            hybrid_mask=hybrid_mask,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.pure_gpu.init_capture_graph(max_seq_len, bs_list)
        self.hybrid.init_capture_graph(max_seq_len, bs_list)
        self.capture_hybrid_mask = torch.zeros(
            (max(bs_list), 1, 1), dtype=torch.bool, device=self.hybrid.pool.device
        )

    def prepare_for_capture(self, batch) -> None:
        self.pure_gpu.prepare_for_capture(batch)
        pure_gpu = batch.attn_metadata
        self.hybrid.prepare_for_capture(batch)
        hybrid = batch.attn_metadata
        assert isinstance(hybrid, ExpertMetadata)
        assert self.capture_hybrid_mask is not None
        batch.attn_metadata = RoutedExpertMetadata(
            pure_gpu=pure_gpu,
            hybrid=hybrid,
            attention_hybrid=False,
            moe_hybrid=False,
            hybrid_mask=self.capture_hybrid_mask[: batch.size],
        )

    def prepare_for_replay(self, batch) -> None:
        metadata = batch.attn_metadata
        assert isinstance(metadata, RoutedExpertMetadata)
        if not metadata.attention_hybrid:
            self._run_with_metadata(
                batch,
                metadata.pure_gpu,
                lambda: self.pure_gpu.prepare_for_replay(batch),
            )
            return

        if (
            bool(getattr(batch, "is_decode", False))
            and metadata.pure_gpu is not None
            and metadata.hybrid_mask is not None
        ):
            assert self.capture_hybrid_mask is not None
            self.capture_hybrid_mask[: batch.padded_size].copy_(metadata.hybrid_mask)
            self._run_with_metadata(
                batch,
                metadata.pure_gpu,
                lambda: self.pure_gpu.prepare_for_replay(batch),
            )
            self._run_with_metadata(
                batch,
                metadata.hybrid,
                lambda: self.hybrid.prepare_for_replay(batch),
            )
            return


        self._run_with_metadata(
            batch,
            metadata.hybrid,
            lambda: self.hybrid.prepare_for_replay(batch),
        )


class DenseExpertAttentionRouter(ExpertAttentionRouter):
    """FlashInfer over persistent KV or a reusable per-layer gather workspace."""

    def forward(self, q, k, v, layer_id, batch):
        metadata = batch.attn_metadata
        if not metadata.attention_hybrid:
            return self._run_with_metadata(
                batch, metadata.pure_gpu,
                lambda: self.pure_gpu.forward(q, k, v, layer_id, batch))
        pool = self.hybrid.pool
        pool.store_kv(k, v, batch.out_loc, layer_id)
        workspace = pool.gather_attention_workspace(layer_id)
        return self.pure_gpu.run_attention(q, layer_id, metadata.pure_gpu, workspace)

    def prepare_metadata(self, batch):
        self.pure_gpu.prepare_metadata(batch)
        metadata = batch.attn_metadata
        batch.attn_metadata = RoutedExpertMetadata(
            pure_gpu=metadata, hybrid=metadata,
            attention_hybrid=bool(getattr(batch, "uses_temporary_kv", False)),
            moe_hybrid=self.use_hybrid())

    def init_capture_graph(self, max_seq_len, bs_list):
        self.pure_gpu.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch):
        self.pure_gpu.prepare_for_capture(batch)
        metadata = batch.attn_metadata
        batch.attn_metadata = RoutedExpertMetadata(
            pure_gpu=metadata, hybrid=metadata,
            attention_hybrid=False, moe_hybrid=False)

    def prepare_for_replay(self, batch):
        self._run_with_metadata(
            batch, batch.attn_metadata.pure_gpu,
            lambda: self.pure_gpu.prepare_for_replay(batch))


class IndirectExpertAttentionRouter(DenseExpertAttentionRouter):
    """Read live KV through its address table, without copying KV or weights."""

    def __init__(self, config, use_hybrid):
        super().__init__(config, use_hybrid)
        from .indirect_fi import (
            IndirectFlashInferUnavailable, get_indirect_module, make_pointer_views)
        pool = self.hybrid.pool
        try:
            self.indirect_module = get_indirect_module(pool.head_dim)
        except IndirectFlashInferUnavailable as exc:
            from minisgl.utils import init_logger
            init_logger(__name__).warning("Using dense KV compatibility path: %s", exc)
            self.indirect_module = None
        if self.indirect_module is not None:
            self.pointer_tables, self.pointer_views = make_pointer_views(
                pool.addresses, pool.heads, pool.head_dim)

    def forward(self, q, k, v, layer_id, batch):
        metadata = batch.attn_metadata
        if not metadata.attention_hybrid:
            return self._run_with_metadata(
                batch, metadata.pure_gpu,
                lambda: self.pure_gpu.forward(q, k, v, layer_id, batch))
        # Non-tensor-core decode uses a different FlashInfer kernel family.
        if self.indirect_module is None or (batch.is_decode and not self.pure_gpu.use_tensor_cores):
            return super().forward(q, k, v, layer_id, batch)
        self.hybrid.pool.store_kv(k, v, batch.out_loc, layer_id)
        return self.pure_gpu.run_attention(
            q, layer_id, metadata.pure_gpu, self.pointer_views[layer_id],
            module=self.indirect_module)
