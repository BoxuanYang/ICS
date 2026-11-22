"""FlashInfer absorbed MLA over persistent and expert-backed compressed pages."""

from dataclasses import dataclass

import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kernel.triton.mla import indirect_decode, transfer_rows
from minisgl.models.deepseek_v2 import yarn_mscale
from minisgl.utils import div_even

from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class MLAMetadata(BaseAttnMetadata):
    qo: torch.Tensor
    kv: torch.Tensor
    lengths: torch.Tensor
    indices: torch.Tensor
    last: torch.Tensor
    req_ids: torch.Tensor
    device_lengths: torch.Tensor
    attention_hybrid: bool
    wrapper: BatchMLAPagedAttentionWrapper | None = None

    def get_last_indices(self, bs):
        return self.last[:bs]


class MLABackend(BaseAttnBackend):
    def __init__(self, config):
        self.pool = get_global_ctx().kv_cache
        self.heads = div_even(config.num_qo_heads, get_tp_info().size)
        scaling = config.rotary_config.scaling or {}
        gain = yarn_mscale(scaling.get("factor", 1), scaling.get("mscale_all_dim", 0))
        self.scale = gain**2 * config.head_dim**-0.5
        self.workspace = torch.empty(128 << 20, dtype=torch.uint8, device=self.pool.device)
        self.eager = BatchMLAPagedAttentionWrapper(self.workspace)
        self.graphs = {}
        self.plan_ready = torch.cuda.Event()

    def prepare_metadata(self, batch):
        reqs = batch.padded_reqs
        lengths = torch.tensor([r.device_len for r in reqs], device="cpu", dtype=torch.int32)
        qo = torch.tensor([0] + [r.extend_len for r in reqs], device="cpu", dtype=torch.int32)
        qo.cumsum_(0)
        kv = torch.nn.functional.pad(lengths, (1, 0)).cumsum(0, dtype=torch.int32)
        dev = self.pool.device
        table = get_global_ctx().page_table
        hybrid = getattr(batch, "uses_temporary_kv", False)
        indices = (
            table.new_empty(0)
            if hybrid and batch.is_decode
            else torch.cat([table[r.table_idx, : r.device_len] for r in reqs])
        )
        batch.attn_metadata = MLAMetadata(
            qo,
            kv,
            lengths,
            indices,
            (qo[1:] - 1).to(dev),
            torch.tensor([r.table_idx for r in reqs], device=dev, dtype=torch.int32),
            lengths.to(dev),
            hybrid,
        )

    def _plan(self, batch, wrapper):
        m = batch.attn_metadata
        indices = m.indices
        if m.attention_hybrid and batch.is_prefill:
            indices = torch.arange(len(indices), device=self.pool.device, dtype=torch.int32)
        # The host planning buffer is reused by the next asynchronous H2D copy.
        self.plan_ready.synchronize()
        wrapper.plan(
            m.qo,
            m.kv,
            indices,
            m.lengths,
            self.heads,
            self.pool.rank,
            self.pool.width - self.pool.rank,
            1,
            batch.is_prefill,
            self.scale,
            self.pool.dtype,
            self.pool.dtype,
        )
        self.plan_ready.record()
        m.wrapper = wrapper

    def forward(self, q, pe, rows, layer_id, batch):
        m = batch.attn_metadata
        if m.attention_hybrid:
            addresses = self.pool.addresses[layer_id]
            transfer_rows(rows, batch.out_loc, addresses, store=True)
            if batch.is_decode:
                return indirect_decode(
                    q,
                    pe,
                    addresses,
                    get_global_ctx().page_table,
                    m.req_ids,
                    m.device_lengths,
                    self.scale,
                )
            # Compact only referenced compressed rows for this layer's prefill.
            cache = rows.new_empty((len(m.indices), self.pool.width))
            transfer_rows(cache, m.indices, addresses, store=False)
        else:
            cache = self.pool.rows[layer_id]
            cache.index_copy_(0, batch.out_loc.long(), rows)
        if m.wrapper is None:
            self._plan(batch, self.eager)
        ckv, kpe = cache.unsqueeze(1).split([self.pool.rank, self.pool.width - self.pool.rank], -1)
        return m.wrapper.run(q, pe, ckv, kpe)

    def init_capture_graph(self, max_seq_len, bs_list):
        bs = max(bs_list)

        def ints(n):
            return torch.empty(n, dtype=torch.int32, device=self.pool.device)

        self.graph_qo, self.graph_kv = ints(bs + 1), ints(bs + 1)
        self.graph_indices = ints(bs * max_seq_len)
        self.graph_lengths, self.graph_reqs = ints(bs), ints(bs)

    def prepare_for_capture(self, batch):
        bs = batch.size
        self.prepare_metadata(batch)
        wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace,
            True,
            self.graph_qo[: bs + 1],
            self.graph_kv[: bs + 1],
            self.graph_indices,
            self.graph_lengths[:bs],
        )
        self.graphs[bs] = wrapper
        self.prepare_for_replay(batch)

    def prepare_for_replay(self, batch):
        bs, m = batch.padded_size, batch.attn_metadata
        self.graph_reqs[:bs].copy_(m.req_ids)
        self.graph_lengths[:bs].copy_(m.device_lengths)
        m.req_ids, m.device_lengths = self.graph_reqs[:bs], self.graph_lengths[:bs]
        if not m.attention_hybrid:
            self._plan(batch, self.graphs[bs])
