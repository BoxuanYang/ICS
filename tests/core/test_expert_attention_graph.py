from types import SimpleNamespace

import torch
from minisgl.attention.expert import ExpertAttention, ExpertMetadata
from minisgl.core import Batch, Context, Req


def _req(table_idx: int, length: int) -> Req:
    return Req(
        input_ids=torch.arange(length, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=length - 1,
        output_len=2,
        uid=table_idx,
        sampling_params=None,
        cache_handle=None,
    )


def test_capture_buffers_receive_dynamic_decode_metadata(monkeypatch) -> None:
    import minisgl.core as core

    ctx = Context(page_size=1)
    ctx.kv_cache = SimpleNamespace(device=torch.device("cpu"), heads=2)
    ctx.page_table = torch.arange(32, dtype=torch.int32).view(4, 8)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)

    backend = ExpertAttention(SimpleNamespace(num_kv_heads=2))
    backend.init_capture_graph(max_seq_len=8, bs_list=[2, 4])

    batch = Batch(reqs=[_req(1, 3), _req(2, 5)], phase="decode")
    batch.padded_reqs = batch.reqs
    backend.prepare_metadata(batch)
    metadata = batch.attn_metadata
    assert isinstance(metadata, ExpertMetadata)
    assert metadata.seq_lens.tolist() == [3, 5]
    assert metadata.page_table.shape == (2, 5)

    backend.prepare_for_replay(batch)
    assert backend.capture is not None
    assert backend.capture.seq_lens[:2].tolist() == [3, 5]
    torch.testing.assert_close(
        backend.capture.page_table[:2, :5],
        ctx.page_table[torch.tensor([1, 2]), :5],
    )

    capture_batch = Batch(reqs=[_req(0, 1)] * 2, phase="decode")
    capture_batch.padded_reqs = capture_batch.reqs
    backend.prepare_for_capture(capture_batch)
    capture_metadata = capture_batch.attn_metadata
    assert isinstance(capture_metadata, ExpertMetadata)
    assert capture_metadata.graph_mode
    assert capture_metadata.page_table.data_ptr() == backend.capture.page_table.data_ptr()
