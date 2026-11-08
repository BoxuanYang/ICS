"""The private FA2 module must agree with standard FA2 for real KV aliases."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("decode,qlen", [(True, 1), (False, 1), (False, 19)])
def test_indirect_flashinfer_matches_dense_and_replays(decode, qlen, monkeypatch):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
    from minisgl.attention.indirect_fi import get_indirect_module, make_pointer_views
    import minisgl.distributed.info as info
    from minisgl.distributed import set_tp_info
    from minisgl.kvcache.expert_pool import ExpertKVCache
    from minisgl.moe.storage import ExpertStorage, ExpertStorageCatalog

    monkeypatch.setattr(info, "_TP_INFO", None)
    set_tp_info(0, 2)
    torch.manual_seed(19)
    device = torch.device("cuda")
    config = SimpleNamespace(num_layers=2, num_kv_heads=4, head_dim=128)
    weights = [torch.randn(65536, dtype=torch.bfloat16, device=device) for _ in range(2)]
    catalog = ExpertStorageCatalog([ExpertStorage(0, 0, 0.01, *weights)])
    pool = ExpertKVCache(config, 17, device, catalog)
    pool.set_expert_active((0, 0), True)
    owners, views = make_pointer_views(pool.addresses, 2, 128)
    module = get_indirect_module(128)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    bs, length = 3, 23
    indices = torch.randperm(pool.total_slots, device=device, dtype=torch.int32)[:bs * length]
    indptr = torch.arange(bs + 1, dtype=torch.int32) * length
    ones = torch.ones(bs, dtype=torch.int32)
    if decode:
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_tensor_cores=True, backend="fa2")
        wrapper.plan(indptr, indices, ones, 16, 2, 128, 1,
                     data_type=torch.bfloat16, q_data_type=torch.bfloat16)
    else:
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
        wrapper.plan(torch.arange(bs + 1, dtype=torch.int32) * qlen,
                     indptr, indices, ones, 16, 2, 128, 1, causal=True,
                     q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    standard = wrapper._cached_module
    for layer in range(config.num_layers):
        k = torch.randn(bs * length, 2, 128, dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        q = torch.randn(bs * qlen, 16, 128, dtype=torch.bfloat16, device=device)
        pool.store_kv(k, v, indices, layer)
        wrapper._cached_module = standard
        expected = wrapper.run(q, pool.gather_attention_workspace(layer))
        wrapper._cached_module = module
        actual = wrapper.run(q, views[layer])
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = wrapper.run(q, views[layer])
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
    # The adapter must not mutate either pointer table or authoritative KV.
    torch.testing.assert_close(owners[..., :pool.total_slots], pool.addresses, atol=0, rtol=0)
