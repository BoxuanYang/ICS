"""Run on CUDA: real weight aliasing, address-table attention and restoration."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import set_tp_info
from minisgl.kvcache.expert_pool import ExpertKVCache
from minisgl.moe.memory import ExpertMemoryManager
from minisgl.moe.offload import ExpertEvictionProfile, ExpertOffloadController, ExpertProfileEntry
from minisgl.moe.storage import ExpertStorage, ExpertStorageCatalog

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("prefix,qlen", [(0, 19), (7, 1), (5, 17)])
def test_attention_reads_expert_storage_and_restores_weights(prefix, qlen, monkeypatch):
    import minisgl.distributed.info as info
    from minisgl.kernel.triton.expert_attention import decode_attention, paged_attention

    monkeypatch.setattr(info, "_TP_INFO", None)
    set_tp_info(rank=0, size=1)
    torch.manual_seed(7)
    device = torch.device("cuda")
    cfg = SimpleNamespace(num_layers=2, num_kv_heads=2, head_dim=64)
    unit = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim
    gate = torch.randn(unit * 30 + 11, dtype=torch.bfloat16, device=device)
    down = torch.randn(unit * 15 + 13, dtype=torch.bfloat16, device=device)
    original = gate.clone(), down.clone()
    key = (0, 0)
    catalog = ExpertStorageCatalog([ExpertStorage(*key, 0.1, gate, down)])
    pool = ExpertKVCache(cfg, num_pages=3, device=device, catalog=catalog)
    profile = ExpertEvictionProfile([ExpertProfileEntry(*key, 0.1)])

    class Backend:
        cpu = False

        def route_to_cpu(self, key):
            self.cpu = True

        def restore_weights(self, key):
            gate.copy_(original[0])
            down.copy_(original[1])

        def route_to_gpu(self, key):
            self.cpu = False

    backend = Backend()
    manager = ExpertMemoryManager(
        ExpertOffloadController(profile), catalog, pool.allocator, backend
    )
    manager.begin_eviction(key)
    manager.complete_eviction(key)
    length = prefix + qlen
    blocks = pool.allocator.allocate(length)
    # Shuffle slots, mixing persistent and expert memory in one request.
    order = torch.randperm(length).tolist()
    slots = torch.tensor([pool.slot_id(blocks[i]) for i in order], device=device, dtype=torch.int32)
    for layer in range(cfg.num_layers):
        k = torch.randn(length, 2, 64, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(qlen, 4, 64, device=device, dtype=torch.bfloat16)
        pool.store_kv(k, v, slots, layer)
        actual = torch.empty_like(q)
        paged_attention(q, actual, slots, pool.addresses[:, layer], prefix, 2)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.repeat_interleave(2, 1).float()) / 8
        causal = torch.arange(length, device=device)[None, :] <= (
            prefix + torch.arange(qlen, device=device)[:, None]
        )
        probs = scores.masked_fill(~causal, -float("inf")).softmax(-1)
        expected = torch.einsum("hqk,khd->qhd", probs, v.repeat_interleave(2, 1).float())
        torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.025)
        if qlen == 1:
            graph_actual = torch.empty_like(q)
            decode_attention(
                q,
                graph_actual,
                slots.unsqueeze(0),
                pool.addresses[:, layer],
                torch.tensor([length], dtype=torch.int32, device=device),
                2,
            )
            torch.testing.assert_close(graph_actual.float(), expected, atol=0.025, rtol=0.025)
    assert pool.region_views[0].data_ptr() == gate.data_ptr()
    assert not torch.equal(gate, original[0])
    assert torch.equal(gate[-11:], original[0][-11:])  # Tail bytes never allocated.
    torch.cuda.synchronize()
    pool.allocator.release(blocks)
    assert not backend.cpu
    assert torch.equal(gate, original[0])
    assert torch.equal(down, original[1])


@pytest.mark.parametrize("cpu_experts", [[1, 3], [0, 1, 2, 3]])
def test_fused_gpu_path_does_not_read_cpu_expert_weights(cpu_experts):
    pytest.importorskip("sgl_kernel")
    pytest.importorskip("flashinfer")
    from minisgl.moe.fused import fused_experts_impl

    torch.manual_seed(3)
    x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(4, 128, 64, device="cuda", dtype=torch.bfloat16) * 0.05
    ids = torch.tensor([[0, 1], [2, 3], [1, 3], [0, 2], [3, 2]], device="cuda", dtype=torch.int32)
    weights = torch.full((5, 2), 0.5, device="cuda")
    masked = ids.clone()
    expected = torch.zeros_like(x)
    for token in range(5):
        for slot in range(2):
            expert = ids[token, slot].item()
            if expert in cpu_experts:
                masked[token, slot] = -1
            else:
                gate, up = (x[token].float() @ w1[expert].float().T).chunk(2)
                expected[token] += (
                    (torch.nn.functional.silu(gate) * up) @ w2[expert].float().T * 0.5
                ).to(x.dtype)
    # Poison overwritten expert storage. No NaN may reach the GPU output.
    w1[cpu_experts] = float("nan")
    w2[cpu_experts] = float("nan")
    actual = fused_experts_impl(x.clone(), w1, w2, weights, masked, masked_experts=True)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
