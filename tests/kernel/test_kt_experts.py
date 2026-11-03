"""Actual pinned KT BF16 backend; no mocked CPU expert computation."""

import pytest
import torch
from safetensors.torch import save_file

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_kt_mask_and_fetch_back_match_all_gpu(tmp_path, monkeypatch):
    pytest.importorskip("kt_kernel")
    pytest.importorskip("sgl_kernel")
    pytest.importorskip("flashinfer")
    import minisgl.distributed.info as info
    from minisgl.distributed import set_tp_info
    from minisgl.layers.moe import MoELayer
    from minisgl.moe.kt import KTExperts
    from minisgl.moe.storage import ExpertStorage, ExpertStorageCatalog

    monkeypatch.setattr(info, "_TP_INFO", None)
    set_tp_info(0, 1)
    torch.manual_seed(17)
    with torch.device("cuda"):
        layer = MoELayer(0, 4, 2, 256, 256)
        layer.gate_up_proj = torch.randn(4, 512, 256, dtype=torch.bfloat16) * 0.03
        layer.down_proj = torch.randn(4, 256, 256, dtype=torch.bfloat16) * 0.03
    tensors = {}
    for expert in range(4):
        gate, up = layer.gate_up_proj[expert].chunk(2)
        for name, tensor in (("gate", gate), ("up", up), ("down", layer.down_proj[expert])):
            tensors[f"model.layers.0.mlp.experts.{expert}.{name}_proj.weight"] = tensor.cpu()
    save_file(tensors, str(tmp_path / "model.safetensors"))
    catalog = ExpertStorageCatalog(
        [ExpertStorage(0, e, 0.1, *layer.expert_tensor_views(e)) for e in range(4)]
    )
    backend = KTExperts([layer], str(tmp_path), threads=2, max_tokens=32, catalog=catalog)
    original = layer.gate_up_proj.clone(), layer.down_proj.clone()
    mask_ptr = backend.wrappers[0].gpu_experts_mask.data_ptr()
    for tokens in (1, 17):
        x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(tokens, 4, device="cuda")
        kwargs = {
            "w1": layer.gate_up_proj,
            "w2": layer.down_proj,
            "gating_output": logits,
            "topk": 2,
            "renormalize": True,
        }
        baseline = backend.forward(0, hidden_states=x.clone(), **kwargs).clone()
        torch.cuda.synchronize()
        for experts in ([0, 2], [0, 1, 2, 3]):
            for expert in experts:
                backend.route_to_cpu((0, expert))
                layer.gate_up_proj[expert].fill_(float("nan"))
                layer.down_proj[expert].fill_(float("nan"))
            actual = backend.forward(0, hidden_states=x.clone(), **kwargs)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, baseline, atol=0.025, rtol=0.05)
            assert backend.wrappers[0].gpu_experts_mask.data_ptr() == mask_ptr
            for expert in experts:
                backend.restore_weights((0, expert))
                backend.route_to_gpu((0, expert))
            assert torch.equal(layer.gate_up_proj, original[0])
            assert torch.equal(layer.down_proj, original[1])
