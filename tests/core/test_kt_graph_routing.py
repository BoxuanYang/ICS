import sys
from types import ModuleType, SimpleNamespace

import torch
from minisgl.core import Context
from minisgl.moe.kt import KTExperts


class _Wrapper:
    def __init__(self) -> None:
        self.submissions = 0

    def submit_forward(self, hidden_states, ids, weights, stream) -> None:
        self.submissions += 1

    def sync_forward(self, hidden_states, stream):
        return torch.zeros_like(hidden_states)


def test_gguf_method_and_path_are_forwarded_to_kt(monkeypatch) -> None:
    calls = []

    class Wrapper:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)
            self.gpu_experts_mask = torch.zeros(kwargs["num_experts"], dtype=torch.bool)

        def load_weights(self, expert_map) -> None:
            self.expert_map = expert_map

    kt_kernel = ModuleType("kt_kernel")
    kt_kernel.KTMoEWrapper = Wrapper
    monkeypatch.setitem(sys.modules, "kt_kernel", kt_kernel)
    layer = SimpleNamespace(
        layer_id=0,
        num_experts=2,
        top_k=1,
        hidden_size=4,
        intermediate_size=8,
        gate_up_proj=torch.empty((2, 16, 4)),
    )
    catalog = [
        SimpleNamespace(key=(0, 0), gate_up_tensor=torch.empty(1), down_tensor=torch.empty(1))
    ]

    backend = KTExperts(
        [layer], "/models/qwen.gguf", threads=4, max_tokens=32, catalog=catalog, method="LLAMAFILE"
    )

    assert calls[0]["weight_path"] == "/models/qwen.gguf"
    assert calls[0]["method"] == "LLAMAFILE"
    assert backend.wrappers[0].gpu_experts_mask.tolist() == [True, True]


def test_graph_selects_pure_gpu_or_hybrid_topology(monkeypatch) -> None:
    import minisgl.core as core
    import minisgl.moe.kt as kt

    ctx = Context(page_size=1)
    ctx._batch = SimpleNamespace(attn_metadata=SimpleNamespace(graph_mode=True, moe_hybrid=False))
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    monkeypatch.setattr(
        kt,
        "fused_topk",
        lambda *args, **kwargs: (
            torch.ones((2, 1), dtype=torch.float32),
            torch.zeros((2, 1), dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        kt,
        "fused_experts_impl",
        lambda hidden_states, *args, **kwargs: torch.ones_like(hidden_states),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=7),
    )

    wrapper = _Wrapper()
    backend = KTExperts.__new__(KTExperts)
    backend.wrappers = {0: wrapper}
    backend.gpu_masks = {0: torch.ones(2, dtype=torch.bool)}
    backend.offloaded = {0: set()}
    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    kwargs = {
        "hidden_states": x,
        "gating_output": torch.zeros((2, 2)),
        "topk": 1,
        "renormalize": True,
        "w1": torch.empty(0),
        "w2": torch.empty(0),
    }

    torch.testing.assert_close(backend.forward(0, **kwargs), torch.ones_like(x))
    assert wrapper.submissions == 0

    # The separately captured hybrid graph retains KT callbacks even when the
    # current mask happens to be all-GPU, so later mask flips need no recapture.
    ctx._batch.attn_metadata.moe_hybrid = True
    torch.testing.assert_close(backend.forward(0, **kwargs), torch.ones_like(x))
    assert wrapper.submissions == 1

    ctx._batch.attn_metadata.moe_hybrid = False
    backend.offloaded[0].add(1)
    torch.testing.assert_close(backend.forward(0, **kwargs), torch.ones_like(x))
    assert wrapper.submissions == 2
    assert backend.has_offloaded_experts()
