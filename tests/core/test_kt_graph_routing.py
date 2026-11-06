from types import SimpleNamespace

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


def test_graph_capture_records_hybrid_path_with_all_gpu_mask(monkeypatch) -> None:
    import minisgl.core as core
    import minisgl.moe.kt as kt

    ctx = Context(page_size=1)
    ctx._batch = SimpleNamespace(attn_metadata=SimpleNamespace(graph_mode=True))
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
    assert wrapper.submissions == 1

    # Outside graph capture, an all-GPU batch retains the original fast path.
    ctx._batch.attn_metadata.graph_mode = False
    torch.testing.assert_close(backend.forward(0, **kwargs), torch.ones_like(x))
    assert wrapper.submissions == 1
