import sys
from types import ModuleType, SimpleNamespace

import torch
from minisgl.attention.expert import DenseExpertAttentionRouter, ExpertAttentionRouter, RoutedExpertMetadata
from minisgl.core import Context
from minisgl.moe.kt import KTExperts


class _Wrapper:
    def __init__(self) -> None:
        self.submissions = 0

    def submit_forward(self, hidden_states, ids, weights, stream) -> None:
        self.submissions += 1

    def sync_forward(self, hidden_states, stream):
        return torch.zeros_like(hidden_states)


class _AttentionBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []

    def prepare_metadata(self, batch) -> None:
        batch.attn_metadata = SimpleNamespace(
            name=self.name,
            get_last_indices=lambda bs: torch.arange(bs),
        )

    def forward(self, q, k, v, layer_id, batch):
        self.calls.append(batch.attn_metadata.name)
        return self.name


def test_dense_router_preserves_pure_path_and_gathers_only_temporary_batches():
    router = DenseExpertAttentionRouter.__new__(DenseExpertAttentionRouter)
    router.pure_gpu = _AttentionBackend("pure")
    calls = []
    pool = SimpleNamespace(
        store_kv=lambda *args: calls.append("store"),
        gather_attention_workspace=lambda layer: calls.append("gather") or "workspace")
    router.hybrid = SimpleNamespace(pool=pool)
    router.use_hybrid = lambda: True
    router.pure_gpu.run_attention = lambda q, layer, metadata, workspace: workspace
    batch = SimpleNamespace(attn_metadata=None, out_loc=None, uses_temporary_kv=False)
    router.prepare_metadata(batch)
    assert router.forward(None, None, None, 0, batch) == "pure"
    assert calls == []
    batch.uses_temporary_kv = True
    router.prepare_metadata(batch)
    assert router.forward(None, None, None, 0, batch) == "workspace"
    assert calls == ["store", "gather"]


def test_attention_router_uses_flashinfer_until_an_expert_is_offloaded() -> None:
    offloaded = False
    router = ExpertAttentionRouter.__new__(ExpertAttentionRouter)
    router.pure_gpu = _AttentionBackend("pure")
    router.hybrid = _AttentionBackend("hybrid")
    router.use_hybrid = lambda: offloaded
    batch = SimpleNamespace(attn_metadata=None)

    router.prepare_metadata(batch)
    assert isinstance(batch.attn_metadata, RoutedExpertMetadata)
    assert not batch.attn_metadata.moe_hybrid
    assert router.forward(None, None, None, 0, batch) == "pure"
    assert router.pure_gpu.calls == ["pure"]
    assert isinstance(batch.attn_metadata, RoutedExpertMetadata)

    offloaded = True
    router.prepare_metadata(batch)
    assert batch.attn_metadata.moe_hybrid
    assert not batch.attn_metadata.attention_hybrid
    assert router.forward(None, None, None, 0, batch) == "pure"

    batch.uses_temporary_kv = True
    router.prepare_metadata(batch)
    assert batch.attn_metadata.moe_hybrid
    assert batch.attn_metadata.attention_hybrid
    assert router.forward(None, None, None, 0, batch) == "hybrid"
    assert router.hybrid.calls == ["hybrid"]


def test_gguf_method_and_path_are_forwarded_to_kt(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: SimpleNamespace())

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
        [layer],
        "/models/qwen.gguf",
        threads=4,
        max_tokens=32,
        catalog=catalog,
        method="LLAMAFILE",
        threadpool_count=3,
        max_deferred_experts_per_token=2,
    )

    assert calls[0]["weight_path"] == "/models/qwen.gguf"
    assert calls[0]["method"] == "LLAMAFILE"
    assert calls[0]["cpuinfer_threads"] == 4
    assert calls[0]["threadpool_count"] == 3
    assert calls[0]["max_deferred_experts_per_token"] == 2
    assert backend.wrappers[0].gpu_experts_mask.tolist() == [True, True]


def test_cli_accepts_standard_kt_parameters() -> None:
    from minisgl.server.args import parse_args

    args, _ = parse_args(
        [
            "--model",
            "/models/qwen",
            "--dtype",
            "bfloat16",
            "--kt-method",
            "AMXINT8",
            "--kt-weight-path",
            "/cpu/weights",
            "--kt-cpuinfer",
            "64",
            "--kt-threadpool-count",
            "2",
            "--kt-max-deferred-experts-per-token",
            "3",
            "--max-offload-experts-num",
            "192",
        ]
    )

    assert args.kt_method == "AMXINT8"
    assert args.kt_weight_path == "/cpu/weights"
    assert args.kt_cpuinfer == 64
    assert args.kt_threadpool_count == 2
    assert args.kt_max_deferred_experts_per_token == 3
    assert args.max_offload_experts_num == 192


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
    backend.gpu_expert_maps = {0: torch.arange(2, dtype=torch.int64)}
    backend.offloaded = {0: set()}
    backend.tp_rank = 0
    backend.tp_size = 1
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


def test_hybrid_graph_keeps_non_candidate_layers_on_fused_gpu(monkeypatch) -> None:
    import minisgl.core as core
    import minisgl.moe.kt as kt

    ctx = Context(page_size=1)
    ctx._batch = SimpleNamespace(attn_metadata=SimpleNamespace(moe_hybrid=True))
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

    wrapper = _Wrapper()
    backend = KTExperts.__new__(KTExperts)
    backend.wrappers = {0: wrapper}
    backend.candidate_layer_ids = frozenset({1})
    backend.gpu_masks = {0: torch.ones(2, dtype=torch.bool)}
    backend.gpu_expert_maps = {0: torch.arange(2, dtype=torch.int64)}
    backend.offloaded = {1: set()}
    backend.tp_rank = 0
    backend.tp_size = 1
    x = torch.zeros((2, 4), dtype=torch.bfloat16)

    actual = backend.forward(
        0,
        hidden_states=x,
        gating_output=torch.zeros((2, 2)),
        topk=1,
        renormalize=True,
        w1=torch.empty(0),
        w2=torch.empty(0),
    )

    torch.testing.assert_close(actual, torch.ones_like(x))
    assert wrapper.submissions == 0


def test_only_primary_tp_rank_contributes_replicated_cpu_output(monkeypatch) -> None:
    import minisgl.core as core
    import minisgl.moe.kt as kt

    ctx = Context(page_size=1)
    ctx._batch = SimpleNamespace(attn_metadata=SimpleNamespace(moe_hybrid=True))
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
    wrapper.sync_forward = lambda hidden_states, stream: torch.full_like(hidden_states, 4)
    backend = KTExperts.__new__(KTExperts)
    backend.wrappers = {0: wrapper}
    backend.gpu_masks = {0: torch.tensor([False, True])}
    backend.gpu_expert_maps = {0: torch.tensor([-1, 1], dtype=torch.int64)}
    backend.offloaded = {0: {0}}
    backend.tp_rank = 0
    backend.tp_size = 2
    x = torch.zeros((2, 4), dtype=torch.bfloat16)

    actual = backend.forward(
        0,
        hidden_states=x,
        gating_output=torch.zeros((2, 2)),
        topk=1,
        renormalize=True,
        w1=torch.empty(0),
        w2=torch.empty(0),
    )

    torch.testing.assert_close(actual, torch.full_like(x, 5))
    assert wrapper.submissions == 1

    backend.tp_rank = 1
    torch.testing.assert_close(
        backend.forward(
            0,
            hidden_states=x,
            gating_output=torch.zeros((2, 2)),
            topk=1,
            renormalize=True,
            w1=torch.empty(0),
            w2=torch.empty(0),
        ),
        torch.ones_like(x),
    )
    assert wrapper.submissions == 1


def test_static_offload_remaps_original_ids_to_compact_gpu_rows(monkeypatch) -> None:
    import minisgl.core as core
    import minisgl.moe.kt as kt

    ctx = Context(page_size=1)
    ctx._batch = SimpleNamespace(attn_metadata=SimpleNamespace(moe_hybrid=True))
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    monkeypatch.setattr(
        kt,
        "fused_topk",
        lambda *args, **kwargs: (
            torch.ones((3, 1), dtype=torch.float32),
            torch.tensor([[0], [1], [2]], dtype=torch.int32),
        ),
    )
    seen_ids = []

    def fake_fused(hidden_states, w1, w2, weights, ids, **kwargs):
        seen_ids.append(ids.clone())
        return torch.ones_like(hidden_states)

    monkeypatch.setattr(kt, "fused_experts_impl", fake_fused)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=7),
    )

    wrapper = _Wrapper()
    backend = KTExperts.__new__(KTExperts)
    backend.wrappers = {0: wrapper}
    backend.candidate_layer_ids = frozenset({0})
    backend.offloaded = {0: {1}}
    backend.static_offload = True
    backend.gpu_expert_maps = {0: torch.tensor([0, -1, 1], dtype=torch.int64)}
    backend.tp_rank = 0
    backend.tp_size = 2
    x = torch.zeros((3, 4), dtype=torch.bfloat16)

    actual = backend.forward(
        0,
        hidden_states=x,
        gating_output=torch.zeros((3, 3)),
        topk=1,
        renormalize=True,
        w1=torch.empty((2, 1, 1)),
        w2=torch.empty((2, 1, 1)),
    )

    torch.testing.assert_close(actual, torch.ones_like(x))
    torch.testing.assert_close(
        seen_ids[0], torch.tensor([[0], [-1], [1]], dtype=torch.int64)
    )
    assert wrapper.submissions == 1
