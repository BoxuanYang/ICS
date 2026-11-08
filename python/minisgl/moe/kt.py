"""BF16 experts: Mini-SGLang on GPU, KTransformers on CPU."""

from __future__ import annotations

import torch
from minisgl.distributed import try_get_tp_info
from minisgl.utils import init_logger

from .fused import fused_experts_impl, fused_topk

logger = init_logger(__name__)


class KTExperts:
    def __init__(
        self,
        layers,
        model_path: str,
        threads: int,
        max_tokens: int,
        catalog,
        method: str = "BF16",
        threadpool_count: int = 1,
        max_deferred_experts_per_token: int = 0,
        commit_stream=None,
        static_offload: bool = False,
    ):
        from kt_kernel import KTMoEWrapper

        self.layers = {layer.layer_id: layer for layer in layers}
        catalog = tuple(catalog)
        self.candidate_keys = frozenset(storage.key for storage in catalog)
        self.candidate_layer_ids = frozenset(key[0] for key in self.candidate_keys)
        self.static_offload = static_offload
        self.wrappers = {}
        self.gpu_masks = {}
        self.gpu_expert_maps = {}
        self.expert_maps = {}  # KT's native interface reads uint64_t entries.
        self.offloaded = {layer_id: set() for layer_id in self.candidate_layer_ids}
        self.commit_stream = commit_stream
        self.fetch_stream = torch.cuda.Stream(device=layers[0].gate_up_proj.device)
        tp_info = try_get_tp_info()
        self.tp_rank = tp_info.rank if tp_info is not None else 0
        self.tp_size = (
            tp_info.size
            if tp_info is not None
            else (getattr(layers[0], "tp_size", 1) if layers else 1)
        )
        if self.tp_rank == 0:
            logger.info(
                "KT CPU executor: cpuinfer=%d, threadpool_count=%d, "
                "max_deferred_experts_per_token=%d",
                threads,
                threadpool_count,
                max_deferred_experts_per_token,
            )
        # Exact GPU-layout copies only for candidates. KT owns all CPU experts.
        self.originals = (
            {}
            if static_offload
            else {
                storage.key: (
                    storage.gate_up_tensor.cpu().pin_memory(),
                    storage.down_tensor.cpu().pin_memory(),
                )
                for storage in catalog
            }
        )
        max_layer_id = max(self.layers) if self.layers else -1
        max_experts = max((layer.num_experts for layer in layers), default=0)
        mask_device = layers[0].gate_up_proj.device if layers else "cpu"
        # Single device-side source of truth used by every MoE layer.  KT keeps
        # its native host mask per wrapper; route_to_cpu/route_to_gpu update both
        # views as one residency transaction.
        self.global_gpu_mask = torch.ones(
            (max_layer_id + 1, max_experts), dtype=torch.bool, device=mask_device
        )
        for layer in layers:
            if layer.layer_id not in self.candidate_layer_ids:
                continue
            wrapper = KTMoEWrapper(
                layer_idx=layer.layer_id,
                num_experts=layer.num_experts,
                num_experts_per_tok=layer.top_k,
                hidden_size=layer.hidden_size,
                moe_intermediate_size=layer.intermediate_size,
                gpu_experts_mask=None,  # Load ALL experts before marking them GPU resident.
                cpuinfer_threads=threads,
                threadpool_count=threadpool_count,
                weight_path=model_path,
                chunked_prefill_size=max_tokens,
                method=method,
                max_deferred_experts_per_token=max_deferred_experts_per_token,
            )
            expert_map = torch.arange(layer.num_experts, dtype=torch.int64, device="cpu")
            self.expert_maps[layer.layer_id] = expert_map
            wrapper.load_weights(expert_map)
            wrapper.gpu_experts_mask.fill_(True)  # Preserve the pointer held by KT C++.
            cold_ids = sorted(
                expert_id
                for layer_id, expert_id in self.candidate_keys
                if layer_id == layer.layer_id
            )
            if static_offload:
                wrapper.gpu_experts_mask[cold_ids] = False
                self.offloaded[layer.layer_id].update(cold_ids)
            self.wrappers[layer.layer_id] = wrapper
            self.gpu_masks[layer.layer_id] = self.global_gpu_mask[
                layer.layer_id, : layer.num_experts
            ]
            if not static_offload:
                self.gpu_expert_maps[layer.layer_id] = torch.arange(
                    layer.num_experts, dtype=torch.int64, device=mask_device)
            if static_offload:
                self.gpu_masks[layer.layer_id][cold_ids] = False
                gpu_expert_map = torch.full(
                    (layer.num_experts,),
                    -1,
                    dtype=torch.int64,
                    device=mask_device,
                )
                gpu_expert_ids = getattr(
                    layer,
                    "gpu_expert_ids",
                    tuple(i for i in range(layer.num_experts) if i not in cold_ids),
                )
                gpu_expert_map[
                    torch.tensor(gpu_expert_ids, dtype=torch.int64, device=mask_device)
                ] = torch.arange(len(gpu_expert_ids), dtype=torch.int64, device=mask_device)
                self.gpu_expert_maps[layer.layer_id] = gpu_expert_map

    @staticmethod
    def set_capture_batch_sizes(batch_sizes) -> None:
        """Keep KT's pinned CPU/GPU buffers stable across CUDA graph replay."""
        from kt_kernel import KTMoEWrapper

        KTMoEWrapper.set_capture_batch_sizes(list(batch_sizes))

    def route_to_cpu(self, key) -> None:
        layer, expert = key
        self.wrappers[layer].gpu_experts_mask[expert] = False
        stream = self.commit_stream or torch.cuda.current_stream(
            self.gpu_masks[layer].device
        )
        with torch.cuda.stream(stream):
            self.gpu_masks[layer][expert] = False
            self.gpu_expert_maps[layer][expert] = -1
            if hasattr(self, "kv_pool"):
                self.kv_pool.set_expert_active(key, True)
        self.offloaded[layer].add(expert)

    def restore_weights(self, key):
        if self.static_offload:
            raise RuntimeError("Static CPU experts cannot be restored to GPU")
        layer, expert = key
        gate, down = self.layers[layer].expert_tensor_views(expert)
        original_gate, original_down = self.originals[key]
        # A speculative decode may still read logically released KV pages.
        if self.commit_stream is not None:
            with torch.cuda.stream(self.commit_stream):
                if hasattr(self, "kv_pool"):
                    self.kv_pool.set_expert_active(key, False)
            self.fetch_stream.wait_stream(self.commit_stream)
        with torch.cuda.stream(self.fetch_stream):
            gate.copy_(original_gate, non_blocking=True)
            down.copy_(original_down, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.fetch_stream)
        return done

    def route_to_gpu(self, key) -> None:
        if self.static_offload:
            raise RuntimeError("Static CPU experts cannot be routed to GPU")
        layer, expert = key
        self.wrappers[layer].gpu_experts_mask[expert] = True
        # Enqueue the device-mask commit on the engine stream. It is therefore
        # ordered after the preceding batch and before the next batch, without
        # synchronizing either stream on the CPU.
        stream = self.commit_stream or torch.cuda.current_stream(
            self.gpu_masks[layer].device
        )
        with torch.cuda.stream(stream):
            self.gpu_masks[layer][expert] = True
            self.gpu_expert_maps[layer][expert] = expert
        self.offloaded[layer].remove(expert)
        logger.info_rank0("KV-MoE restore layer=%d expert=%d", layer, expert)

    def has_offloaded_experts(self) -> bool:
        return any(self.offloaded.values())

    def agree_fetch_completion(self, ready):
        if self.tp_size == 1:
            return ready
        values = torch.tensor(ready, dtype=torch.int32, device="cpu")
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MIN,
                                     group=self.tp_cpu_group)
        return values.tolist()

    def forward(self, layer_id: int, **kwargs) -> torch.Tensor:
        x = kwargs["hidden_states"]
        weights, ids = fused_topk(x, kwargs["gating_output"], kwargs["topk"], kwargs["renormalize"])
        # Prefill and uncaptured decode can keep the cheap all-GPU fast path.
        # Graph capture must record the hybrid topology even when every mask bit
        # is initially true, because later replays may flip experts to CPU.
        from minisgl.core import get_global_ctx

        try:
            metadata = getattr(get_global_ctx().batch, "attn_metadata", None)
        except AssertionError:  # Direct backend/kernel tests do not install an engine context.
            metadata = None
        hybrid_graph = bool(getattr(metadata, "moe_hybrid", False))
        # This branch is invariant for the lifetime of the engine.  Layers with
        # no profile candidates can never be evicted, so even the hybrid graph
        # keeps their original fused GPU MoE path and avoids a KT callback.
        candidate_layers = getattr(self, "candidate_layer_ids", self.wrappers.keys())
        if layer_id not in candidate_layers:
            return fused_experts_impl(x, kwargs["w1"], kwargs["w2"], weights, ids)
        if not hybrid_graph and not self.offloaded[layer_id]:
            return fused_experts_impl(x, kwargs["w1"], kwargs["w2"], weights, ids)

        wrapper = self.wrappers[layer_id]
        stream = torch.cuda.current_stream(x.device).cuda_stream
        # The hybrid graph has one stable topology and reads the masks at replay.
        # A separate pure-GPU graph avoids these host callbacks when every expert
        # has returned to the GPU.
        # KT holds a complete CPU expert, while the GPU expert weights are TP
        # shards.  Compute the replicated CPU contribution on one rank only;
        # the MoE layer's existing all-reduce broadcasts it to every rank.
        if self.tp_rank == 0:
            wrapper.submit_forward(x, ids, weights, stream)
        # A stable identity/-1 lookup needs one gather, avoiding separate
        # bool-index, inversion and masked-fill kernels in every MoE layer.
        gpu_ids = self.gpu_expert_maps[layer_id][ids.long()]
        gpu_output = fused_experts_impl(
            x, kwargs["w1"], kwargs["w2"], weights, gpu_ids, masked_experts=True
        )
        if self.tp_rank != 0:
            return gpu_output
        return gpu_output.add_(wrapper.sync_forward(x, stream))
