"""Single-GPU BF16 experts: Mini-SGLang on GPU, KTransformers on CPU."""

from __future__ import annotations

import torch
from minisgl.utils import init_logger

from .fused import fused_experts_impl, fused_topk

logger = init_logger(__name__)


class KTExperts:
    def __init__(self, layers, model_path: str, threads: int, max_tokens: int, catalog):
        from kt_kernel import KTMoEWrapper

        self.layers = {layer.layer_id: layer for layer in layers}
        self.wrappers = {}
        self.gpu_masks = {}
        self.expert_maps = {}  # KT's native interface reads uint64_t entries.
        self.offloaded = {layer.layer_id: set() for layer in layers}
        # Exact GPU-layout copies only for candidates. KT owns all CPU experts.
        self.originals = {
            storage.key: (storage.gate_up_tensor.cpu(), storage.down_tensor.cpu())
            for storage in catalog
        }
        for layer in layers:
            wrapper = KTMoEWrapper(
                layer_idx=layer.layer_id,
                num_experts=layer.num_experts,
                num_experts_per_tok=layer.top_k,
                hidden_size=layer.hidden_size,
                moe_intermediate_size=layer.intermediate_size,
                gpu_experts_mask=None,  # Load ALL experts before marking them GPU resident.
                cpuinfer_threads=threads,
                threadpool_count=1,
                weight_path=model_path,
                chunked_prefill_size=max_tokens,
                method="BF16",
                max_deferred_experts_per_token=0,
            )
            expert_map = torch.arange(layer.num_experts, dtype=torch.int64, device="cpu")
            self.expert_maps[layer.layer_id] = expert_map
            wrapper.load_weights(expert_map)
            wrapper.gpu_experts_mask.fill_(True)  # Preserve the pointer held by KT C++.
            self.wrappers[layer.layer_id] = wrapper
            self.gpu_masks[layer.layer_id] = wrapper.gpu_experts_mask.to(layer.gate_up_proj.device)

    def route_to_cpu(self, key) -> None:
        layer, expert = key
        self.wrappers[layer].gpu_experts_mask[expert] = False
        self.gpu_masks[layer][expert] = False
        self.offloaded[layer].add(expert)

    def restore_weights(self, key) -> None:
        layer, expert = key
        gate, down = self.layers[layer].expert_tensor_views(expert)
        original_gate, original_down = self.originals[key]
        gate.copy_(original_gate)
        down.copy_(original_down)
        torch.cuda.current_stream(gate.device).synchronize()

    def route_to_gpu(self, key) -> None:
        layer, expert = key
        self.wrappers[layer].gpu_experts_mask[expert] = True
        self.gpu_masks[layer][expert] = True
        self.offloaded[layer].remove(expert)
        logger.info_rank0("KV-MoE restore layer=%d expert=%d", layer, expert)

    def forward(self, layer_id: int, **kwargs) -> torch.Tensor:
        x = kwargs["hidden_states"]
        weights, ids = fused_topk(x, kwargs["gating_output"], kwargs["topk"], kwargs["renormalize"])
        if not self.offloaded[layer_id]:
            return fused_experts_impl(x, kwargs["w1"], kwargs["w2"], weights, ids)

        wrapper = self.wrappers[layer_id]
        stream = torch.cuda.current_stream(x.device).cuda_stream
        # KT copies inputs then submits CPU work through the CUDA stream callback.
        # GPU work queued next overlaps that CPU work. Both use the original top-k weights.
        wrapper.submit_forward(x, ids, weights, stream)
        gpu_ids = ids.masked_fill(~self.gpu_masks[layer_id][ids.long()], -1)
        gpu_output = fused_experts_impl(
            x, kwargs["w1"], kwargs["w2"], weights, gpu_ids, masked_experts=True
        )
        cpu_output = wrapper.sync_forward(x, stream)
        return gpu_output.add_(cpu_output)
