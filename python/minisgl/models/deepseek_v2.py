"""DeepSeek-V2-Lite with absorbed multi-head latent attention."""

from dataclasses import replace
from math import log

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearReplicated,
    RMSNorm,
    get_rope,
)
from minisgl.utils import div_even

from .qwen3_moe import Qwen3DecoderLayer, Qwen3MoeForCausalLM
from .utils import GatedMLP, MoEMLP


def yarn_mscale(factor, mscale):
    return 1.0 + 0.1 * mscale * log(factor) if factor > 1 else 1.0


def _neox_order(t):
    # DeepSeek rotates adjacent pairs. FlashInfer's NeoX kernel pairs the halves.
    return t.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape(len(t), -1).contiguous()


class LatentAttention(BaseOP):
    def __init__(self, config, layer_id):
        self.layer_id = layer_id
        self.heads = div_even(config.num_qo_heads, get_tp_info().size)
        self.nope = config.qk_nope_head_dim
        self.rope_dim = config.rotary_config.rotary_dim
        self.rank = config.kv_lora_rank
        self.value_dim = config.v_head_dim
        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [config.num_qo_heads * config.head_dim], False
        )
        self.kv_a_proj_with_mqa = LinearReplicated(
            config.hidden_size, self.rank + self.rope_dim, False
        )
        self.kv_a_layernorm = RMSNorm(self.rank, config.rms_norm_eps)
        self.kv_b_proj = LinearColParallelMerged(
            self.rank, [config.num_qo_heads * (self.nope + self.value_dim)], False
        )
        self.o_proj = LinearOProj(config.num_qo_heads * self.value_dim, config.hidden_size, False)
        rope = config.rotary_config
        scaling = rope.scaling or {}
        self.rotary = get_rope(
            self.rope_dim,
            self.rope_dim,
            rope.max_position,
            rope.base,
            tuple(scaling.items()) or None,
        )
        factor = scaling.get("factor", 1)
        self.rope_gain = yarn_mscale(factor, scaling.get("mscale", 1)) / yarn_mscale(
            factor, scaling.get("mscale_all_dim", 0)
        )

    def forward(self, x):
        ctx = get_global_ctx()
        q = self.q_proj.forward(x).view(len(x), self.heads, -1)
        q_nope, q_pe = q.split([self.nope, self.rope_dim], dim=-1)
        latent, k_pe = self.kv_a_proj_with_mqa.forward(x).split([self.rank, self.rope_dim], -1)
        latent = self.kv_a_layernorm.forward(latent.contiguous())

        q_pe, k_pe = self.rotary.forward(ctx.batch.positions, _neox_order(q_pe), _neox_order(k_pe))
        wk, wv = self.kv_b_proj.weight.view(self.heads, -1, self.rank).split(
            [self.nope, self.value_dim], dim=1
        )
        query = torch.bmm(q_nope.transpose(0, 1), wk).transpose(0, 1).contiguous()
        q_pe = q_pe.view(len(x), self.heads, self.rope_dim)
        if self.rope_gain != 1:
            q_pe, k_pe = q_pe * self.rope_gain, k_pe * self.rope_gain
        compressed = ctx.attn_backend.forward(
            query, q_pe, torch.cat((latent, k_pe), -1), self.layer_id, ctx.batch
        )
        values = torch.bmm(compressed.transpose(0, 1), wv.transpose(1, 2))
        return self.o_proj.forward(values.transpose(0, 1).reshape(len(x), -1))


class DeepseekMoE(MoEMLP):
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        self.shared_experts = GatedMLP(
            replace(
                config, intermediate_size=config.n_shared_experts * config.moe_intermediate_size
            )
        )

    def forward(self, x):
        # The fused routed-expert kernel reuses x for its output.
        shared = self.shared_experts.forward(x)
        logits = F.linear(x.float(), self.gate.weight.float())
        return self.experts.forward(x, logits) + shared


class DeepseekDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_id):
        self.self_attn = LatentAttention(config, layer_id)
        self.mlp = (
            GatedMLP(config)
            if layer_id < config.first_k_dense_replace
            else DeepseekMoE(config, layer_id)
        )
        self._init_norms(config, layer_id)


class DeepseekV2ForCausalLM(Qwen3MoeForCausalLM):
    decoder_layer = DeepseekDecoderLayer
