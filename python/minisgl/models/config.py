from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    n_shared_experts: int = 0
    first_k_dense_replace: int = 0

    @property
    def is_moe(self) -> bool:
        return self.model_type == "deepseek_v2" or "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]
        rotary_dim = head_dim
        if model_type == "deepseek_v2":
            if (config.q_lora_rank is not None or config.topk_method != "greedy"
                    or getattr(config, "scoring_func", "softmax") != "softmax"
                    or config.routed_scaling_factor != 1 or getattr(config, "moe_layer_freq", 1) != 1
                    or config.attention_bias or getattr(config, "mlp_bias", False)):
                raise ValueError("Expected DeepSeek-V2-Lite attention and routing configuration")
            num_experts = config.n_routed_experts
            num_kv_heads = config.num_attention_heads
            rotary_dim = config.qk_rope_head_dim
            head_dim = config.qk_nope_head_dim + rotary_dim
            if (config.kv_lora_rank, rotary_dim) != (512, 64):
                raise ValueError("DeepSeek-V2-Lite MLA requires 512 latent and 64 rotary channels")
            if rope_scaling:
                rope_scaling = dict(rope_scaling)
                rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            kv_lora_rank=getattr(config, "kv_lora_rank", 0),
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 0),
            v_head_dim=getattr(config, "v_head_dim", 0),
            n_shared_experts=getattr(config, "n_shared_experts", 0),
            first_k_dense_replace=getattr(config, "first_k_dense_replace", 0),
        )
