from math import log

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lite_yarn_matches_reference_beyond_original_context():
    from minisgl.layers import get_rope
    from minisgl.models.deepseek_v2 import _neox_order
    from transformers import DeepseekV2Config
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2RotaryEmbedding,
        apply_rotary_emb,
    )

    scaling = {
        "rope_type": "yarn",
        "factor": 40.0,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    config = DeepseekV2Config(rope_scaling=scaling, max_position_embeddings=16384)
    positions = torch.tensor([0, 128, 4096, 8193, 15000], device="cuda")
    q = torch.randn(len(positions), 16, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q[:, :1])
    reference = DeepseekV2RotaryEmbedding(config, device="cuda")
    freqs = reference(q, positions.unsqueeze(0))
    expected = apply_rotary_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], freqs)
    with torch.device("cuda"):
        rope = get_rope(64, 64, 16384, 10000, tuple(scaling.items()))
    rotated = rope.forward(positions, _neox_order(q), _neox_order(k))
    for actual, target in zip(rotated, expected):
        torch.testing.assert_close(
            actual, _neox_order(target[0].transpose(0, 1)), atol=0.016, rtol=0.01
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_indirect_mla_matches_flashinfer_on_fragmented_pages():
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from minisgl.kernel.triton.mla import indirect_decode

    torch.manual_seed(12)
    lengths = torch.tensor([4097, 19], dtype=torch.int32)
    heads, latent, rotary = 16, 512, 64
    rows = torch.randn(4116, latent + rotary, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(len(rows), device="cuda", dtype=torch.int32)
    table = slots.new_zeros((2, 4097))
    table[1] = slots[:4097]
    table[0, :19] = slots[4097:]
    req = slots.new_tensor([1, 0])
    q = torch.randn(2, heads, latent, device="cuda", dtype=rows.dtype)
    pe = torch.randn(2, heads, rotary, device="cuda", dtype=rows.dtype)
    addresses = rows.data_ptr() + torch.arange(len(rows), device="cuda") * rows.stride(0) * 2
    scale = 192**-0.5 * (1 + 0.1 * 0.707 * log(40)) ** 2
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    )
    wrapper.plan(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 4097, 4116], dtype=torch.int32),
        slots,
        lengths,
        heads,
        latent,
        rotary,
        1,
        False,
        scale,
        rows.dtype,
        rows.dtype,
    )
    ckv, kpe = rows.unsqueeze(1).split([latent, rotary], -1)
    expected = wrapper.run(q, pe, ckv, kpe)
    actual = indirect_decode(q, pe, addresses, table, req, lengths.cuda(), scale)
    torch.testing.assert_close(actual, expected, atol=0.009, rtol=0.025)
