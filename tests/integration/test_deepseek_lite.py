"""Exercise the real checkpoint loader, prefill, graph decode and expert KV reuse."""

import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(sys.platform != "linux" or not torch.cuda.is_available(), reason="Linux CUDA")
def test_deepseek_lite_compressed_serving(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(101)
    config = DeepseekV2Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        first_k_dense_replace=1,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=4,
        n_shared_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = DeepseekV2ForCausalLM(config).to(torch.bfloat16).eval()
    model.save_pretrained(tmp_path)
    tokenizer = Tokenizer(models.WordLevel({str(i): i for i in range(256)}, unk_token="0"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="0", eos_token="2"
    ).save_pretrained(tmp_path)
    profile = tmp_path / "profile.txt"
    profile.write_text("[(1, 0, 0.1)]")
    prompt = [4, 15, 24, 37, 9, 20, 41, 83]
    with torch.inference_mode():
        tokens = torch.tensor([prompt])
        for _ in range(4):
            logits = model(tokens, use_cache=False).logits[:, -1]
            tokens = torch.cat((tokens, logits.argmax(-1, keepdim=True)), dim=1)
    expected = tokens[0, len(prompt) :].tolist()
    for hybrid in (False, True):
        code = f"""
from minisgl.llm.llm import LLM
from minisgl.core import SamplingParams
llm = LLM({str(tmp_path)!r}, max_running_req=2, max_extend_tokens=6, cache_type="naive",
          max_seq_len_override=32, cuda_graph_bs=[1, 2], kt_cpuinfer=2,
          expert_profile={str(profile) if hybrid else None!r}, num_page_override={4 if hybrid else 30})
pool = llm.engine.kv_cache
assert pool.allocated_nbytes == 2 * ({4 if hybrid else 30} + 1) * 576 * 2
for _ in range(2):
    result = llm.generate([{prompt}], SamplingParams(ignore_eos=True, max_tokens=4))
    assert result[0]['token_ids'] == {expected}, (result, {expected})
manager = llm.engine.expert_memory
if manager is not None:
    import torch
    torch.cuda.synchronize()
    manager.flush_pending_restorations()
    torch.cuda.synchronize()
    manager.poll_restorations()
    kt = llm.engine.kt_backend
    assert not kt.has_offloaded_experts()
    for key, saved in kt.originals.items():
        views = kt.layers[key[0]].expert_tensor_views(key[1])
        assert all(torch.equal(gpu.cpu(), cpu) for gpu, cpu in zip(views, saved))
llm.engine.shutdown()
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=os.environ,
            timeout=360,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        if hybrid:
            assert "KV-MoE evict" in proc.stdout + proc.stderr
            assert "variant=kv+moe-hybrid" in proc.stdout + proc.stderr
            assert "KV-MoE restore" in proc.stdout + proc.stderr
