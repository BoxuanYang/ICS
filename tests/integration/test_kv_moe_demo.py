"""Small real Qwen3 MoE checkpoint -> HTTP serving -> eviction -> eager restore."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import torch


@pytest.mark.skipif(sys.platform != "linux" or not torch.cuda.is_available(), reason="Linux CUDA")
def test_two_request_waves(tmp_path):
    pytest.importorskip("kt_kernel")
    pytest.importorskip("fastapi")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path)
    vocab = {f"t{i}": i for i in range(128)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", bos_token="t1", eos_token="t2"
    ).save_pretrained(tmp_path)
    profile = tmp_path / "profile.txt"
    profile.write_text(repr([(layer, expert, 0.1) for layer in range(2) for expert in range(4)]))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "server.log"
    root = Path(__file__).resolve().parents[2]
    args = [
        sys.executable,
        "-m",
        "minisgl",
        "--model-path",
        str(tmp_path),
        "--expert-profile",
        str(profile),
        "--kt-cpu-threads",
        "2",
        "--num-pages",
        "4",
        "--cache-type",
        "naive",
        "--dtype",
        "bfloat16",
        "--max-running-requests",
        "4",
        "--max-prefill-length",
        "32",
        "--cuda-graph-max-bs",
        "4",
        "--max-seq-len-override",
        "128",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    def request(_):
        data = json.dumps(
            {
                "model": str(tmp_path),
                "prompt": "t5 " * 35,
                "max_tokens": 4,
                "temperature": 0,
                "ignore_eos": True,
            }
        ).encode()
        req = Request(url + "/v1/chat/completions", data, {"Content-Type": "application/json"})
        with urlopen(req, timeout=120) as response:
            return json.load(response)["choices"][0]["message"]["content"]

    with log_path.open("w") as log:
        process = subprocess.Popen(args, cwd=root, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while True:
                assert process.poll() is None, log_path.read_text()
                try:
                    with urlopen(url + "/v1/models", timeout=1):
                        break
                except (URLError, TimeoutError):
                    assert time.monotonic() < deadline, log_path.read_text()
                    time.sleep(0.5)
            replies = []
            for _ in range(2):
                with ThreadPoolExecutor(2) as executor:
                    replies.extend(executor.map(request, range(2)))
            assert all(replies)
            assert len(set(replies)) == 1  # Same greedy result after restore and re-eviction.
            logs = log_path.read_text()
            assert "Start capturing CUDA graphs" in logs, logs
            assert logs.count("KV-MoE evict") >= 2, logs
            assert logs.count("KV-MoE restore") >= 2, logs
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
