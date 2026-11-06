#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model="${1:?Usage: bash scripts/demo_kv_moe.sh /path/to/Qwen3-30B-A3B}"
gguf="${2:?Usage: bash scripts/demo_kv_moe.sh /path/to/Qwen3-30B-A3B /path/to/Qwen3-30B-A3B-GGUF}"
# A deliberately small persistent pool makes eviction visible in a short demo.
exec python -m minisgl \
  --model-path "$model" --dtype bfloat16 --tp-size 1 \
  --expert-profile coldest_experts.txt --kt-gguf-path "$gguf" \
  --kt-cpu-threads "${KT_CPU_THREADS:-32}" \
  --num-pages 128 --page-size 1 --cache-type naive \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-8}" \
  --max-running-requests 8 --max-prefill-length 512 \
  --max-seq-len-override 4096 --host 127.0.0.1 --port 1919
