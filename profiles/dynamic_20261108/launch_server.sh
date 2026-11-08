#!/bin/bash
set -e
cd /home/boxuan/ICS
export PATH=/home/boxuan/miniconda3/envs/minisgl/bin:$PATH
export LIBRARY_PATH=/home/boxuan/miniconda3/envs/minisgl/targets/x86_64-linux/lib/stubs:/home/boxuan/miniconda3/envs/minisgl/lib/python3.11/site-packages/nvidia/nccl/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/home/boxuan/miniconda3/envs/minisgl/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${ICS_CUDA_DEVICES:-6,7}
export CPATH=/home/boxuan/miniconda3/envs/minisgl/lib/python3.11/site-packages/nvidia/curand/include:${CPATH:-}
exec python -m minisgl --model /data/models/Qwen3-30B-A3B --host 127.0.0.5 --port 2000 --tp-size 2 --dtype bfloat16 --attention-backend fi --moe-backend fused --page-size 1 --num-pages 16384 --max-seq-len-override 8192 --max-prefill-length 8192 --max-running-requests 96 --cuda-graph-max-bs 96 --expert-profile ./30B_experts_frequency_sorted.json --max-offload-experts-num 100 --kt-method LLAMAFILE --kt-weight-path /data1/models/Qwen3-30B-A3B-GGUF --kt-cpuinfer 64 --kt-threadpool-count 2 --kt-max-deferred-experts-per-token 0
