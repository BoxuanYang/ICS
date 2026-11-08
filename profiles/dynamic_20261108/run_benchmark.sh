#!/bin/bash
set -e
cd /home/boxuan/ICS
export PATH=/home/boxuan/miniconda3/envs/minisgl/bin:$PATH
result_prefix=${1:-profiles/dynamic_20261108/manual_rps8}
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --dataset /data1/zhangrui/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --rates 8 \
  --summary-output "${result_prefix}_summary.json" \
  --request-output "${result_prefix}_requests.json"
