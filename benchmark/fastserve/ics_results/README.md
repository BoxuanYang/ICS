# ICS benchmark results (2026-11-08)

## Final configuration

The selected configuration uses an elastic admission/offload window of 24 experts and the 40 coldest experts from layer 44 (`profiles/cold_layer44_40.txt`). It preserves low-load performance while increasing high-load throughput. All measurements use the same dataset, seed, request sample, token limits, arrival process, and persistent KV pool size as `benchmark/fastserve/minisgl_results/summary.json`.

| RPS | Requests | Baseline output tok/s | ICS output tok/s | Throughput change | Baseline normalized latency | ICS normalized latency | Successful |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 184.887 | 184.698 | -0.10% | 23.228 ms/token | 24.090 ms/token | 100/100 |
| 8 | 800 | 581.033 | 585.215 | +0.72% | 5.714 s/token | 5.687 s/token | 800/800 |
| 14 | 1400 | 629.377 | 674.643 | +7.19% | 12.345 s/token | 11.239 s/token | 1400/1400 |

Result files: `rps1_summary.json`, `rps8_summary.json`, and `rps14_summary.json`.

## Launch command

```bash
ulimit -n 65535
cd /home/boxuan/ICS
CUDA_VISIBLE_DEVICES=5,6 /home/boxuan/miniconda3/envs/minisgl/bin/python -m minisgl \
  --model /data/models/Qwen3-30B-A3B \
  --host 127.0.0.4 --port 2000 \
  --tp-size 2 --dtype bfloat16 \
  --page-size 1 --num-pages 16384 \
  --max-seq-len-override 8192 --max-prefill-length 8192 \
  --max-running-requests 96 --cuda-graph-max-bs 96 \
  --expert-profile /home/boxuan/ICS/profiles/cold_layer44_40.txt \
  --kt-method LLAMAFILE \
  --kt-weight-path /data1/models/Qwen3-30B-A3B-GGUF \
  --kt-cpuinfer 64 --kt-threadpool-count 2 \
  --kt-max-deferred-experts-per-token 0
```

The convenience launcher `/data1/boxuan/run_ics_server.sh` also sets the CUDA and library paths required by this host.

## Benchmark command

Replace `RATE` with 1, 8, or 14 and `OUTPUT` with the desired result prefix.

```bash
ulimit -n 65535
cd /home/boxuan/ICS
/home/boxuan/miniconda3/envs/minisgl/bin/python \
  benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --base-url http://127.0.0.4:2000 \
  --model /data/models/Qwen3-30B-A3B \
  --tokenizer /data/models/Qwen3-30B-A3B \
  --dataset /data1/zhangrui/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --rates RATE --duration-per-rate 100 --seed 0 \
  --max-prompt-tokens 4096 --max-total-tokens 8192 \
  --inter-group-delay 0 \
  --summary-output "${OUTPUT}_summary.json" \
  --request-output "${OUTPUT}_requests.json"
```

## Interpretation

The final RPS=8 and RPS=14 runs did not enter the CPU-expert execution path. Their gain comes from elastic admission: the scheduler can use the expert-backed KV address space as bounded headroom instead of rejecting work based on MiniSGLang's conservative full-output reservation. Configurations with windows of 28 or 32 did trigger eviction at RPS=8, but KT CPU MoE cost outweighed the extra concurrency and reduced throughput; they were therefore rejected. A 2x gain was not achieved, and the remaining bottleneck is actual offloaded-expert execution rather than KV capacity or fetch/evict bookkeeping.
