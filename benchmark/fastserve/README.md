# FastServe-style ShareGPT benchmark for Mini-SGLang

This benchmark replays ShareGPT requests with Poisson arrivals against
Mini-SGLang's OpenAI-compatible Chat Completions endpoint. By default it sweeps
offered load from 1 through 20 requests per second (RPS).

For request `i`, the script measures end-to-end latency `T_i` from HTTP
submission until the stream's `[DONE]` event. This includes server queueing,
prefill, and decode. The per-request normalized latency is:

```text
t_i = T_i / N_i
```

`N_i` is the completion-token count reported in a final usage chunk when the
server supplies one. Mini-SGLang currently does not emit streaming usage, so the
client falls back to counting its one-token-per-event streaming chunks. The
reported normalized latency is the arithmetic mean of `t_i` over successful
requests in one RPS group. Failed requests are listed separately and excluded.

## Prepare the ShareGPT dataset

The ShareGPT benchmark downloads its dataset from Hugging Face on first use. If
the compute node cannot access Hugging Face reliably, download the dataset
before running the benchmark:

```bash
mkdir -p benchmark/fastserve/datasets

DATASET_URL="${HF_ENDPOINT:-https://huggingface.co}/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_PATH="benchmark/fastserve/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

curl -L --fail --retry 5 -C - \
  -o benchmark/fastserve/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  "https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
```

If Hugging Face is not reachable from the current network, set `HF_ENDPOINT`
to an accessible Hugging Face mirror before running the commands above:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then pass the local file to the benchmark so it does not try to download the
dataset again:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --dataset benchmark/fastserve/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```


## Run

Start Mini-SGLang (its default port is 1919):

```bash
python -m minisgl --model Qwen/Qwen3-0.6B
```

Then run the benchmark from the repository root:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py
```

The script discovers the served model from `/v1/models` and uses it as the
tokenizer by default. You can override either value when the API model ID is not
a usable local path or Hugging Face model ID:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --base-url http://127.0.0.1:1919 \
  --model Qwen/Qwen3-0.6B \
  --tokenizer Qwen/Qwen3-0.6B
```

When `--dataset` is omitted, the standard
`anon8231489123/ShareGPT_Vicuna_unfiltered` dataset is downloaded through
Hugging Face. To use a local copy, add:

```bash
--dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

The default request count is `ceil(RPS * 200)` for each group, giving 200
requests at RPS 1 and 4000 requests at RPS 20. This keeps the offered-load
window at 200 seconds per group. The full default sweep submits 42,000 requests
and takes at least 66 minutes and 40 seconds, plus queue drain and inter-group
delays.

For a shorter one-minute sweep:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --duration-per-rate 60
```

To use one fixed request count at every RPS:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --num-prompts 1000
```

You can also provide one count per rate:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --rates 1 5 10 15 20 \
  --num-prompts 200 1000 2000 3000 4000
```

The script writes group summaries to `benchmark/fastserve/results/summary.json`
and per-request measurements to
`benchmark/fastserve/results/requests.json`. Both files are atomically updated
after every completed RPS group so interrupted sweeps retain finished results.

The script only uses dependencies already included with Mini-SGLang; no extra
benchmark installation is required.

For high-backlog tests, raise the client machine's file-descriptor limit before
running. `--max-client-connections 0` leaves the HTTP connection pool unlimited
so the client does not impose an extra concurrency cap.
