"""Send two concurrent request waves; check server logs for evict/restore cycles."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1919")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    with urlopen(args.url + "/v1/models", timeout=30) as response:
        model = json.load(response)["data"][0]["id"]

    def request(index):
        prompt = f"Request {index}. " + "The quick brown fox jumps over the lazy dog. " * 40
        prompt += "\nSummarize the preceding text in one sentence."
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0,
            "ignore_eos": True,
        }
        req = Request(
            args.url + "/v1/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=600) as response:
            result = json.load(response)
        text = result["choices"][0]["message"]["content"]
        if not text:
            raise RuntimeError("Server returned an empty completion")
        return text

    for wave in range(2):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            replies = list(executor.map(request, range(args.concurrency)))
        print(f"wave={wave + 1} completed={len(replies)} seconds={time.perf_counter() - start:.2f}")
        print(replies[0][:160])
    print("Check server logs: each wave should contain KV-MoE evict and KV-MoE restore.")


if __name__ == "__main__":
    main()
