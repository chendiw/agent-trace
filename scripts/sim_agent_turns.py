#!/usr/bin/env python3
"""
sim_agent_turns.py — simulate a multi-turn agent task against a local SGLang endpoint.

Sends N turns of a coding-agent conversation where each turn extends the shared
prefix from the previous turn. After each turn, reads SGLang metrics to show:
  - tokens in / out
  - wall-clock time
  - TTFT (from metrics endpoint, cumulative sum / count)
  - cache hit rate (rises across turns as shared prefix grows)
  - KV pool state (available, evictable, used)

Usage:
    python3 sim_agent_turns.py [--url http://localhost:30000] [--model MODEL]
"""
import sys
import time
import json
import argparse
import urllib.request

SYSTEM_PROMPT = (
    "You are an expert Python software engineer. "
    "You have access to tools: bash(cmd), read_file(path), write_file(path, content). "
    "Always reason step by step before calling a tool. "
    "Keep responses concise."
)

TURNS = [
    {"role": "user", "content": "Task: fix the division-by-zero bug in utils.py."},
    {"role": "user", "content": "Tool result: def divide(a, b):\\n    return a / b"},
    {"role": "user", "content": "Tool result: File written successfully."},
]


def chat(base_url: str, model: str, messages: list, max_tokens: int = 100) -> dict:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def read_metrics(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        raw = resp.read().decode()
    out = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for key in (
            "sglang:cache_hit_rate{",
            "sglang:kv_available_tokens{",
            "sglang:kv_evictable_tokens{",
            "sglang:kv_used_tokens{",
            "sglang:time_to_first_token_seconds_count{",
            "sglang:time_to_first_token_seconds_sum{",
        ):
            if key in line:
                name = line.split("{")[0].replace("sglang:", "")
                out[name] = float(line.rsplit(" ", 1)[-1].strip())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    print(f"endpoint: {args.url}  model: {args.model}\n")
    print(f"{'turn':>4}  {'in':>5}  {'out':>5}  {'wall(s)':>8}  {'hit_rate':>9}  {'kv_avail':>9}  {'kv_evict':>9}")
    print("-" * 70)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    prev_ttft_count = 0.0
    prev_ttft_sum = 0.0

    for i, turn in enumerate(TURNS):
        history.append(turn)
        t0 = time.time()
        resp = chat(args.url, args.model, history, args.max_tokens)
        wall = time.time() - t0

        usage = resp["usage"]
        assistant_msg = resp["choices"][0]["message"]
        history.append(assistant_msg)

        m = read_metrics(args.url)
        hit_rate = m.get("cache_hit_rate", float("nan"))
        avail = m.get("kv_available_tokens", float("nan"))
        evict = m.get("kv_evictable_tokens", float("nan"))

        # incremental TTFT for this turn
        ttft_count = m.get("time_to_first_token_seconds_count", 0.0)
        ttft_sum = m.get("time_to_first_token_seconds_sum", 0.0)
        delta_count = ttft_count - prev_ttft_count
        delta_sum = ttft_sum - prev_ttft_sum
        ttft_this = (delta_sum / delta_count) if delta_count > 0 else float("nan")
        prev_ttft_count, prev_ttft_sum = ttft_count, ttft_sum

        print(
            f"{i+1:>4}  {usage['prompt_tokens']:>5}  {usage['completion_tokens']:>5}"
            f"  {wall:>8.2f}  {hit_rate:>9.3f}  {avail:>9.0f}  {evict:>9.0f}"
            f"  ttft={ttft_this:.3f}s"
        )

    print("\nfinal assistant turn:")
    print(history[-1].get("content", "")[:200])


if __name__ == "__main__":
    main()
