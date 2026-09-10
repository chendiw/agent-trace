#!/usr/bin/env python3
"""
verify_radix_cache.py — confirm radix cache is active by sending the same prompt twice.

If the radix cache works:
  - Request 1: kv_available drops by ~N (N tokens allocated fresh)
  - Request 2: kv_available does NOT drop (tokens already in radix tree, no new allocation)

Usage:
    python3 verify_radix_cache.py [--url http://localhost:30000] [--model MODEL]
"""
import sys
import json
import argparse
import urllib.request

PROMPT = (
    "You are an expert Python software engineer. "
    "Write a Python function that implements binary search on a sorted list. "
    "Include docstring and edge cases."
)


def chat(base_url: str, model: str, prompt: str, max_tokens: int = 5) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def kv_available(base_url: str) -> float:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        for line in resp.read().decode().splitlines():
            if "sglang:kv_available_tokens{" in line:
                return float(line.rsplit(" ", 1)[-1].strip())
    return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    print(f"endpoint: {args.url}  model: {args.model}")
    print(f"prompt: {PROMPT[:60]}...\n")
    print("Using max_tokens=5 to minimize output tokens and isolate the prefill effect.\n")

    before1 = kv_available(args.url)
    r1 = chat(args.url, args.model, PROMPT, max_tokens=5)
    after1 = kv_available(args.url)
    u1 = r1["usage"]

    print(f"Request 1  — in={u1['prompt_tokens']} out={u1['completion_tokens']}")
    print(f"  kv_available: {before1:.0f} → {after1:.0f}  (delta={after1 - before1:+.0f})")

    before2 = kv_available(args.url)
    r2 = chat(args.url, args.model, PROMPT, max_tokens=5)
    after2 = kv_available(args.url)
    u2 = r2["usage"]

    print(f"\nRequest 2  — in={u2['prompt_tokens']} out={u2['completion_tokens']}")
    print(f"  kv_available: {before2:.0f} → {after2:.0f}  (delta={after2 - before2:+.0f})")

    delta1 = after1 - before1
    delta2 = after2 - before2

    print("\nConclusion:")
    if abs(delta2) <= 5 and abs(delta1) > 5:
        print("  CACHE ACTIVE — request 2 consumed ~0 new KV slots (prompt served from radix tree)")
    elif delta2 < 0 and abs(delta2) < abs(delta1) * 0.3:
        print("  CACHE LIKELY ACTIVE — request 2 consumed far fewer slots than request 1")
    else:
        print("  CACHE NOT CONFIRMED — both requests consumed similar KV slots")
        print(f"  expected: delta2 ≈ 0, got delta1={delta1:+.0f}, delta2={delta2:+.0f}")


if __name__ == "__main__":
    main()
