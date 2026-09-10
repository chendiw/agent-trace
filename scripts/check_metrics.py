#!/usr/bin/env python3
"""
check_metrics.py — verify the 3 SGLang metrics needed for Phase 0 are exposed.

Usage:
    python3 check_metrics.py [--url http://localhost:30000]

Exits 0 if all 3 metrics are present, 1 otherwise.
"""
import sys
import argparse
import urllib.request

REQUIRED = [
    "sglang:time_to_first_token_seconds",
    "sglang:inter_token_latency_seconds",
    "sglang:cache_hit_rate",
]


def fetch_metrics(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        return resp.read().decode()


def check(base_url: str) -> bool:
    try:
        raw = fetch_metrics(base_url)
    except Exception as e:
        print(f"ERROR: could not reach {base_url}/metrics — {e}")
        return False

    found = {}
    for line in raw.splitlines():
        for metric in REQUIRED:
            if line.startswith(metric):
                found[metric] = line.rsplit(" ", 1)[-1].strip()

    ok = True
    for metric in REQUIRED:
        if metric in found:
            print(f"  OK  {metric} = {found[metric]}")
        else:
            print(f"MISS  {metric}")
            ok = False

    # also print pool state
    for line in raw.splitlines():
        for key in ("kv_available_tokens{", "kv_evictable_tokens{", "kv_used_tokens{"):
            if key in line:
                name = line.split("{")[0]
                val = line.rsplit(" ", 1)[-1].strip()
                print(f"      {name} = {val}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:30000")
    args = parser.parse_args()

    ok = check(args.url)
    sys.exit(0 if ok else 1)
