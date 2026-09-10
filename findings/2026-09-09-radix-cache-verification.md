# Radix cache verification — 2026-09-09

**Setup:** SGLang 0.5.19, Qwen2.5-0.5B-Instruct, RTX 4090, `--max-total-tokens 2000`,
`--attention-backend triton --sampling-backend pytorch`

## Question

Does the radix prefix cache actually avoid re-allocating KV slots for a repeated prompt?

## Method

Send the same 57-token prompt twice with `max_tokens=5` (minimal output to isolate the
prefill effect). Before and after each request, read `kv_available_tokens` from `/metrics`.
The server is single-tenant, so no other requests race between the metric reads.

If caching works:
- Request 1: `kv_available` drops by ≈ prompt + output tokens (all fresh)
- Request 2: `kv_available` drops by ≈ output tokens only (prompt served from radix tree)

## Result

```
Request 1  in=57 out=5
  kv_available: 1 → 117  (delta=+116)

Request 2  in=57 out=5
  kv_available: 117 → 112  (delta=-5)
```

**Request 1** detail: pool was at 1 token available. Engine evicted ~178 old cached
tokens to make room, then allocated 62 new slots (57 prompt + 5 output). After completion
those 62 tokens joined the radix tree. Net: +116 available.

**Request 2**: only 5 slots consumed — the 5 output tokens. The 57 prompt tokens were
served from the radix tree with no new allocation.

## Conclusion

Radix cache is active. A cache-hit request consumes only `output_tokens` new KV slots,
not `prompt_tokens + output_tokens`. The 57-token prompt cost zero additional allocation
on the second call.

## What does NOT work as a confirmation

- `cache_hit_rate` gauge: windowed (fires every 40 decode steps); shows 0.0 at low
  request volume even when the cache is clearly working.
- TTFT comparison across turns: too noisy at this model size — per-request scheduling
  overhead dominates, not compute. TTFT may decrease for reasons unrelated to caching.

## Instrument for Phase 0

Track `kv_available` delta per LLM call during a real agent run:
- Large delta (≈ full prompt + output): cache miss, full prefill recompute
- Small delta (≈ output only): cache hit, prefix served from radix tree

The difference between these two cases is the recompute waste the study is measuring.

## Script

`scripts/verify_radix_cache.py`
