# PR #31167: [Perf] Remove blocking copy in GDN Attention

- field: attention  | area: v1/attention
- author: benchislett
- state: MERGED  created: 2025-12-22T17:45:10Z  merged: 2025-12-22T22:25:22Z
- url: https://github.com/vllm-project/vllm/pull/31167
- labels: performance, ready, v1

## Body

## Purpose

Remove a blocking copy in the GDN attention that stops async scheduling from working properly for Qwen3-Next.
Nets +1% speedup at BS1 on 4xB200. The main purpose of this PR is to remove blockers for making async scheduling + MTP fully non-blocking for Qwen3-Next (follow-ups to come).

## Testing

GSM8k:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8469|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.7961|±  |0.0111|
```



## Comments

(no discussion comments)