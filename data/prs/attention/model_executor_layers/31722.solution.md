# PR #31722: [PERF] Speed-up of GDN attention decode part (Qwen3-Next)

- field: attention  | area: model_executor_layers
- author: vadiklyutiy
- state: MERGED  created: 2026-01-05T12:37:58Z  merged: 2026-01-06T17:32:47Z
- url: https://github.com/vllm-project/vllm/pull/31722
- labels: performance, ready, qwen

# Issue

## Problem

The Triton kernel `fused_recurrent_gated_delta_rule_fwd` sets its V-dimension block size `BV = min(next_power_of_2(V), 8)`, capping the value-block at 8 elements. With the head value dimension V used by Qwen3-Next's gated delta-net far larger than 8, this forces `NV = cdiv(V, BV)` into many small blocks, spawning excessive Triton programs that each do tiny amounts of work. The result is poor GPU occupancy and high per-block overhead during the recurrent decode step.

## Example

For Qwen3-Next shapes on H200 at batch size 1024, the decode kernel took 345 μs (TP=4) before; the small BV=8 cap meant the V dimension was chopped into many 8-wide blocks instead of a few wide ones.

## Use case

Autoregressive decode (single-token-per-step) serving of Qwen3-Next linear-attention / gated delta-net models, where this fused recurrent kernel runs every decode step.

## Resources

GPU compute occupancy and kernel-launch/grid overhead — the kernel is dominated by too many under-utilized Triton program instances along the V axis.

## Type of users

Users serving Qwen3-Next (GDN / linear-attention, incl. FP8 MoE variants) with tensor parallelism (TP=2/TP=4) on modern NVIDIA GPUs (H200, B200).

## Kind of inputs

Decode-phase workloads with large batch/concurrency (32–1024) and short per-step token length (T=1), where the value head dimension V is much greater than 8.

## Mechanism with the issue

In `vllm/model_executor/layers/fla/ops/fused_recurrent.py`, the `BV` block size is clamped to 8, so `NV = cdiv(V, BV)` becomes large and the Triton grid launches many narrow V-blocks. Each program processes only 8 value elements, wasting lanes and inflating scheduling overhead for the memory-bound recurrent decode loop.

# Solution

## Fix in words

The diff raises the V-block-size cap in `fused_recurrent_gated_delta_rule_fwd` from `BV = min(next_power_of_2(V), 8)` to `min(next_power_of_2(V), 32)`. With BV now up to 32, `NV = cdiv(V, BV)` shrinks 4x, so the Triton kernel launches far fewer, wider V-blocks. Each program handles more value elements, improving GPU occupancy and cutting grid/launch overhead. Net effect: the decode kernel runs ~1.4–1.6x faster across batch sizes on H200/B200, yielding ~16.8% higher end-to-end decode throughput.

## Author's solution

The author sped up the GDN decode kernel `fused_recurrent_gated_delta_rule_fwd` for Qwen3-Next shapes, benchmarking it in isolation on H200/B200 (TP=2 and TP=4, batch 32–1024) and showing ~1.4–1.6x kernel speedups. End-to-end, serving Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 at TP=2 with expert parallelism and async scheduling, output token throughput rose from 23891 to 27904 tok/s, a 16.8% improvement.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fla/ops/fused_recurrent.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fla/ops/fused_recurrent.py b/vllm/model_executor/layers/fla/ops/fused_recurrent.py
index 0f27504780ac..91b07129dca8 100644
--- a/vllm/model_executor/layers/fla/ops/fused_recurrent.py
+++ b/vllm/model_executor/layers/fla/ops/fused_recurrent.py
@@ -189,7 +189,7 @@ def fused_recurrent_gated_delta_rule_fwd(
     B, T, H, K, V = *k.shape, v.shape[-1]
     HV = v.shape[2]
     N = B if cu_seqlens is None else len(cu_seqlens) - 1
-    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
+    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
     NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
     assert NK == 1, "NK > 1 is not supported yet"
     num_stages = 3

```
