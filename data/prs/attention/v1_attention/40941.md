# PR #40941: [Attention][TurboQuant] Share dequant buffers, eliminate float16_copy

- field: attention  | area: v1/attention
- author: bhoomit
- state: MERGED  created: 2026-04-26T22:49:44Z  merged: 2026-04-27T05:48:36Z
- url: https://github.com/vllm-project/vllm/pull/40941
- labels: ready, v1

## Body

# TurboQuant Backend Optimizations

**Builds on top of**: shared-decode-buffer PR (#40655)

## Summary

Backend-level optimizations for the TurboQuant KV cache attention backend, reducing memory usage and improving prefill latency for long-context workloads.

**Key results** (L4 TP=4, GPTQ-Int4 + TQ k8v4):
- **Saves 57 GB memory at 1M context** — shared dequant buffers via WorkspaceManager in the **continuation prefill path** (60× reduction; the earlier PR #40655 shared buffers in the decode path only)
- **Eliminates float16_copy kernel** in decode (stage2 OUTPUT_FP16) — removes redundant copy after stage2 reduction
- **Enables CUDA Graph capture for continuation prefill** — the old per-layer `_tq_k_dequant_buf` grew lazily on first long request (triggering cudaMalloc inside a captured graph crashes); WorkspaceManager pre-allocates at profiling time so buffer addresses are fixed during replay
- **Fixes output_buf dtype mismatch** in shared-buffer decode path
- **117 existing TQ unit tests pass**

## Changes

### Memory Optimizations

**1. Shared dequant buffers via WorkspaceManager** (P0 — critical)

`_continuation_prefill` allocated per-layer FP16 dequant buffers (`_tq_k_dequant_buf`, `_tq_v_dequant_buf`) that grew to the max continuation length seen. With 60 active TQ layers × 2 buffers, memory grew with O(layers × context_len):

| Context | Per-layer (old) | Shared (this PR) | Savings |
|---|---|---|---|
| 8K | 480 MB | 8 MB | 472 MB |
| 32K | 1.9 GB | 32 MB | 1.87 GB |
| 128K | 7.5 GB | 125 MB | 7.4 GB |
| 1M | 58.6 GB | 976 MB | **57.6 GB** |

Fix: Use `current_workspace_manager().get_simultaneous()` to allocate dequant buffers from the shared workspace pool. This ensures:
- Buffers are shared across all layers (workspace is global per ubatch)
- Buffer sizes participate in GPU memory profiling (correct KV capacity calculation)
- **Required for CUDA Graph capture** — per-layer growth pattern is incompatible with CG (buffers must be fixed-size at capture time)

The shared-decode-buffer PR (#40655) already shares the decode-path buffers (`mid_o_buf`, `output_buf`, `lse_buf`) via WorkspaceManager in `_decode_attention`. This PR extends the same pattern to the **prefill/continuation** dequant buffers — different buffers in a different code path.

### Prefill Optimizations

**2. Remove redundant `.zero_()` on dequant buffers**

The `_tq_full_dequant_kv` kernel writes all positions up to `cached_len`, and we only read `[:cached_len]` afterwards. The trailing positions are never read. Removing `.zero_()` eliminates 2 memset kernels per layer per continuation.

**3. fp16 Hadamard for inverse rotation**

The continuation prefill path cast keys to fp32 for rotation (`k_flat.float() @ Pi`), then cast back to fp16. Since the Hadamard matrix has ±1/√D entries (perfectly representable in fp16), the matmul can be done in fp16 directly, halving bandwidth and using fast fp16 tensor cores.

**4. Pre-sized continuation buffer, eliminate allocation storm**

`_continuation_prefill` did `.transpose(0,1).contiguous()` (materializing a full copy) + `torch.cat()` (another full allocation) per layer per continuation. At 15K context, each call allocated ~200 MB of throwaway tensors.

Fix: Pre-allocate `k_full` and `v_full` at `(seq_len, Hk, D)`, copy cached and new K/V into slices. Eliminates `.contiguous()` copies and `torch.cat` allocations.

**Impact**: -14% TTFT at 15K context (377ms → 326ms).

### Decode Optimizations

**5. Eliminate float16_copy kernel (stage2 dtype cast)**

The stage2 reduction kernel output in float32, then a separate `output.to(query.dtype)` launched a `float16_copy` kernel — 240K extra launches per profile window. Modified `_fwd_kernel_stage2` to accept an `OUTPUT_FP16` constexpr and cast on the final `tl.store`.

**output_buf dtype fix**: The shared-decode-buffer PR allocates `output_buf` as `torch.float32` via WorkspaceManager. Since change 5 writes directly in query dtype (float16), this caused a dtype mismatch on every step, forcing re-allocation. Fixed by changing the WorkspaceManager `output_buf` allocation to use `query.dtype`.

### Kernel Launch Reduction

**6. Cache arange and cu_seqlens**

Pre-built an arange tensor on `self` and sliced views per request, eliminating per-request `torch.arange` kernel launches in the continuation-decode fast path (used by speculative decoding verification). Cached cu_seqlens buffers on `self` instead of creating `torch.tensor([0, q_len])` per continuation prefill call.

## Benchmarks

All benchmarks on L4 TP=4, GPTQ-Int4, Qwen2.5-32B.

### Decode Throughput (TPOT)

| Mode | KV Cache | Base (shared-buffer) | + This PR | Delta |
|---|---|---|---|---|
| enforce_eager | TQ k8v4 | 20.4 tok/s | 21.0 tok/s | +3% (within noise) |
| CUDA Graphs | TQ k8v4 | 43.9 tok/s | 44.0 tok/s | +0.2% (within noise) |

Decode throughput is within noise in both eager and CG modes. The float16_copy elimination (change 5) removes a redundant kernel launch but its latency contribution (<1µs) is below measurement resolution at batch=1.

### TTFT / Continuation Prefill

Multi-turn benchmark (prefix_caching=OFF, unique prompts, 3 runs median):

| Context | Tokens | Baseline | Optimized | Delta | Path |
|---|---|---|---|---|---|
| 2K | 3405 | 2067 ms | 2069 ms | 0% | first-chunk only |
| 4K | 6683 | 4137 ms | 4150 ms | 0% | first-chunk only |
| 8K | 13244 | 8393 ms | 8395 ms | 0% | continuation |

No measurable TTFT improvement in eager mode — the WorkspaceManager per-call overhead balances the per-layer getattr savings. The value of change 1 is **memory**, not speed.

### Memory Savings (change 1)

Per the table above, shared dequant buffers reduce continuation-prefill memory from O(layers × context) to O(context):
- 8K context: saves 472 MB
- 128K context: saves 7.4 GB
- 1M context: saves 57.6 GB

Without this change, TQ continuation prefill at >128K is impossible on most GPU configurations.

### CUDA Graphs

| Config | Eager (ms/tok) | CG (ms/tok) | CG Speedup |
|---|---|---|---|
| Base (shared-buffer) | 49.1 | 22.8 | 2.15× |
| + This PR | 47.5 | 22.7 | 2.10× |

CG eliminates Python dispatch overhead, giving 2× speedup. This PR does not improve CG throughput measurably at batch=1. The primary CG-related benefit of change 1 is **enabling** CG capture for the continuation path (per-layer growth is incompatible with graph replay).

### Accuracy (unchanged)

| Config | GSM8K (Blackwell) |
|---|---|
| FP8 KV | 82.4% |
| TQ k8v4 | 81.7% (±1.1pp) |
| TQ 4bit_nc | 80.4% |
| TQ 3bit_nc | 79.4% |

All accuracy unchanged — these are pure computation/memory optimizations.

## Relationship to Other PRs

**Shared decode buffers PR (#40655)** — **this PR builds on top of it**. That PR shares decode-path buffers (`mid_o_buf`, `output_buf`, `lse_buf`) via WorkspaceManager in `_decode_attention`. This PR:
1. Extends WorkspaceManager sharing to the continuation-prefill dequant buffers (change 1)
2. Fixes a dtype mismatch in that PR's `output_buf` allocation (`torch.float32` → `query.dtype`) to support the in-kernel fp16 cast (change 5)
3. Adds all the prefill/decode optimizations listed above


## Testing

- 117 existing TQ unit tests pass (no changes to tests needed)
- Smoke tested on L4 TP=4 with GPTQ-Int4 + TQ k8v4
- Verified output correctness: "4. This is the sum of 2 and 2..."
- Benchmarked decode throughput on shared-buffer baseline
- Benchmarked TTFT at 1K/4K/8K/15K/30K context
- Validated CUDA graph capture at ml=512
- nsys profiled before/after for kernel-level verification

## AI Assistance Disclosure

AI assistance was used for code review, checklist validation,
and conflict resolution during rebasing. All code changes were reviewed and
validated by the submitter.


## Comments

(no discussion comments)