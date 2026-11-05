# PR #35219: [BUGFIX][Mamba][Qwen3.5] Zero freed SSM cache blocks on GPU

- field: attention  | area: v1/core
- author: vadiklyutiy
- state: MERGED  created: 2026-02-24T19:21:59Z  merged: 2026-03-10T10:32:20Z
- url: https://github.com/vllm-project/vllm/pull/35219
- labels: bug, ready, v1, qwen, ready-run-all-tests

## Body

### Essential problem

Fixes https://github.com/vllm-project/vllm/issues/35138
Workaround for https://github.com/Dao-AILab/flash-attention/issues/1974

Hybrid models (e.g. Qwen3.5-397B-A17B) share a unified block pool between attention (fp8/fp16) and Mamba/SSM (fp32) layers. When a block previously used by Mamba (fp32 state) is reallocated to an attention layer with a smaller dtype, leftover fp32 bit patterns can appear as NaN/Inf in the new dtype. Attention kernels (FlashAttn3, FlashInfer-TRTLLM, etc.) use multiply-by-zero masking for unused positions, which does **not** clear NaN (`0 * NaN = NaN`). The stale NaN then propagates across **all** requests sharing the same KV-cache block, causing progressive accuracy degradation over time.

### What this PR does

Zeroes GPU memory of freshly allocated **full-attention** KV-cache blocks **before** they are used, but **only for hybrid models** (models with Mamba layers). Mamba/SSM blocks are not zeroed (they overwrite their state fully on each step). The approach:

1. **Scheduler side** — `SingleTypeKVCacheManager` tracks block IDs allocated since the last scheduling step (only for `FullAttentionSpec` layers). After scheduling, the scheduler drains these IDs into `SchedulerOutput.new_block_ids_to_zero`, gated behind `self.has_mamba_layers`.

2. **Worker side** — `GPUModelRunner._update_states()` receives the block IDs and calls `_zero_block_ids()`, which launches a single Triton kernel (`_zero_kv_blocks_kernel`) to zero the corresponding memory across all KV-cache segments in one GPU launch.

3. **Optimized zeroing** — A one-time `_init_kv_zero_meta()` precomputes absolute byte addresses of all KV-cache segments (handling both block-dim-0 and block-dim-1 layouts, multi-buffer backends, and virtual block splitting). Block IDs are transferred via pre-allocated pinned memory to overlap the H2D copy with kernel launch. This avoids 15 separate `index_fill_` calls (For Qwen3.5-379B, one per layer).

4. **CuMem compatibility** — `_init_kv_zero_meta()` is called in `gpu_worker.py` **outside** the CuMem pool context, so the bookkeeping tensors (segment addresses, block-ID buffers) use the standard PyTorch allocator and survive sleep/wake cycles.

### Scope

- **Hybrid models only** — The zeroing is gated by `self.has_mamba_layers` in the scheduler. Non-hybrid (pure attention) models are completely unaffected.
- **Only freshly allocated blocks** — Prefix-cached blocks (cache hits) are not zeroed, preserving prefix caching correctness and performance.
- **Only `FullAttentionSpec` blocks** — Mamba/SSM blocks are not zeroed (they overwrite their state fully each step); only attention blocks that may inherit stale Mamba fp32 data are cleared.

### Performance overhead

Measured on B200, Qwen/Qwen3-0.6B, BS=500:

| Phase | Blocks zeroed | Median latency | vs. forward step |
|-------|--------------|----------------|------------------|
| Prefill (BS ~8K) | ~515 blocks (~920 MiB) | ~170 μs | ~1% of 18ms step |
| Decode (BS ~500) | ~30 blocks (~55 MiB) | ~15 μs | ~0.1% of 13ms step |

End-to-end benchmark on B200 (Qwen3.5-397B-A17B-FP8, TP=1 PP=1 DP=8, 2048 prompts, 500 output tokens) showed no measurable throughput degradation (output tokens/s within ±2% noise).


### Test plan

- **The original issue (#35138) reproduction test passes**: 5 consecutive GSM8K evaluations on Qwen3.5-397B-A17B-FP8 (TP=4, FlashInfer backend) — accuracy stable at ~90% across all runs with no degradation. Previously, accuracy would drop significantly after the first run due to NaN accumulation.
- End-to-end throughput benchmark (B200, 2048 prompts) confirms no performance regression.


## Comments

### LucasWilkinson @ 2026-02-24T19:37:25Z

Could you please provide data on possible perf overhead? also with async scheduling I think it may be risky to zero on free, we may need to move this into the model runner to ensure it ends up in the correct order in the GPU stream 
---
### vadiklyutiy @ 2026-02-24T20:04:57Z

> we may need to move this into the model runner to ensure it ends up in the correct order in the GPU stream

Actual zeroing happens in `GPUModelRunner._update_states`. "On free" we just collect corresponding blocks. 
Or didn't get you comment? 


---
### vadiklyutiy @ 2026-02-24T21:39:58Z

Below is from slack discussion. But I think it's worth attention and sharing here

_Seems both FlashAttn and trt-llm attn use mul by 0 to mask not used values. 
To be sure that its correct we must guarantee that no NaN. In general case masking by mul by 0 is incorrect.
In my flavor, it should/must be fixed in kernels.
Seems even common full attn can produce NaN for some corner cases. It is likely ok to produce NaN(and garbage tokens) for one specific request, but with kernels that not tolerant to NaN, we propagate this problem to another good requests with subsequent garbage for all requests_.
---
### tdoublep @ 2026-02-24T22:05:44Z

How does this interact with prefix caching? If we zero out blocks when their ref_cnt hits zero, doesn't that mean they can't be re-used if something comes along later that gets a cache hit? 

Wouldn't it to be better to detect the event when we use a block for attention that was previously used for mamba (in some other dtype) and zero it out at that point? 
---
### vadiklyutiy @ 2026-02-24T23:52:19Z

> Could you please provide data on possible perf overhead? also with async scheduling I think it may be risky to zero on free, we may need to move this into the model runner to ensure it ends up in the correct order in the GPU stream

Ran on B200
```
VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --port 8000 -tp 1 -pp 1 -dp 8 --enable-expert-parallel --language-model-only --reasoning-parser qwen3 --kv-cache-dtype fp8 --stream-interval=100
```
```
 vllm bench serve --backend vllm --model Qwen/Qwen3.5-397B-A17B-FP8 \
            --endpoint /v1/completions --dataset-name random --random-input 2 \
            --random-output 500  --max-concurrency 512 --num-prompt 2048 \
            --ignore-eos --temperature=0.0 
```

With changes and without changes the Output Tokens vary around 15000+-300. Definitely nothing dramatical from perf point of view but not exact numbers. 
---
### vadiklyutiy @ 2026-02-25T00:42:01Z

> How does this interact with prefix caching? If we zero out blocks when their ref_cnt hits zero, doesn't that mean they can't be re-used if something comes along later that gets a cache hit?
> 
> Wouldn't it to be better to detect the event when we use a block for attention that was previously used for mamba (in some other dtype) and zero it out at that point?

Good catch I broke the prefix caching :/
Move to draft to think better way
---
### vadiklyutiy @ 2026-02-25T02:19:07Z

> How does this interact with prefix caching? If we zero out blocks when their ref_cnt hits zero, doesn't that mean they can't be re-used if something comes along later that gets a cache hit?
> 
> Wouldn't it to be better to detect the event when we use a block for attention that was previously used for mamba (in some other dtype) and zero it out at that point?

I redo this PR similar to what @tdoublep proposed. 

I decided to zero out every new block, whether it comes from attention or from the SSM.

Justification: attention can also produce NaNs in certain corner cases. Getting garbage for one specific request is likely acceptable, but without zeroing, the NaN could propagate to all requests.
---
### vadiklyutiy @ 2026-02-25T02:19:28Z

pls take a look 
---
### pavanimajety @ 2026-02-25T21:19:53Z

@heheda12345 for review as well
---
### vadiklyutiy @ 2026-02-25T22:56:41Z

I optimized zeroing code:

B200, BS=500, Qwen/Qwen3-0.6B


## Prefill
The forward step takes around 18ms (BS around 8K)
| Range | Proj Avg | Proj Med | Proj Min | Proj Max |
|-------|----------|----------|----------|----------|
| :_zero_block_ids (n=513, 919296 KiB) | 206.134 μs | 188.351 μs | 154.976 μs | 282.815 μs |
| :_zero_block_ids (n=517, 926464 KiB) | 180.867 μs | 156.303 μs | 149.440 μs | 277.055 μs |
| :_zero_block_ids (n=515, 922880 KiB) | 201.732 μs | 174.000 μs | 153.696 μs | 282.143 μs |
| :_zero_block_ids (n=518, 928256 KiB) | 172.525 μs | 151.424 μs | 141.536 μs | 258.783 μs |
| :_zero_block_ids (n=514, 921088 KiB) | 159.638 μs | 155.999 μs | 146.336 μs | 190.623 μs |
| :_zero_block_ids (n=519, 930048 KiB) | 158.889 μs | 155.151 μs | 146.815 μs | 179.775 μs |
| :_zero_block_ids (n=516, 924672 KiB) | 207.849 μs | 194.879 μs | 165.471 μs | 263.199 μs |

## Decode
The forward step takes around 13ms (BS around 500)
| Range | Proj Avg | Proj Med | Proj Min | Proj Max |
|-------|----------|----------|----------|----------|
| :_zero_block_ids (n=25, 44800 KiB) | 13.599 μs | 13.392 μs | 12.639 μs | 15.200 μs |
| :_zero_block_ids (n=29, 51968 KiB) | 15.089 μs | 15.040 μs | 13.952 μs | 16.224 μs |
| :_zero_block_ids (n=31, 55552 KiB) | 15.453 μs | 15.232 μs | 13.952 μs | 19.904 μs |
| :_zero_block_ids (n=32, 57344 KiB) | 15.710 μs | 15.728 μs | 14.176 μs | 19.168 μs |
| :_zero_block_ids (n=33, 59136 KiB) | 16.111 μs | 16.272 μs | 14.912 μs | 17.184 μs |
| :_zero_block_ids (n=34, 60928 KiB) | 16.555 μs | 16.608 μs | 15.200 μs | 17.952 μs |

---
### vadiklyutiy @ 2026-02-26T00:58:47Z

I made all changes that I wanted. 
Right now the overhead is really small. 
Please take a look
---
### vadiklyutiy @ 2026-02-26T15:04:54Z

> Thanks for your contribution! I think we can find the blocks that we need to clear from gpu model runner.
> We can get NewRequestData.block_ids and CachedRequestData.new_block_ids which are the list of newly allocated blocks.
> Within these blocks, the num_computed_tokens / block_size tokens of NewRequestData and resumed_req_ids are cache hit blocks that we shouldn't clear, and all other block can be set to zero.

@heheda12345 Thank you for your comment. I though about compute(reconstruct) block ids for zeroing in flight inside gpu model runner. 
But it seems a little bit more complicated by implementation and fragile. 
Complication. As example: seems for cross-attention blocks `num_computed_tokens / block_size` doesn't work and need more sophisticated approach. 
Fragile. Some changes in management of cache or prefix caching approach might broke such "reconstruction". For example caching partial block. 
So, I think it is more reliable to make zeroing directly. 
---
### vadiklyutiy @ 2026-02-26T15:24:48Z

> And IMO we only need this fix when different kv cache use different dtype. Better to only run this step for this case (and it should be easy to add new case later)

I did it intentionally for full attention.

**Justification.** 
1. It is confirmed that trtllm-gen attn and FA3 expect no NaN in whole block. Otherwise they likely propagate NaN to request.  Also I talked with author of trtllm-gen and seems fixing it in kernel brings very big performance overhead. 
2. May NaN appears in full attn? Yes. Although models do normalization, kernel itself use log-sum to avoid overflow, there is no 100% guarantee that there is no NaN. Seems for some corner cases with very long context, it possible. 

So, let consider the following scenario. In some big production system, after week of work comes very long request that produce NaN. This NaN poison the system and will be propagate to **ALL REQUEST**. Because this is very rare case when attn produce NaN, it would be total nightmare to debug it.

From opposite side, the overhead to zero after allocate is pretty small and fully solve the problem.  

---
### vadiklyutiy @ 2026-02-26T18:39:25Z

set ready tag to run CI
---
### vadiklyutiy @ 2026-02-27T11:56:24Z

> just a note to address
> 
> https://github.com/vllm-project/vllm/blob/b66a74649e40022c6b1180b98fbb1b6e4b4af74a/vllm/v1/attention/backends/flash_attn.py#L75

@tdoublep Could clarify where this info came from?
---
### tdoublep @ 2026-02-27T12:56:26Z

> > just a note to address
> > https://github.com/vllm-project/vllm/blob/b66a74649e40022c6b1180b98fbb1b6e4b4af74a/vllm/v1/attention/backends/flash_attn.py#L75
> 
> @tdoublep Could clarify where this info came from?

This code (restricting the block size for hybrid models if fp32 mamba state is used) was specifically added to workaround the same problem that this PR is addressing (for FlashAttention). I believe if we zero-out the fp32 mamba blocks before re-using them as attention blocks, we no longer need to restrict the block size in this way. 
---
### vadiklyutiy @ 2026-02-27T13:19:33Z

> > > just a note to address
> > > https://github.com/vllm-project/vllm/blob/b66a74649e40022c6b1180b98fbb1b6e4b4af74a/vllm/v1/attention/backends/flash_attn.py#L75
> > 
> > 
> > @tdoublep Could clarify where this info came from?
> 
> This code (restricting the block size for hybrid models if fp32 mamba state is used) was specifically added to workaround the same problem that this PR is addressing (for FlashAttention). I believe if we zero-out the fp32 mamba blocks before re-using them as attention blocks, we no longer need to restrict the block size in this way.

Do you know/could you clarify why restriction to 16,32,64 only helps to fix FlashAttn kernel to be tolerant to NaN?
---
### tdoublep @ 2026-02-27T13:35:35Z

@vadiklyutiy Good question. We found empirically that these block sizes do not suffer from the issue (see [this](https://github.com/vllm-project/vllm/issues/27264#issuecomment-3467959676) comment). However, I don't have any guarantee that it could not still happen tbh. That is why I am generally supportive of a more explicit solution like this. I would add that we haven't had any reports of NaN issue in the field with FlashAttention since introducing the block size constraint. 
---
### benchislett @ 2026-02-27T16:04:47Z

I believe it comes down to an internal heuristic in FlashAttention, where for sizes <= 64 it will use a different approach for loading data than for >64. I don't fully recall but it's a kernel specific change intended to work around a specific behaviour (the same thing that breaks FlashInfer I think)
---
### xinli-sw @ 2026-02-27T18:56:10Z

+1 to Ben, more details shared by Lucas:

FA does not use TMAs for those block sizes because kBlockN is already greater than 64: [reference to heuristic](https://github.com/vllm-project/flash-attention/blob/140c00c0241bb60cc6e44e7c1be9998d4b20d8d2/hopper/flash_api.cpp#L434-L444), otherwise it uses predicated cp.async which is safe
---
### xinli-sw @ 2026-02-27T19:10:01Z

So in a sense there's no "guarantee" that it will work in the future. Ideally all kernels should not do this multiply by 0 trick but same can be argued that FW's kv-cache should not have NaNs in them. (even in the hybrid kv-cache manager's fp32 -> fp16 case)
---
### vadiklyutiy @ 2026-03-02T01:55:15Z

> I am also concerned that the implementation seems over-optimized. Maybe it is justified but can we see the numbers from a simpler implementation of zero'ing the blocks?

I did 2 optimizations. 
1. Triton kernel. We allocate 15 tensors for Qwen3.5 and have to call 15 `index_fill_` instead of 1 triton kernel
2. Preallocated buffer of ids to zero. Collect ids to zero on CPU and transfer them to GPU. If don't do preallocation triton kernel launch only after mem transfer finish. And this overhead is around 500-600us what is pretty sufficient. 
---
### vadiklyutiy @ 2026-03-02T02:51:09Z

> So in a sense there's no "guarantee" that it will work in the future. Ideally all kernels should not do this multiply by 0 trick but same can be argued that FW's kv-cache should not have NaNs in them. (even in the hybrid kv-cache manager's fp32 -> fp16 case)

I discussed it with Julien, he said that kernel has to "manually" checks NaN and it will brings a big overhead. 

---
### mergify @ 2026-03-02T02:51:21Z

Hi @vadiklyutiy, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mergify @ 2026-03-03T15:51:35Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @vadiklyutiy.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### benchislett @ 2026-03-03T16:13:48Z

@vadiklyutiy I think we should limit this to FlashInfer and/or hybrid-only deployments, at least for the short-term where we want to de-risk existing FlashAttention-based workloads
---
### vadiklyutiy @ 2026-03-03T16:22:07Z

> @vadiklyutiy I think we should limit this to FlashInfer and/or hybrid-only deployments, at least for the short-term where we want to de-risk existing FlashAttention-based workloads

I am debugging current fails now
---
### benchislett @ 2026-03-03T16:26:18Z

Not just for bugs, but there are concerns over impacting performance for existing non-hybrid deployments. We should apply to hybrid models only for now, until we settle on a long-term fix
---
### vadiklyutiy @ 2026-03-03T18:07:18Z

> Not just for bugs, but there are concerns over impacting performance for existing non-hybrid deployments. We should apply to hybrid models only for now, until we settle on a long-term fix

With Triton kernel zeroing takes just 100us. 
---
### benchislett @ 2026-03-03T18:36:52Z

Nonetheless, I think we do not have sufficient justification to apply this to all deployments right now. I think we should limit this feature to a targeted bugfix for a known issue and try to merge asap, and we can revisit the non-hybrid case in a follow-up
---
### vadiklyutiy @ 2026-03-04T21:08:17Z

I see 2 fails 
[buildkite/ci/pr/crosslayer-kv-layout-distributed-nixlconnector-pd](https://buildkite.com/vllm/ci/builds/54477#019cba84-64c3-46ee-8e23-432afdc2acf6)[buildkite/ci/pr/distributed-nixlconnector-pd-accuracy-4-gpus](https://buildkite.com/vllm/ci/builds/54477#019cba84-6508-49cf-973b-f1e0f426bace)

Seems they are real in CI (retried several times) but when I tried to reproduce locally both passed successfully. Any ideas? 
---
### mergify @ 2026-03-04T21:54:24Z

Hi @vadiklyutiy, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mergify @ 2026-03-04T22:33:51Z

Hi @vadiklyutiy, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mergify @ 2026-03-04T23:16:31Z

Hi @vadiklyutiy, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### vadiklyutiy @ 2026-03-05T01:06:49Z

Because a lot of changes was introduced since PR was open, I fully updated PR description as below



### Essential problem

Fixes https://github.com/vllm-project/vllm/issues/35138
Workaround for https://github.com/Dao-AILab/flash-attention/issues/1974

Hybrid models (e.g. Qwen3.5-397B-A17B) share a unified block pool between attention (fp8/fp16) and Mamba/SSM (fp32) layers. When a block previously used by Mamba (fp32 state) is reallocated to an attention layer with a smaller dtype, leftover fp32 bit patterns can appear as NaN/Inf in the new dtype. Attention kernels (FlashAttn3, FlashInfer-TRTLLM, etc.) use multiply-by-zero masking for unused positions, which does **not** clear NaN (`0 * NaN = NaN`). The stale NaN then propagates across **all** requests sharing the same KV-cache block, causing progressive accuracy degradation over time.

### What this PR does

Zeroes GPU memory of freshly allocated **full-attention** KV-cache blocks **before** they are used, but **only for hybrid models** (models with Mamba layers). Mamba/SSM blocks are not zeroed (they overwrite their state fully on each step). The approach:

1. **Scheduler side** — `SingleTypeKVCacheManager` tracks block IDs allocated since the last scheduling step (only for `FullAttentionSpec` layers). After scheduling, the scheduler drains these IDs into `SchedulerOutput.new_block_ids_to_zero`, gated behind `self.has_mamba_layers`.

2. **Worker side** — `GPUModelRunner._update_states()` receives the block IDs and calls `_zero_block_ids()`, which launches a single Triton kernel (`_zero_kv_blocks_kernel`) to zero the corresponding memory across all KV-cache segments in one GPU launch.

3. **Optimized zeroing** — A one-time `_init_kv_zero_meta()` precomputes absolute byte addresses of all KV-cache segments (handling both block-dim-0 and block-dim-1 layouts, multi-buffer backends, and virtual block splitting). Block IDs are transferred via pre-allocated pinned memory to overlap the H2D copy with kernel launch. This avoids 15 separate `index_fill_` calls (For Qwen3.5-379B, one per layer).

4. **CuMem compatibility** — `_init_kv_zero_meta()` is called in `gpu_worker.py` **outside** the CuMem pool context, so the bookkeeping tensors (segment addresses, block-ID buffers) use the standard PyTorch allocator and survive sleep/wake cycles.

### Scope

- **Hybrid models only** — The zeroing is gated by `self.has_mamba_layers` in the scheduler. Non-hybrid (pure attention) models are completely unaffected.
- **Only freshly allocated blocks** — Prefix-cached blocks (cache hits) are not zeroed, preserving prefix caching correctness and performance.
- **Only `FullAttentionSpec` blocks** — Mamba/SSM blocks are not zeroed (they overwrite their state fully each step); only attention blocks that may inherit stale Mamba fp32 data are cleared.

### Performance overhead

Measured on B200, Qwen/Qwen3-0.6B, BS=500:

| Phase | Blocks zeroed | Median latency | vs. forward step |
|-------|--------------|----------------|------------------|
| Prefill (BS ~8K) | ~515 blocks (~920 MiB) | ~170 μs | ~1% of 18ms step |
| Decode (BS ~500) | ~30 blocks (~55 MiB) | ~15 μs | ~0.1% of 13ms step |

End-to-end benchmark on B200 (Qwen3.5-397B-A17B-FP8, TP=1 PP=1 DP=8, 2048 prompts, 500 output tokens) showed no measurable throughput degradation (output tokens/s within ±2% noise).


### Test plan

- **The original issue (#35138) reproduction test passes**: 5 consecutive GSM8K evaluations on Qwen3.5-397B-A17B-FP8 (TP=4, FlashInfer backend) — accuracy stable at ~90% across all runs with no degradation. Previously, accuracy would drop significantly after the first run due to NaN accumulation.
- End-to-end throughput benchmark (B200, 2048 prompts) confirms no performance regression.

---
### vadiklyutiy @ 2026-03-05T03:52:13Z

@all PR is ready, please take a look
---
### benchislett @ 2026-03-05T23:01:21Z

Not an expert on this, but LGTM
---
### voipmonitor @ 2026-03-07T07:29:38Z

Hello,

I suspect that this can be related to the flashinfer race condition issue caused by enabled PDL on blackwell which needs cutlass 4.3.0 and using synchronisations which seems to be misisng due to missing comlile flags. I was hunting similar NaN propagation issues in sglang for sm120 and the root cause was in race in flashinfer cutlass gemm, please check this

https://github.com/flashinfer-ai/flashinfer/pull/2716
---
### vadiklyutiy @ 2026-03-09T10:36:25Z

> I'd also still really like to see the E2E performance of doing it in a simpler way without the kernel and all the pre-computation stuff.

the optimizations assumed that zeroing will apply to not hybrid model also. For hybrid model the overhead should be smaller because blocksizes of kv-chache are much bigger in hybrid models. For Qwen3-0.6B overhead was around 10%. 
Lets me check what impact on hybrid models.  

---
### vadiklyutiy @ 2026-03-09T14:08:16Z

# KV-Cache Zeroing: Triton Kernel vs index_fill_ Benchmark Results

## Model & Hardware

- **Model**: Qwen/Qwen3.5-35B-A3B-FP8
- **GPU**: 1x B200 (TP=1)

## Server Command

```bash
VLLM_ENGINE_READY_TIMEOUT_S=1800 \
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --port 8000 \
    -tp 1 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --kv-cache-dtype fp8 \
    --stream-interval=100
```

## Client Commands

### Decode-heavy

```bash
vllm bench serve \
    --backend vllm \
    --model Qwen/Qwen3.5-35B-A3B-FP8 \
    --port 8000 \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input 2 \
    --random-output 500 \
    --max-concurrency 256 \
    --num-prompt 512 \
    --ignore-eos --temperature=0.0
```

### Prefill-heavy

```bash
vllm bench serve \
    --backend vllm \
    --model Qwen/Qwen3.5-35B-A3B-FP8 \
    --port 8000 \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input 8000 \
    --random-output 1 \
    --max-concurrency 4 \
    --num-prompt 32 \
    --ignore-eos --temperature=0.0
```

## Results

All runs performed after a warmup iteration of each workload.

### Decode-Heavy (input=2, output=500, concurrency=256, prompts=512)

#### Triton kernel (baseline)

| Run | Output tok/s | TPOT median (ms) | TTFT mean (ms) | Duration (s) |
|-----|-------------|-------------------|-----------------|--------------|
| 1   | 15,603      | 15.86             | 287.43          | 16.41        |
| 2   | 15,614      | 15.92             | 265.31          | 16.40        |
| 3   | 15,685      | 15.86             | 259.37          | 16.32        |
| **Mean** | **15,634** | **15.88**    | **270.70**      | **16.38**    |

#### index_fill_ (simplified)

| Run | Output tok/s | TPOT median (ms) | TTFT mean (ms) | Duration (s) |
|-----|-------------|-------------------|-----------------|--------------|
| 1   | 15,314      | 16.22             | 284.88          | 16.72        |
| 2   | 15,293      | 16.28             | 269.81          | 16.74        |
| 3   | 15,314      | 16.22             | 278.88          | 16.72        |
| **Mean** | **15,307** | **16.24**    | **277.86**      | **16.73**    |

#### Decode-heavy delta

| Metric | Triton | index_fill_ | Delta |
|--------|--------|-------------|-------|
| Output tok/s | 15,634 | 15,307 | **-2.1%** |
| TPOT median (ms) | 15.88 | 16.24 | **+2.3%** |

### Prefill-Heavy (input=8000, output=1, concurrency=4, prompts=32)

#### Triton kernel (baseline)

| Run | Total tok/s | Median TTFT (ms) | P99 TTFT (ms) | Req/s |
|-----|------------|-------------------|---------------|-------|
| 1   | 77,900     | 407.76            | 490.32        | 9.74  |
| 2   | 77,903     | 408.46            | 493.90        | 9.74  |
| 3   | 77,928     | 408.18            | 491.57        | 9.74  |
| **Mean** | **77,910** | **408.13**   | **491.93**    | **9.74** |

#### index_fill_ (simplified)

| Run | Total tok/s | Median TTFT (ms) | P99 TTFT (ms) | Req/s |
|-----|------------|-------------------|---------------|-------|
| 1   | 74,221     | 430.02            | 506.63        | 9.28  |
| 2   | 74,238     | 429.67            | 506.07        | 9.28  |
| 3   | 74,214     | 430.34            | 508.73        | 9.28  |
| **Mean** | **74,224** | **430.01**   | **507.14**    | **9.28** |

#### Prefill-heavy delta

| Metric | Triton | index_fill_ | Delta |
|--------|--------|-------------|-------|
| Total tok/s | 77,910 | 74,224 | **-4.7%** |
| Median TTFT (ms) | 408.13 | 430.01 | **+5.4%** (+22ms) |
| P99 TTFT (ms) | 491.93 | 507.14 | **+3.1%** (+15ms) |
| Req/s | 9.74 | 9.28 | **-4.7%** |

---
### vadiklyutiy @ 2026-03-09T15:11:58Z

below is a diff 

<details>
<summary>diff for `index_fill_`</summary>

```
$ git diff
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 572bf8a0c..dc7e5d78b 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -373,46 +373,6 @@ class ExecuteModelState(NamedTuple):
     slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


-@triton.jit
-def _zero_kv_blocks_kernel(
-    seg_addrs_ptr,
-    block_ids_ptr,
-    n_blocks,
-    N_SEGS: tl.constexpr,
-    PAGE_SIZE_EL: tl.constexpr,
-    BLOCK_SIZE: tl.constexpr,
-):
-    """Zero KV cache blocks across all segments in a single launch.
-
-    Each segment is a contiguous region of one block's data.  For backends
-    where blocks are outermost (block_dim=0) there is one segment per
-    buffer.  For backends where K/V is outermost (block_dim=1) there are
-    two segments per buffer (one for K, one for V).
-
-    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
-    allowing segments to live in different CUDA allocations.
-
-    Programs are mapped as (block_index, seg_index, chunk_index).
-    """
-    pid = tl.program_id(0)
-    chunks = PAGE_SIZE_EL // BLOCK_SIZE
-    work_per_block = N_SEGS * chunks
-    block_index = pid // work_per_block
-    if block_index >= n_blocks:
-        return
-    remainder = pid % work_per_block
-    seg_index = remainder // chunks
-    chunk_index = remainder % chunks
-    block_id = tl.load(block_ids_ptr + block_index)
-    seg_addr = tl.load(seg_addrs_ptr + seg_index)
-    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
-    offset = (
-        block_id.to(tl.int64) * PAGE_SIZE_EL + chunk_index.to(tl.int64) * BLOCK_SIZE
-    )
-    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
-    tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))
-
-
 class GPUModelRunner(
     LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
 ):
@@ -962,28 +922,19 @@ class GPUModelRunner(
             )

     def _init_kv_zero_meta(self) -> None:
-        """One-time precomputation for _zero_block_ids.
-
-        Builds absolute-address table for the Triton zeroing kernel.
-        Each entry is the absolute byte address of a segment start on the
-        GPU, so segments in different CUDA allocations work correctly.
+        """One-time setup for _zero_block_ids.

-        Block IDs from the scheduler reference logical blocks whose size
-        may differ from the kernel block size (virtual block splitting).
-        PAGE_SIZE_EL accounts for this ratio so that
-        ``block_id * PAGE_SIZE_EL`` lands at the correct offset.
+        Collects (kv_tensor, block_dim, ratio) tuples for all unique
+        FullAttentionSpec KV cache tensors so _zero_block_ids can zero
+        them with index_fill_.

         Only AttentionSpec layers are processed; Mamba layers are skipped.
         """
-        from itertools import product as iprod
-
         from vllm.v1.kv_cache_interface import FullAttentionSpec

         kernel_block_sizes = self._kernel_block_sizes

         def find_block_dim(group, kernel_bs, spec) -> int:
-            """Use a sentinel value to discover which tensor dim is the
-            block index, since different backends lay out dims differently."""
             _S = 1234567
             shape = group.backend.get_kv_cache_shape(
                 _S,
@@ -994,9 +945,8 @@ class GPUModelRunner(
             )
             return shape.index(_S)

+        targets: list[tuple[torch.Tensor, int, int]] = []
         seen_ptrs: set[int] = set()
-        seg_addrs: list[int] = []
-        page_size_el: int | None = None

         for group in self._kv_cache_spec_attn_group_iterator():
             spec = group.kv_cache_spec
@@ -1021,84 +971,28 @@ class GPUModelRunner(
                 if dp in seen_ptrs:
                     continue
                 seen_ptrs.add(dp)
+                targets.append((kv, block_dim, ratio))

-                el = kv.element_size()
-                cur_bytes = kv.stride(block_dim) * el
-                assert cur_bytes % 4 == 0
-                kernel_block_el = cur_bytes // 4
-                cur_page_el = kernel_block_el * ratio
-                if page_size_el is None:
-                    page_size_el = cur_page_el
-                else:
-                    assert page_size_el == cur_page_el, (
-                        f"Non-uniform page sizes: {page_size_el} vs {cur_page_el}"
-                    )
-
-                # Enumerate dims with stride larger than a block to find
-                # independent contiguous segments, then record each base addr.
-                block_stride_bytes = cur_bytes
-                outer_dims = [
-                    d
-                    for d in range(block_dim)
-                    if kv.stride(d) * el > block_stride_bytes
-                ]
-                outer_strides = [kv.stride(d) * el for d in outer_dims]
-                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
-                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
-                    seg_addrs.append(dp + off_bytes)
-
-        if not seg_addrs or page_size_el is None:
-            self._kv_zero_meta = None
-            return
-
-        # Pick largest power-of-2 dividing page_size_el as Triton BLOCK_SIZE
-        # (capped at 1024), and pre-allocate pinned + GPU buffers for block IDs.
-        alignment = page_size_el & (-page_size_el)
-        blk_size = min(alignment, 1024)
-        self._kv_zero_id_cap = 8192
-        self._kv_zero_ids_pinned = torch.empty(
-            self._kv_zero_id_cap,
-            dtype=torch.int64,
-            pin_memory=self.pin_memory,
-        )
-        self._kv_zero_ids_gpu = torch.empty(
-            self._kv_zero_id_cap, dtype=torch.int64, device=self.device
-        )
-        self._kv_zero_meta = (
-            torch.tensor(seg_addrs, dtype=torch.int64, device=self.device),
-            page_size_el,
-            blk_size,
-            len(seg_addrs),
-        )
+        self._kv_zero_targets = targets if targets else None

     def _zero_block_ids(self, block_ids: list[int]) -> None:
         """Zero the KV cache memory for the given block IDs."""
-        if not block_ids or self._kv_zero_meta is None:
+        if not block_ids or self._kv_zero_targets is None:
             return
-        seg_addrs, page_size_el, blk_size, n_segs = self._kv_zero_meta
-        n_blocks = len(block_ids)
-        if n_blocks > self._kv_zero_id_cap:
-            self._kv_zero_id_cap = n_blocks * 2
-            self._kv_zero_ids_pinned = torch.empty(
-                self._kv_zero_id_cap,
-                dtype=torch.int64,
-                pin_memory=self.pin_memory,
-            )
-            self._kv_zero_ids_gpu = torch.empty(
-                self._kv_zero_id_cap, dtype=torch.int64, device=self.device
-            )
-        self._kv_zero_ids_pinned[:n_blocks].numpy()[:] = block_ids
-        idx = self._kv_zero_ids_gpu[:n_blocks]
-        idx.copy_(self._kv_zero_ids_pinned[:n_blocks], non_blocking=True)
-        grid = (n_blocks * n_segs * (page_size_el // blk_size),)
-        _zero_kv_blocks_kernel[grid](
-            seg_addrs,
-            idx,
-            n_blocks,
-            N_SEGS=n_segs,
-            PAGE_SIZE_EL=page_size_el,
-            BLOCK_SIZE=blk_size,
-        )
+        for kv, block_dim, ratio in self._kv_zero_targets:
+            if ratio == 1:
+                idx = torch.tensor(
+                    block_ids, dtype=torch.long, device=kv.device
+                )
+            else:
+                expanded = []
+                for bid in block_ids:
+                    for r in range(ratio):
+                        expanded.append(bid * ratio + r)
+                idx = torch.tensor(
+                    expanded, dtype=torch.long, device=kv.device
+                )
+            kv.index_fill_(block_dim, idx, 0)

     # Note: used for model runner override.
     def _init_device_properties(self) -> None:
```

</details>

---
### vadiklyutiy @ 2026-03-10T02:07:54Z

@tdoublep 
Thank you for your detailed review. I have addressed all of your points. Please take a look at the current version.

---