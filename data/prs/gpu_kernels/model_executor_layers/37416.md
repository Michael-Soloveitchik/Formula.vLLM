# PR #37416: [Kernel] Mamba support different layout for Conv state

- field: gpu_kernels  | area: model_executor/layers
- author: NickLucche
- state: MERGED  created: 2026-03-18T10:55:20Z  merged: 2026-04-02T23:50:10Z
- url: https://github.com/vllm-project/vllm/pull/37416
- labels: ready, qwen

## Body

This PR proposes swapping the 2D layout of the Conv state from `(state_len, dim)` to `(dim, state_len)`, similarly to the work done for NHD->HND layout for the same purposes of efficient indexing in PD.
This new layout leads to overall better performance (TTFT particularly) as well as enabling HeterogeneousTP support for PD disagg deployments.

## The problem

Mamba layout is currently inefficient for heterogeneous TP in disaggregated scenarios:
 - Temporal SSM and Conv state are sharded along different axis. Nothing lines up with a single KV-style slice, so paths stay Mamba-specific
 - More importantly, Conv state is sharding on an inner dim: this makes indexing into the state for the purpose of *selecting one or more TP shards* complicated and inefficient, as it requires permuting on a staging area (in VRAM buffer).
 - Conv in particular alternates inner dims at loading time to shard on "groups" dimension (https://github.com/vllm-project/vllm/blob/f262a62aa191b4fc3070a5fd7f418246cb326f06/vllm/model_executor/layers/mamba/mamba_utils.py#L140), but then re-groups to maintain a packed representation of x/B/C. 

By having `dim` on the first axis we can slice x/B/C in 3 separate reads for Connectors such as Nixl to transfer mamba states with no additional staging/buffer area penalty.

The NixlConnector changes will follow in a [separate PR](https://github.com/vllm-project/vllm/pull/37635), the scope of this one is ensuring the layout swap maintains accuracy and perf in colocated setup.

## Approach

We define a new env variable (similar to KV layout) `VLLM_SSM_CONV_STATE_LAYOUT` with values `['SD (current default)', 'DS']`.

A side benefit of "DS" is that conv_state doesn't need to be transposed prior to `conv_ssm_forward` here https://github.com/vllm-project/vllm/blob/a889b7f58426e612b55aa67d7d243dbbceaf1184/vllm/model_executor/layers/mamba/mamba_mixer2.py#L580 though a stride trick is currently used to ensure the physical layout is actually already contiguous. 
This is not needed with DS layout.

## Test with

```
pytest -v -s tests/models/language/generation/test_hybrid.py
```

# Benchmarks

~~First benchmarks (sharegpt, H100) are showing a huge throughput increase with the newly added DS layout.
I'll come back to this to double check numbers shortly (looks fishy).~~
```
================================================================================
BENCHMARK SUMMARY
================================================================================
model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
port: 55483
tp_values: [1, 2, 4]
base_serve_args: --trust-remote-code --no-enable-prefix-caching

DS:
------------------------------------------------------------
  TP |    Req/s |      Tok/s |   TTFT(ms) |   TPOT(ms)
------------------------------------------------------------
   1 |    20.98 |    4062.75 |     124.69 |      23.17
   2 |    22.08 |    4258.65 |     370.55 |      22.25
   4 |    24.63 |    4787.19 |     367.72 |      19.82

SD:
------------------------------------------------------------
  TP |    Req/s |      Tok/s |   TTFT(ms) |   TPOT(ms)
------------------------------------------------------------
   1 |    20.62 |    3944.23 |     167.36 |      24.12
   2 |    11.29 |    2187.66 |    4179.76 |      47.00
   4 |    11.87 |    2309.94 |    4187.93 |      43.23
```

I re-ran benchmarks, but removing this patch here https://github.com/vllm-project/vllm/pull/38270 which is causing a severe jitter in performance (observed in previous results above). Will address this separately.
DS layout is still exhibiting significantly better TTFT (up to ~1.5x).

```
================================================================================
BENCHMARK SUMMARY
================================================================================

DS:
------------------------------------------------------------
  TP |    Req/s |      Tok/s |   TTFT(ms) |   TPOT(ms)
------------------------------------------------------------
   1 |    20.61 |    4009.30 |     128.67 |      23.50
   2 |    24.15 |    4686.47 |     119.85 |      21.18
   4 |    28.48 |    5556.18 |     112.23 |      18.23

SD:
------------------------------------------------------------
  TP |    Req/s |      Tok/s |   TTFT(ms) |   TPOT(ms)
------------------------------------------------------------
   1 |    20.34 |    3954.31 |     142.86 |      23.78
   2 |    23.86 |    4681.23 |     161.49 |      20.92
   4 |    26.63 |    5182.43 |     167.71 |      19.39

================================================================================
```

cc @tdoublep  @ZhanqiuHu 


## Comments

### mergify @ 2026-03-18T21:53:30Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @NickLucche.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-03-31T15:37:49Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @NickLucche.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-04-01T11:07:24Z

Hi @NickLucche, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit>=4.5.1
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
---
### tdoublep @ 2026-04-01T16:54:17Z

It would also be good to check some eval scores before/after the change, just to be safe. 
---
### ZhanqiuHu @ 2026-04-01T20:43:43Z

> It would also be good to check some eval scores before/after the change, just to be safe.

Ran gsm8k 5-shot accuracy comparison (Nemotron 30B, TP=1, colocated, temp=0.0,
2 server restarts × 3 eval repeats per layout):

| Layout | Scores | Mean | Std |
|--------|--------|------|-----|
| SD | 0.8461, 0.8537, 0.8484, 0.8499, 0.8506, 0.8590 | 0.8513 | ±0.0044 |
| DS | 0.8453, 0.8491, 0.8385, 0.8484, 0.8476, 0.8484 | 0.8462 | ±0.0040 |
---
### tdoublep @ 2026-04-02T13:23:45Z

Let's merge this with the env variable to unblock HeterogeneousTP and then follow it up by changing the default layout (if we can figure out how to solve the align mode issue). Would be good to open an issue on that to track. 
---
### NickLucche @ 2026-04-03T09:02:38Z

@tdoublep tracking here https://github.com/vllm-project/vllm/issues/38898
---