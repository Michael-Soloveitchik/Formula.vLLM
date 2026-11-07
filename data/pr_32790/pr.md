# PR #32790: [MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H)

- field: gpu_kernels  | area: model_executor/layers
- author: danielafrimi
- state: MERGED  created: 2026-01-21T17:07:39Z  merged: 2026-02-02T14:18:51Z
- url: https://github.com/vllm-project/vllm/pull/32790
- labels: ready

## Body

Enable parallel CUDA stream execution between shared and routed experts for latent MoE architectures (e.g., Nemotron-H).

## Problem

Latent MoE compresses the input before routing to experts:
- **Routed experts** receive compressed input: `[S, moe_latent_size]` (e.g., 1024)
- **Shared experts** need original input: `[S, hidden_size]` (e.g., 4096)

Previously, this dimension mismatch prevented the streaming optimization where shared and routed experts run in parallel on separate CUDA streams.

**Before (Sequential - No Streaming in nemotron_h.py):**
```
Time ────────────────────────────────────────────────────────────────►

GPU:  [fc1_latent_proj] ──► [routed_experts] ──► [shared_experts] ──► [combine]
      [S,4096]->[S,1024]     [S,1024]             [S,4096]
      
      |<──────────────────── Total Time ────────────────────────────>|
```

## Solution

Save the original `hidden_states` before applying the transform, allowing shared experts to use the original while routed experts use the transformed input:

**After (Parallel Streaming):**
```
Time ────────────────────────────────────────────────────────────────►

MAIN     ┌──────────────────────────────────────────┐
STREAM   │ [fc1_latent_proj] ──► [routed_experts]   │──► [combine]
         │ [S,4096]->[S,1024]     [S,1024]          │
         │                                          │
         │            (in parallel)                 │
         │                                          │
AUX      │ [shared_experts]                         │
STREAM   │ [S,4096] -> [S,4096]                     │
         └──────────────────────────────────────────┘
                                                   SYNC
         
         |<──────────── Reduced Time ─────────────>|
```

## Implementation Flow (Implemented it on NemotronH for example)

```
NemotronHMoE.forward(hidden_states=[S, 4096])
    |
    v
SharedFusedMoE(routed_input_transform=fc1_latent_proj)
    |
    v
FusedMoE.forward_native()
    |
    |-- Save: _shared_experts_input = hidden_states     [S, 4096]
    |-- Transform: hidden_states = fc1_latent_proj()    [S, 1024]
    |
    v
FusedMoE.forward_impl()
    |
    |-- Clone _shared_experts_input for AUX stream
    |
    +---------------------------+---------------------------+
    |       MAIN STREAM         |        AUX STREAM         |
    |                           |                           |
    |   router(hidden_states)   |                           |
    |   routed_experts          |   shared_experts          |
    |   [S, 1024]               |   [S, 4096]               |
    +---------------------------+---------------------------+
                                |
                               SYNC
                                |
                                v
    Return (shared_output [S, 4096], routed_output [S, 1024])
```

## Changes

| File | Change |
|------|--------|
| `layer.py` | Add `apply_routed_input_transform()` hook in `FusedMoE` base class |
| `shared_fused_moe.py` | Add `routed_input_transform` parameter, override hook |
| `nemotron_h.py` | Pass `routed_input_transform=fc1_latent_proj` to `SharedFusedMoE` |


## Benchmark Results Summary (H100, TP=4)

  | Test | Batch | ISL | OSL | Streaming ON (tok/s) | Streaming OFF (tok/s) | Difference | Verdict |
  |------|-------|-----|-----|----------------------|------------------------|------------|---------|
  | 1 | 1 | 100 | 2000 | 64.71 | 65.04 | -0.5% | Neutral |
  | 2 | 4 | 100 | 1000 | 258.61 | 250.14 | **+3.4%** | ✅ Helps |
  | 3 | 128 | 100 | 2000 | 3701.34 | 3701.12 | ~0% | Neutral |
  | 4 | 128 | 500 | 4000 | 3446.49 | 3443.06 | **+0.1%** | Neutral |


  ## Serving Benchmark Results (B200, 8K Input / 64K Output)

  | Metric | Streaming ON | Streaming OFF | Difference |
  |--------|--------------|---------------|------------|
  | Output token throughput (tok/s) | 4989.99 | 4986.64 | +0.07% |
  | Mean TTFT (ms) | 27157.10 | 26324.07 | +3.2% |
  | Mean TPOT (ms) | 25.22 | 25.25 | -0.1% |

  **Verdict:** Neutral (~0% throughput difference)

  **Config:** 256 requests, 128 max concurrency, 8K ISL / 64K OSL

## Comments

### mergify @ 2026-01-21T17:12:53Z

Hi @danielafrimi, the pre-commit checks have failed. Please run:

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
### mergify @ 2026-01-22T16:13:29Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @danielafrimi.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-01-25T09:43:13Z

Hi @danielafrimi, the pre-commit checks have failed. Please run:

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
### mergify @ 2026-01-26T14:01:25Z

Hi @danielafrimi, the pre-commit checks have failed. Please run:

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
### danielafrimi @ 2026-01-27T12:27:53Z

@tylertitsworth Fixed :) 
---
### mergify @ 2026-01-29T09:53:08Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @danielafrimi.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### bnellnm @ 2026-02-05T18:41:19Z

@danielafrimi Can you provide a pointer to a model that exercises this feature?  I'm doing some refactoring of the `FusedMoE` class and I want to make sure I do not break this.
---