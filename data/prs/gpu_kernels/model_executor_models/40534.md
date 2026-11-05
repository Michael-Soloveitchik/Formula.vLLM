# PR #40534: [Model] Gemma4: add bidirectional vision attention for sliding layers with window guard

- field: attention  | area: model_executor/models
- author: lucianommartins
- state: MERGED  created: 2026-04-21T17:13:16Z  merged: 2026-04-24T08:27:47Z
- url: https://github.com/vllm-project/vllm/pull/40534
- labels: ready, v1

## Body

## Purpose

Implement `use_bidirectional_attention="vision"` support for Gemma4 models
(gemma-4-31B-it, gemma-4-26B-A4B-it), addressing #40106.

Gemma4's architecture applies bidirectional attention to vision tokens in
sliding_attention layers only, allowing image patches to attend to each
other regardless of causal ordering. This matches the HF transformers
reference (`create_causal_mask_mapping` applies an OR mask to
`create_sliding_window_causal_mask` for sliding layers, while full_attention
layers use plain causal).

### Changes

- Register `gemma4` in `MM_PREFIX_LM_MODELS` to enable `mm_prefix_range`
  population for vision token ranges during prefill
- Add `_clear_mm_prefix_for_full_attn_layers` in
  `Gemma4ForConditionalGeneration.forward()` to clear `mm_prefix_range` on
  full_attention layers. This runs outside the `@support_torch_compile`
  boundary since `_run_decoder_layers` is inside a compiled graph where
  Python side effects are eliminated
- Add per-range sliding window guard in `gpu_model_runner`: skip bidi for
  image ranges where `num_tokens > sliding_window`

### Why the sliding window guard

Gemma4 uses `sliding_window=1024`. At the default `max_soft_tokens=280`,
images produce ~256 tokens (within the window) and bidi works correctly.
At `max_soft_tokens=1120`, images produce ~1092 tokens (exceeding the
window). 

The guard filters per-range, so in multi-image or video prompts:
- Small images (≤1024 tokens): bidi enabled
- Large images (>1024 tokens): bidi skipped
- Video frames (~64 tokens each): always enabled

This mirrors the spirit of the `use_bidirectional_attention="all"` config
which halves `sliding_window` to compensate for bidirectional span doubling.

## Test Result

**MMMU-Pro** (gemma-4-31B-it, 1730 samples):

| max_soft_tokens | Image tokens | Within window? | No bidi | With bidi | Delta |
|----------------|-------------|---------------|---------|-----------|-------|
| 280 | 256 | Yes | 72.14% | 72.95% | **+0.81%** |
| 560 | 529 | Yes | 73.29% | 74.39% | **+1.10%** |
| 1120 | 1092 | No (guard skips) | 74.28% | 74.28% | 0% (guarded) |

**ScreenSpot-Pro** (gemma-4-31B-it, 1580 samples, max_soft_tokens=1120):

| Config | Click Accuracy |
|--------|---------------|
| No bidi | 37.26% |
| With bidi + guard | 37.26% (guarded, no regression) |

Bidi improves MMMU-Pro by +0.81–1.10% when images fit within the sliding
window. The guard prevents the 3–4% ScreenSpot-Pro regression observed
without it (caused by attention explosion at max_soft_tokens=1120).

## Comments

### lucianommartins @ 2026-04-21T17:14:59Z

hey @ywang96 @Isotr0py - This PR implements bidi vision for larger Gemma4 models and helps with https://github.com/vllm-project/vllm/issues/40106

the challenge here is the combo slide_window + bidi for different soft_tokens size. I created this approach adding guardrails to avoid scenarios where the bidi will generate more tokens than the slide_window can take care of.

Thoughts?
---
### mergify @ 2026-04-24T04:35:42Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @lucianommartins.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-04-24T06:19:53Z

Hi @lucianommartins, the pre-commit checks have failed. Please run:

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