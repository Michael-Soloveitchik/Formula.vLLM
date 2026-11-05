# PR #34049: [Spec Decode] Reduce TP communication for speculative decoding draft token generation

- field: decoding  | area: vllm/config
- author: zixi-qi
- state: MERGED  created: 2026-02-07T09:17:20Z  merged: 2026-02-22T22:59:16Z
- url: https://github.com/vllm-project/vllm/pull/34049
- labels: speculative-decoding, ready, v1, llama

## Body

<!-- markdownlint-disable -->

## Purpose
Reduce TP communication overhead in Eagle speculative decoding by replacing the all-gather of full vocabulary logits with a local argmax + gather reduction.                                                                                                                                      
                                                                                                                                                   
## Problem                                                   

During draft token generation, Eagle speculative decoding all-gathers the full logits tensor across TP ranks — O(batch_size × vocab_size) data per step. For large vocabularies (e.g., 200K+ tokens in Llama 4), this becomes unnecessary overhead since only the argmax token ID is needed for greedy draft selection.

 ## Solution

  Add a use_local_argmax_reduction option to speculative_config that:

  1. Computes local argmax on each TP rank's vocab shard
  2. Gathers only (max_value, global_index) pairs — O(batch_size × 2 × tp_size) instead of O(batch_size × vocab_size)
  3. Reduces across ranks to find the global argmax


## Test Plan

Benchmarked on Llama4 Maverick + EAGLE and observed on par draft acceptance rate + small improvement on TTIT. Generally this optimization is helpful when batch_size * draft_vocab_size is sufficiently large, when bs is small or draft model has smaller vocabulary the gains would become marginal, so default to False. 

```
  Benchmark Results                                                                                                                                
                                                            
  Setup: Llama 4 Maverick 17B-128E-Instruct-FP8 + EAGLE (morgendave/EAGLE-Llama-4-Maverick-17B-128E-Instruct), TP=8, 1000 ShareGPT requests,       
  num_speculative_tokens=5, cudagraph_mode=FULL_AND_PIECEWISE
  ┌────────────────────────────┬──────────┬──────────────┬─────────┐
  │           Metric           │ Baseline │ Local Argmax │  Delta  │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Output throughput (tok/s)  │ 2647.27  │ 2634.42      │ -0.5%   │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Total throughput (tok/s)   │ 5726.19  │ 5702.24      │ -0.4%   │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Request throughput (req/s) │ 14.29    │ 14.24        │ -0.3%   │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Acceptance rate (%)        │ 37.44    │ 37.07        │ -0.37pp │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Mean acceptance length     │ 2.87     │ 2.85         │ -0.7%   │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Mean TPOT (ms)             │ 25.02    │ 24.70        │ -1.3%   │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Mean ITL (ms)              │ 62.57    │ 61.97        │ -1.0%   │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Mean TTFT (ms)             │ 31009    │ 30910        │ -0.3%   │
  ├────────────────────────────┼──────────┼──────────────┼─────────┤
  │ Mean E2EL (ms)             │ 35051    │ 34934        │ -0.3%   │
  └────────────────────────────┴──────────┴──────────────┴─────────┘
```

## Test Result

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### mergify @ 2026-02-07T09:21:36Z

Hi @zixi-qi, the pre-commit checks have failed. Please run:

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
### zixi-qi @ 2026-02-11T12:27:31Z

cc @houseroad could you take a look when you get a chance? Thx a lot!
---
### mergify @ 2026-02-12T07:20:49Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @zixi-qi.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### zixi-qi @ 2026-02-13T15:02:34Z

Rebase and resolve merge conflict
---
### dosubot @ 2026-02-22T22:59:25Z

<!-- Answer -->
**Related Documentation**

Checked 0 published document(s) in 1 knowledge base(s). No updates required.


<!-- Dosu Comment Footer -->
<sup>How did I do? Any feedback?</sup>&nbsp;&nbsp;[![Join Discord](https://img.shields.io/badge/join-5865F2?logo=discord&logoColor=white&label=)](https://go.dosu.dev/discord-bot)
---
### ZJY0516 @ 2026-03-15T15:57:46Z

> Generally this optimization is helpful when batch_size * draft_vocab_size is sufficiently large,

May I ask which senario people will use large batchsize in spec decoding?
---
### zixi-qi @ 2026-03-15T16:25:44Z

> > Generally this optimization is helpful when batch_size * draft_vocab_size is sufficiently large,
> 
> May I ask which senario people will use large batchsize in spec decoding?

Generally speaking when batch size is large, speculative decoding is less helpful, but the exact break even point varies based on the base model, hardware, draft model accuracy etc. So there are certain scenarios where this optimization comes in handy. 
---