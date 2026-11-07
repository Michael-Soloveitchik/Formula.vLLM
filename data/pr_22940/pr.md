# PR #22940: [Core] direct indexing on self.block_table_np in compute_slot_mapping

- field: scheduling  | area: v1/worker
- author: linzebing
- state: MERGED  created: 2025-08-14T22:00:36Z  merged: 2025-08-15T19:12:13Z
- url: https://github.com/vllm-project/vllm/pull/22940
- labels: ready, v1

## Body

## Purpose
Streamline slot mapping computation by replacing Torch tensor flattening and conversion with direct NumPy indexing via `ravel`, eliminating redundant copies and conversions

## Test Plan
```
# vLLM Serving
VLLM_USE_V1=1 vllm serve facebook/opt-125m \
    --swap-space 16 \
    --disable-log-requests \
    --host :: \
    --dtype float16

# Capture traces
VLLM_USE_V1=1 vllm bench serve \
    --dataset-name random \
    --model facebook/opt-125m \
    --served-model-name facebook/opt-125m \
    --random-input-len 700 \
    --random-output-len 1 \
    --endpoint /v1/completions \
    --ignore-eos \
    --host localhost \
    --port 8000 \
    --num-prompts 100 \
    --profile
```

Also ran throughput benchmark test:
```
CUDA_VISIBLE_DEVICES=7 VLLM_USE_V1=1 python3 benchmarks/benchmark_throughput.py \
    --model facebook/opt-125m \
    --backend vllm \
    --input-len 800 \
    --output-len 75 \
    --num-prompts 30000
```



## Test Result

Reduced `compute_slot_mapping` from 400+μs to 15-30μs.
Throughput improved 2.07% for opt-125m with input=800 and output=75

Before:
Throughput: 268.45 requests/s, 234888.85 total tokens/s, 20133.98 output tokens/s

<img width="1216" height="535" alt="image" src="https://github.com/user-attachments/assets/355a7899-245b-4bef-b7f3-8f34265ab219" />

After:
Throughput: 274.01 requests/s, 239750.19 total tokens/s, 20550.72 output tokens/s
<img width="1450" height="535" alt="image" src="https://github.com/user-attachments/assets/fa919447-5ac8-4ea1-832f-af5523bab602" />

## (Optional) Documentation Update

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>



## Comments

### github-actions @ 2025-08-14T22:00:43Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### linzebing @ 2025-08-14T22:07:07Z

One-liner with significant perf improvement, @heheda12345 , @LucasWilkinson @houseroad and @njhill , can you take a look?
---
### Jialin @ 2025-08-15T15:01:02Z

Nice! TIL
---
### facebook-github-bot @ 2025-08-15T20:16:48Z

@linzebing has imported this pull request. If you are a Meta employee, you can view this in [D80367371](https://www.internalfb.com/diff/D80367371?dst_version_fbid=800159129199890).
---