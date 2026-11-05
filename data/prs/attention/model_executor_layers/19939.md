# PR #19939: [PERF] Speedup of MRoPE prepare inputs

- field: attention  | area: model_executor/layers
- author: vadiklyutiy
- state: MERGED  created: 2025-06-21T17:54:51Z  merged: 2025-06-24T06:01:26Z
- url: https://github.com/vllm-project/vllm/pull/19939
- labels: ready, v1, qwen

## Body

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results

## Purpose
Speedup of MRoPE prepare inputs. 
#16881 got stuck for a while. I combined parts of #16881 and #17617 to minimize changes and brings sufficient speedup.
`MRotaryEmbedding.get_next_input_positions_tensor` takes a lot of time. Use numpy to speedup it. 


## Performance Test Result
I used
`vllm serve Qwen/Qwen2.5-VL-3B-Instruct --disable-log-requests --no-enable-prefix-caching` 
as a server and 
`fib benchmark -rps 50 --input-token-distribution uniform 250 300     --output-token-distribution uniform 150 250 --num-of-imgs-per-req 1     --img-ratios-per-req 512x512 -n 1000 --base-url http://localhost:8000     --endpoint v1/chat/completions --backend openai-chat
`
to make workload (send 50 requests per sec with one 512x512 image per request). 

I decorated `GPUModelRunner._prepare_inputs` with nvtx to measure the time. 
`_prepare_inputs` average time
Before: 3.869 ms
With this PR: 1.487 ms (speedup 2.6x)
For info, with #17617: 2.511 ms

E2E performance improvement on Qwen2.5-VL-3B-Instruct with high load is around 1.5%.

cc @imkero @ywang96 @simon-mo 




## Comments

### github-actions @ 2025-06-21T17:54:59Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### ywang96 @ 2025-06-24T06:19:38Z

Sorry for the late comment but this is great!
---