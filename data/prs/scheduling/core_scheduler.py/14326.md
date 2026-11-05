# PR #14326: [Bugfix][Core] fix abort_seq_group and memory leak when n>1

- field: scheduling  | area: core/scheduler.py
- author: courage17340
- state: MERGED  created: 2025-03-06T04:11:04Z  merged: 2025-03-06T15:59:33Z
- url: https://github.com/vllm-project/vllm/pull/14326
- labels: ready

## Body

This PR is to fix two bugs when there are requests with n>1:
* Requests with n>1 can't be aborted correctly. The log says they are aborted, but they are still running in fact.
* Elements in self.seq_id_to_seq_group are never deleted.

It's easy to reproduce the first bug: serve a model, make a curl request with n>1 and stream=True, and then press CTRL-C quickly. For easier reproduction, I recommend set larger max_tokens and ignore_eos=True.

Before this fix, you can see they are still running after the `Aborted request` log:
```bash
INFO 03-06 03:31:28 [logger.py:39] Received request chatcmpl-009bfdf842c84ea3972317c764b3958b: prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nwho are you<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=2, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.3, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=4096, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO:     127.0.0.1:56504 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 03-06 03:31:28 [engine.py:289] Added request chatcmpl-009bfdf842c84ea3972317c764b3958b.
INFO 03-06 03:31:28 [metrics.py:481] Avg prompt throughput: 7.1 tokens/s, Avg generation throughput: 0.2 tokens/s, Running: 2 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 03-06 03:31:29 [engine.py:307] Aborted request chatcmpl-009bfdf842c84ea3972317c764b3958b.
INFO 03-06 03:31:33 [metrics.py:481] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 263.3 tokens/s, Running: 2 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.
INFO 03-06 03:31:38 [metrics.py:481] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 261.7 tokens/s, Running: 2 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.3%, CPU KV cache usage: 0.0%.
INFO 03-06 03:31:43 [metrics.py:481] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 260.4 tokens/s, Running: 2 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.4%, CPU KV cache usage: 0.0%.
INFO 03-06 03:31:48 [metrics.py:481] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 259.4 tokens/s, Running: 2 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.5%, CPU KV cache usage: 0.0%.
```

After this fix, everything is OK:
```bash
INFO 03-06 03:55:59 [logger.py:39] Received request chatcmpl-771cfa491889488ca4011e1e6d372b73: prompt: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nwho are you<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=2, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.3, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=4096, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO:     127.0.0.1:47140 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 03-06 03:55:59 [engine.py:289] Added request chatcmpl-771cfa491889488ca4011e1e6d372b73.
INFO 03-06 03:55:59 [metrics.py:481] Avg prompt throughput: 6.7 tokens/s, Avg generation throughput: 0.2 tokens/s, Running: 2 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 03-06 03:56:00 [engine.py:307] Aborted request chatcmpl-771cfa491889488ca4011e1e6d372b73.
INFO 03-06 03:56:10 [metrics.py:481] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 9.4 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
```


## Comments

### github-actions @ 2025-03-06T04:11:16Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### youkaichao @ 2025-03-06T15:59:22Z

failure looks unrelated, merging.
---