# PR #13625: [Kernel] Optimize moe intermediate_cache usage

- field: gpu_kernels  | area: model_executor/layers
- author: mgoin
- state: MERGED  created: 2025-02-20T18:34:43Z  merged: 2025-03-03T21:29:53Z
- url: https://github.com/vllm-project/vllm/pull/13625
- labels: ready

## Body

I'm running into OOM issues at long sequence lengths with deepseek r1, so exploring options here (see [o3-mini chat](https://chatgpt.com/share/67b7970b-6858-8009-ab03-288cd37fffdf)). First I tried moving the silu+mul to be an inplace operation via a new kernel `torch.ops._C.silu_and_mul_inplace(intermediate_cache1.view(-1, N))`, but it seems easier to reuse memory for cache1 and cache3 since there is absolutely no data dependency there

Manual peak measurement shows 15% reduction in memory for fused_moe for 64k prefill

Eval:
```
vllm (pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7642|±  |0.0117|
|     |       |strict-match    |     5|exact_match|↑  |0.7468|±  |0.0120|
```

## Comments

### github-actions @ 2025-02-20T18:34:58Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### mgoin @ 2025-02-25T17:13:08Z

Manual peak measurement shows 15% reduction in memory for fused_moe for 64k prefill

main:
```
python moe_mem.py
Memory usage: 3586 MB
Time: 5.0750 seconds
```

This PR:
```
python moe_mem.py
Memory usage: 3074 MB
Time: 5.0809 seconds
```


```python
import torch
import time
from vllm.model_executor.layers.fused_moe import fused_moe

num_tokens = 64 * 1024
experts = 8
hidden_size = 4096
intermediate_size = 8192
topk = 2
torch.manual_seed(0)
x = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.float16) / 32
w1 = torch.randn((experts, intermediate_size * 2, hidden_size), device="cuda", dtype=torch.float16) / 32
w2 = torch.randn((experts, hidden_size, intermediate_size), device="cuda", dtype=torch.float16) / 32
gating_output = torch.randn((num_tokens, experts), device="cuda", dtype=torch.float16)

# Run once to get peak memory usage
start_memory_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
_ = fused_moe(x, w1, w2, gating_output, topk, True)
end_memory_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
print(f"Memory usage: {end_memory_mb - start_memory_mb} MB")

# Benchmark performance
start = time.perf_counter()
for _ in range(100):
    x = fused_moe(x, w1, w2, gating_output, topk, True)
elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.4f} seconds")
```
---