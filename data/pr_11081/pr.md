# PR #11081: [torch.compile] allow tracking forward time

- field: gpu_kernels  | area: vllm/forward_context.py
- author: youkaichao
- state: MERGED  created: 2024-12-11T04:02:13Z  merged: 2024-12-15T03:45:00Z
- url: https://github.com/vllm-project/vllm/pull/11081
- labels: ready

## Body

when benchmarking `torch.compile` performance, I always get this question: if the performance gain is not satisfactory, is it because `torch.compile` does not optimize the model well, or is it because of the scheduling overhead?

this pr adds the tracking for forward time, so that we can directly test the perf of `torch.compile` for certain sizes.

e.g.

```bash
$ VLLM_LOG_BATCHSIZE_INTERVAL=1.0 python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 --load-format dummy
INFO 12-10 19:52:26 forward_context.py:88] Batchsize forward time stats (batchsize, count, median_time(ms)): [(1, 5054, 6.5), (32, 41, 7.51)]

$ VLLM_LOG_BATCHSIZE_INTERVAL=1.0 python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 --load-format dummy -O "{'level': 3, 'candidate_compile_sizes': [1]}"
INFO 12-10 19:54:34 forward_context.py:88] Batchsize forward time stats (batchsize, count, median_time(ms)): [(1, 5049, 5.93), (32, 41, 7.35)]
```

then it is clear that the forward time improves from 6.5ms to 5.93ms, 8.8% improvement. And the end-to-end 7.7% improvement in latency in https://github.com/vllm-project/vllm/pull/11078 is shadowed a little bit in the end-to-end pipeline.

## Comments

### github-actions @ 2024-12-11T04:02:27Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### youkaichao @ 2024-12-11T04:05:22Z

> Batchsize forward time stats (batchsize, count, median_time(ms)): [(1, 5054, 6.5), (32, 41, 7.51)]

the `32` should be the prefill length. so it is interesting to see that `torch.compile` also accelerates prefill in this case.
---
### youkaichao @ 2024-12-11T15:40:09Z

with this pr and the test code:

```python
from vllm import LLM, SamplingParams
import math

# Create an LLM, and compile for all the batch sizes we care about.
candidate_compile_sizes = [1, 2, 4] + [i * 8 for i in range(1, 33)]

# use compile
llm = LLM(model="meta-llama/Meta-Llama-3-8B", compilation_config={"level": 3, "candidate_compile_sizes": candidate_compile_sizes})

# no compile
# llm = LLM(model="meta-llama/Meta-Llama-3-8B")


for bs in candidate_compile_sizes:
    # in the beginning, we have bs number of sequences, in total about 512 tokens for the prefill
    prompt_token_ids = [[0] * math.floor(512 / bs)] * bs

    # all sequence generates 30 tokens
    sampling_params = SamplingParams(temperature=0, max_tokens=30, ignore_eos=True)

    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

```

I get:

`VLLM_LOG_BATCHSIZE_INTERVAL=1.0 python test.py`

no compile:

INFO 12-10 23:13:55 forward_context.py:89] Batchsize forward time stats (batchsize, count, median_time(ms)): [(248, 31, 9.77), (240, 31, 9.73), (232, 31, 9.67), (224, 31, 9.61), (216, 31, 9.6), (208, 31, 9.67), (200, 31, 9.53), (192, 31, 8.63), (184, 31, 8.6), (176, 31, 8.57), (168, 31, 8.52), (160, 31, 8.51), (152, 31, 8.39), (144, 31, 8.41), (136, 31, 8.41), (128, 31, 7.65), (120, 31, 7.62), (112, 31, 7.59), (104, 31, 7.49), (96, 31, 7.5), (88, 31, 7.46), (80, 31, 7.44), (72, 31, 7.31), (64, 31, 6.66), (56, 31, 6.65), (48, 31, 6.62), (40, 31, 6.58), (32, 31, 6.46), (24, 31, 6.69), (16, 31, 6.45), (8, 31, 6.5), (4, 31, 6.43), (2, 31, 6.52), (1, 31, 6.54), (256, 16, 9.81), (512, 9, 13.54), (480, 7, 15.32), (504, 4, 14.28), (416, 2, 16.86), (448, 2, 17.15), (432, 2, 17.44)]

fully compile:

INFO 12-10 23:11:24 forward_context.py:89] Batchsize forward time stats (batchsize, count, median_time(ms)): [(248, 31, 9.24), (240, 31, 9.19), (232, 31, 9.09), (224, 31, 9.02), (216, 31, 8.86), (208, 31, 8.91), (200, 31, 8.72), (192, 31, 8.43), (184, 31, 8.42), (176, 31, 8.38), (168, 31, 8.29), (160, 31, 8.27), (152, 31, 8.15), (144, 31, 8.1), (136, 31, 8.04), (128, 31, 7.06), (120, 31, 7.05), (112, 31, 7.01), (104, 31, 6.94), (96, 31, 6.76), (88, 31, 6.74), (80, 31, 6.69), (72, 31, 6.65), (64, 31, 6.3), (56, 31, 6.28), (48, 31, 6.24), (40, 31, 6.22), (32, 31, 6.01), (24, 31, 6.21), (16, 31, 6.04), (8, 31, 5.99), (4, 31, 5.97), (2, 31, 6.17), (1, 31, 5.95), (256, 30, 9.2), (512, 9, 13.51), (480, 7, 15.07), (504, 4, 13.96), (416, 2, 16.51), (448, 2, 16.78), (432, 2, 17.08)]

it seems compiled code always runs faster than non-compiled code.
---