# PR #6442: [Core] Use numpy to speed up padded token processing

- field: scheduling  | area: model_executor/sampling_metadata.py
- author: peng1999
- state: MERGED  created: 2024-07-15T09:31:47Z  merged: 2024-07-16T15:13:26Z
- url: https://github.com/vllm-project/vllm/pull/6442
- labels: ready

## Body

This small change replaces list comprehension with NumPy operations. It results in approximately a 50% speedup for the sampler when processing large batch sizes. This improvement will enhance throughput when running small models. (See benchmark results below)

Related: #249 #5289

## Benchmarks

The following test was conducted on an NVIDIA A100 80GB GPU.

### Sampler latency

This specifically measures the latency of the sampler component.
<!--
<details>
<summary>This script is used to measure the lantency of sampler</summary>

```python
import argparse
import random
import time
from typing import List
import cProfile

import torch
import torch.nn.functional as F
from torch import nn

from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available

models = {
    'qwen-a14.2b': {
        'name': 'qwen-moe-a14.2b',
        "h": 3584,
        "vocab": 151936,
    },
    'qwen-32b': {
        'name': 'qwen-dense-32b',
        "h": 5120,
        "vocab": 152064,
    },
    'qwen-72b': {
        'name': 'qwen-dense-72b',
        "h": 8192,
        "vocab": 152064,
    },
    'mixtral-8x7b': {
        'name': 'mixtral-8x7b',
        "h": 4096,
        "vocab": 32000,
    },
    'deepseek-moe-16b': {
        'name': 'deepseek-moe-16b',
        "h": 2048,
        "vocab": 102400,
    },
    'deepseek-v2': {
        'name': 'deepseek-v2',
        "h": 5120,
        "vocab": 102400,
    },
    'llama-3-8b': {
        'name': 'llama-3-8b',
        'h': 4096,
        'vocab': 128256,
    },
}


def prepare_seq_group_metadata(
    request_id: str,
    seq_len: int,
    query_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
):
    seq_group_metadata = SequenceGroupMetadata(
        request_id=request_id,
        is_prompt=False,
        seq_data={
            0: SequenceData(
                prompt_token_ids=[1] * (seq_len // 4 + random.randint(0, 2 * seq_len * 3 // 4)),
                output_token_ids=[2] * query_len,
            )
        },
        sampling_params=SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_beam_search=False,
            repetition_penalty=repetition_penalty,
        ),
        block_tables={
            0: [1],
        },
    )
    return seq_group_metadata

def prepare_sampling_metadata(
    num_seqs: int,
    seq_len: int,
    query_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    device: torch.device,
) -> SamplingMetadata:
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    for request_id in range(num_seqs):
        seq_group_metadata_list.append(
            prepare_seq_group_metadata(
                request_id=str(request_id),
                seq_len=seq_len,
                query_len=query_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        )
    seq_lens = [seq_len] * num_seqs
    query_lens = [query_len] * num_seqs
    return SamplingMetadata.prepare(
        seq_group_metadata_list=seq_group_metadata_list,
        seq_lens=seq_lens,
        query_lens=query_lens,
        device=device,
        pin_memory=is_pin_memory_available(),
    )


@torch.inference_mode()
def main(
    version: str,
    model_name: str,
    num_seqs: int,
    seq_len: int,
    query_len: int,
    iters: int,
    use_logits: bool,
    dtype: torch.dtype,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    profile: bool,
    device: str = "cuda",
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    model = models[model_name]
    sampler = Sampler()

    if use_logits:
        logits = torch.randn(num_seqs * query_len, model['vocab'], device=device, dtype=dtype)
    else:
        hidden_states = torch.randn(num_seqs * query_len, model['h'], device=device, dtype=dtype)
        lmhead = nn.Linear(model['h'], model['vocab']).to(device=device, dtype=dtype)
    sampling_metadata = prepare_sampling_metadata(
        num_seqs=num_seqs,
        seq_len=seq_len,
        query_len=query_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        device=device,
    )
    torch.cuda.synchronize()

    def run_cuda_benchmark(num_iters: int, profile=False):
        torch.cuda.synchronize()
        if profile:
            pr = cProfile.Profile()
            pr.enable()
        start_time = time.perf_counter()
        for _ in range(num_iters):
            if not use_logits:
                logits = lmhead(hidden_states)
            sampler(logits, sampling_metadata)
        torch.cuda.synchronize()
        duration = (time.perf_counter() - start_time) / num_iters
        if profile:
            pr.disable()
            pr.dump_stats(f"sampler-{seq_len}-{num_seqs}.prof")
        return duration

    # Warmup.
    print(f"Warming up: version = {version}, model = {model['name']}, batch_size = {num_seqs}, query_len = {query_len} ...")
    run_cuda_benchmark(num_iters=3)

    # Benchmark.
    latency = run_cuda_benchmark(num_iters=iters, profile=profile)
    print(f"Sampler running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["vllm"],
                        default="vllm")
    parser.add_argument("--model", choices=models)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--use-logits", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(args)

    main(
        version=args.version,
        model_name=args.model,
        num_seqs=args.batch_size,
        seq_len=args.seq_len,
        query_len=args.query_len,
        iters=args.iters,
        use_logits=args.use_logits,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        profile=args.profile,
    )
```

</details>
-->

<img width="482" alt="图片" src="https://github.com/user-attachments/assets/b3f494fa-74cc-48e0-bcf0-382da8f72c29">

When prompt token is long, the optimization will have large benefits.

<img width="482" alt="图片" src="https://github.com/user-attachments/assets/14978524-a017-458c-bcc0-7217e48fb5d7">

For short prompts, the optimization yields a modest speedup. However, it still outperforms the original version.

### Throughput

The output is obtained by running the following script:

```py
from vllm import LLM, SamplingParams

model_path = "/mnt/data/models/Qwen2-0.5B-Instruct"
llm = LLM(model=model_path, disable_log_stats=False, max_num_seqs=2048, trust_remote_code=True)
tokenizer = llm.get_tokenizer()

with open("text.txt", "r") as file:
    text = file.read()
length = 1024
text = tokenizer.decode(tokenizer.encode(text)[:length])
prompts = [text] * 2048
sampling_params = SamplingParams(temperature=1.0, top_p=0.5, top_k=20, max_tokens=128)

output = llm.generate(prompts, sampling_params)
```

<details>
<summary>main version</summary>

```
Processed prompts:   0%|                                                                                   | 0/2048 [00:00<?, ?it/s]INFO 07-15 17:06:37 metrics.py:334] Avg prompt throughput: 5243.8 tokens/s, Avg generation throughput: 5.1 tokens/s, Running: 32 reqs, Swapped: 0 reqs, Pending: 2016 reqs, GPU KV cache usage: 0.6%, CPU KV cache usage: 0.0%
INFO 07-15 17:06:42 metrics.py:334] Avg prompt throughput: 192507.0 tokens/s, Avg generation throughput: 188.0 tokens/s, Running: 992 reqs, Swapped: 0 reqs, Pending: 1056 reqs, GPU KV cache usage: 19.1%, CPU KV cache usage: 0.0%
INFO 07-15 17:06:47 metrics.py:334] Avg prompt throughput: 187176.6 tokens/s, Avg generation throughput: 182.8 tokens/s, Running: 1920 reqs, Swapped: 0 reqs, Pending: 128 reqs, GPU KV cache usage: 36.9%, CPU KV cache usage: 0.0%
INFO 07-15 17:06:52 metrics.py:334] Avg prompt throughput: 26000.9 tokens/s, Avg generation throughput: 3681.8 tokens/s, Running: 2048 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 40.0%, CPU KV cache usage: 0.0%
INFO 07-15 17:06:57 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3821.7 tokens/s, Running: 2045 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 40.6%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:02 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3997.1 tokens/s, Running: 2045 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 40.6%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:08 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3947.8 tokens/s, Running: 2043 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 41.2%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:13 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3580.9 tokens/s, Running: 2043 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 41.8%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:19 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3820.1 tokens/s, Running: 2043 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 41.8%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:24 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3637.1 tokens/s, Running: 2043 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 42.4%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:29 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3586.9 tokens/s, Running: 2042 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 42.4%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:34 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3468.2 tokens/s, Running: 2041 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 43.0%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:39 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3477.8 tokens/s, Running: 2041 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 43.0%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:45 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3415.1 tokens/s, Running: 2041 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 43.6%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:50 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3548.2 tokens/s, Running: 2041 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 44.2%, CPU KV cache usage: 0.0%
INFO 07-15 17:07:55 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3437.2 tokens/s, Running: 2041 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 44.2%, CPU KV cache usage: 0.0%
Processed prompts: 100%|████████████████████████████████████████████████████████████████████████| 2048/2048 [01:22<00:00, 24.82it/s]
```

</details>

<details>
<summary>optimized version</summary>

```
INFO 07-15 17:20:55 metrics.py:295] Avg prompt throughput: 31913.1 tokens/s, Avg generation throughput: 31.2 tokens/s, Running: 160 reqs, Swapped: 0 reqs, Pending: 1888 reqs, GPU KV cache usage: 3.1%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:00 metrics.py:295] Avg prompt throughput: 190974.6 tokens/s, Avg generation throughput: 186.5 tokens/s, Running: 1120 reqs, Swapped: 0 reqs, Pending: 928 reqs, GPU KV cache usage: 21.5%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:05 metrics.py:295] Avg prompt throughput: 187246.9 tokens/s, Avg generation throughput: 182.9 tokens/s, Running: 2048 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 39.3%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:10 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 6416.5 tokens/s, Running: 2046 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 39.9%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:16 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 6046.0 tokens/s, Running: 2045 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 40.5%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:21 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 6079.2 tokens/s, Running: 2045 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 41.1%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:26 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5687.9 tokens/s, Running: 2045 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 41.7%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:31 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5591.1 tokens/s, Running: 2045 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 42.3%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:36 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4830.2 tokens/s, Running: 2044 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 42.9%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:42 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5256.2 tokens/s, Running: 2043 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 43.5%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:47 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5286.0 tokens/s, Running: 2042 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 44.1%, CPU KV cache usage: 0.0%.
INFO 07-15 17:21:52 metrics.py:295] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5070.9 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
Processed prompts: 100%|█| 2048/2048 [00:57<00:00, 35.36it/s, est. speed input: 36204.36 toks/s, output: 4517.
```

</details>

**BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE**

---

<details>
<!-- inside this <details> section, markdown rendering does not work, so we use raw html here. -->
<summary><b> PR Checklist (Click to Expand) </b></summary>

<p>Thank you for your contribution to vLLM! Before submitting the pull request, please ensure the PR meets the following criteria. This helps vLLM maintain the code quality and improve the efficiency of the review process.</p>

<h3>PR Title and Classification</h3>
<p>Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:</p>
<ul>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[Doc]</code> for documentation fixes and improvements.</li>
    <li><code>[Model]</code> for adding a new model or improving an existing model. Model name should appear in the title.</li>
    <li><code>[Frontend]</code> For changes on the vLLM frontend (e.g., OpenAI API server, <code>LLM</code> class, etc.) </li>
    <li><code>[Kernel]</code> for changes affecting CUDA kernels or other compute kernels.</li>
    <li><code>[Core]</code> for changes in the core vLLM logic (e.g., <code>LLMEngine</code>, <code>AsyncLLMEngine</code>, <code>Scheduler</code>, etc.)</li>
    <li><code>[Hardware][Vendor]</code> for hardware-specific changes. Vendor name should appear in the prefix (e.g., <code>[Hardware][AMD]</code>).</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If the PR spans more than one category, please include all relevant prefixes.</p>

<h3>Code Quality</h3>

<p>The PR need to meet the following code quality standards:</p>

<ul>
    <li>We adhere to <a href="https://google.github.io/styleguide/pyguide.html">Google Python style guide</a> and <a href="https://google.github.io/styleguide/cppguide.html">Google C++ style guide</a>.</li>
    <li>Pass all linter checks. Please use <a href="https://github.com/vllm-project/vllm/blob/main/format.sh"><code>format.sh</code></a> to format your code.</li>
    <li>The code need to be well-documented to ensure future contributors can easily understand the code.</li>
    <li>Include sufficient tests to ensure the project to stay correct and robust. This includes both unit tests and integration tests.</li>
    <li>Please add documentation to <code>docs/source/</code> if the PR modifies the user-facing behaviors of vLLM. It helps vLLM user understand and utilize the new features or changes.</li>
</ul>

<h3>Notes for Large Changes</h3>
<p>Please keep the changes as concise as possible. For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue (RFC) discussing the technical design and justification. Otherwise, we will tag it with <code>rfc-required</code> and might not go through the PR.</p>

<h3>What to Expect for the Reviews</h3>

<p>The goal of the vLLM team is to be a <i>transparent reviewing machine</i>. We would like to make the review process transparent and efficient and make sure no contributor feel confused or frustrated. However, the vLLM team is small, so we need to prioritize some PRs over others. Here is what you can expect from the review process: </p>

<ul>
    <li> After the PR is submitted, the PR will be assigned to a reviewer. Every reviewer will pick up the PRs based on their expertise and availability.</li>
    <li> After the PR is assigned, the reviewer will provide status update every 2-3 days. If the PR is not reviewed within 7 days, please feel free to ping the reviewer or the vLLM team.</li>
    <li> After the review, the reviewer will put an <code> action-required</code> label on the PR if there are changes required. The contributor should address the comments and ping the reviewer to re-review the PR.</li>
    <li> Please respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.
 </li>
</ul>

<h3>Thank You</h3>

<p> Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM. Your contributions make vLLM a great tool for everyone! </p>


</details>




## Comments

### github-actions @ 2024-07-15T09:32:01Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only trigger `fastcheck` CI to run, which consists only a small and essential subset of tests to quickly catch errors with the flexibility to run extra individual tests on top (you can do this by unblocking test steps in the Buildkite run). 

Full CI run is still required to merge this PR so once the PR is ready to go, please make sure to run it. If you need all test signals in between PR commits, you can trigger full CI as well.

 To run full CI, you can do one of these:
- Comment `/ready` on the PR
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### DarkLight1337 @ 2024-07-15T10:31:05Z

This seems to share quite a bit in common with `vllm.utils.make_tensor_with_pad` (recently optimized by #4196). I wonder why the utils function wasn't being used in the first place?
---