# PR #4118: [Kernel][FP8] Initial support with dynamic per-tensor scaling

- field: gpu_kernels  | area: model_executor/layers
- author: comaniac
- state: MERGED  created: 2024-04-16T19:52:41Z  merged: 2024-04-20T04:28:57Z
- url: https://github.com/vllm-project/vllm/pull/4118
- labels: 

## Body

Provide an initial support to FP8 computation. This PR is inspired by HuggingFace TGI: https://github.com/huggingface/text-generation-inference/pull/1726

This feature can be enabled with `--quantization fp8` or `-q fp8` when launching an engine.

## Algorithm
We still load a model checkpoint in FP16/BF16. After the weights are loaded, Fp8LinearMethod calculates the per-tensor scaling factor of weights and quantizes the weights accordingly. The scaling factor will then be stored for future use. Meanwhile, the per-tensor scaling factor for activations is calculated in every forward pass.

## Initial Results

Currently tested Mistral-7B on 1xH100. With prompt length ~5 and decoding length 128:
* BF16: 1.47s
* FP8: 1.66s

I'll try to use larger models and try to find more performance bottleneck. Meanwhile, you're welcome to try this PR.

## To Do
- [ ] Unit tests
- [ ] Comprehensive benchmarking
- [ ] Refine the interface

cc @WoosukKwon @zhuohan123 @robertgshaw2-neuralmagic @mgoin 

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

### mgoin @ 2024-04-16T20:14:42Z

Nice job here for a clean implementation. For performance improvement, I think the easiest/clearest measurement would be to measure TTFT for a large prompt since we are trying to improve compute efficiency here, rather than just memory bandwidth like most existing weight-only quantization methods in vllm.
---
### comaniac @ 2024-04-16T23:47:54Z

I did a benchmarking with 50 requests sending with QPS 1. The average prompt length is 512 and decoding length is 128. The results are as follows:

Case | TTFT (ms) | ITL (ms)
--|--|--
FP16 | 28.8 | 8.5
FP8 | 38.8 | 10.2
FP8-test | 27.4 | 9.5

In `FP8-test`, I removed activation quantization to understand its impact. Although this results in garbage output, it shows that activation quantization does introduce a large amount of overhead. Meanwhile, even removing activation quantization, the speedup isn't obvious. This looks weird to me because in this case we only have `.to` and `torch._scaled_mm` in the forward. I'm going to perform a microbenchmark with `torch._scaled_mm` to see what speedup we could actually achieve against `F.linear`.

---
### comaniac @ 2024-04-17T18:10:40Z

**Microbenchmark** between `torch._scaled_mm` and `F.linear` with shapes extracted from Mistral-7B on H100, PyTorch 2.2.1+cu121. Each run includes 100 trials with `torch.cuda.synchronize()`. The results are the average latency.

* FP16: `F.linear`
* FP8: `torch._scaled_mm` with pre-quantized inputs.
* FP8 w. x-to: `torch._scaled_mm` with pre-quantized weights and run time `x.to(torch.float8_e4m3fn)`. Note that this results in incorrect output values.
* FP8 w. x-quantized: `torch._scaled_mm` with pre-quantized weights and run time `quantize(x)`. The `quantize` is compiled with `torch.compile`.

Summary:
1. Shape matters. Larger the better. Without any other overheads such as quantization, we can see relatively obvious speedup with input tokens >=32.
2. Even just `x.to(torch.float8_e4m3fn)` could moderate most speedups. In means we may not get decent speedups if only linear layers are computed in FP8. The activation quantization overhead can be moderated as long as most ops are computed in FP8 and achieve some speedups on average.

M (tokens) | N | K | FP16 (ms) | FP8 (ms) | Speedup | FP8 w. x-to (ms) | Speedup | FP8 w. x-quantized (ms) | Speedup
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
1 | 4096 | 6144 | 0.3205 | 0.7489 | 0.43 | 1.1397 | 0.28 | 0.7889 | 0.41
1 | 4096 | 4096 | 0.0303 | 0.0435 | 0.7 | 0.0491 | 0.62 | 0.1142 | 0.27
1 | 4096 | 28672 | 3.2293 | 0.1094 | 29.52 | 0.1055 | 30.61 | 0.1738 | 18.58
1 | 14336 | 4096 | 0.0667 | 0.0799 | 0.83 | 0.0733 | 0.91 | 0.1443 | 0.46
2 | 4096 | 6144 | 0.2818 | 0.0464 | 6.07 | 0.0514 | 5.48 | 0.1521 | 1.85
2 | 4096 | 4096 | 0.0298 | 0.0412 | 0.72 | 0.0474 | 0.63 | 0.1446 | 0.21
2 | 4096 | 28672 | 0.1003 | 0.1118 | 0.9 | 0.1075 | 0.93 | 0.2046 | 0.49
2 | 14336 | 4096 | 0.0639 | 0.0721 | 0.89 | 0.071 | 0.90 | 0.1709 | 0.37
4 | 4096 | 6144 | 0.04 | 0.0456 | 0.88 | 0.052 | 0.77 | 0.1507 | 0.27
4 | 4096 | 4096 | 0.0294 | 0.0401 | 0.73 | 0.0468 | 0.63 | 0.1443 | 0.20
4 | 4096 | 28672 | 0.1013 | 0.1092 | 0.93 | 0.1116 | 0.91 | 0.2077 | 0.49
4 | 14336 | 4096 | 0.0637 | 0.0689 | 0.92 | 0.0718 | 0.89 | 0.1695 | 0.38
8 | 4096 | 6144 | 0.04 | 0.0471 | 0.85 | 0.0505 | 0.79 | 0.147 | 0.27
8 | 4096 | 4096 | 0.0295 | 0.0403 | 0.73 | 0.0452 | 0.65 | 0.1422 | 0.21
8 | 4096 | 28672 | 0.1003 | 0.111 | 0.9 | 0.1137 | 0.88 | 0.2121 | 0.47
8 | 14336 | 4096 | 0.0652 | 0.0879 | 0.74 | 0.0721 | 0.90 | 0.1692 | 0.39
16 | 4096 | 6144 | 0.0406 | 0.0455 | 0.89 | 0.0502 | 0.81 | 0.1473 | 0.28
16 | 4096 | 4096 | 0.03 | 0.0406 | 0.74 | 0.0469 | 0.64 | 0.1432 | 0.21
16 | 4096 | 28672 | 0.1017 | 0.109 | 0.93 | 0.1078 | 0.94 | 0.2034 | 0.50
16 | 14336 | 4096 | 0.0689 | 0.0718 | 0.96 | 0.0703 | 0.98 | 0.1688 | 0.41
32 | 4096 | 6144 | 0.0436 | 0.0421 | 1.04 | 0.0472 | 0.92 | 0.1432 | 0.30
32 | 4096 | 4096 | 0.0282 | 0.0344 | 0.82 | 0.0404 | 0.70 | 0.1363 | 0.21
32 | 4096 | 28672 | 0.1059 | 0.0786 | 1.35 | 0.0799 | 1.33 | 0.1746 | 0.61
32 | 14336 | 4096 | 0.066 | 0.054 | 1.22 | 0.058 | 1.14 | 0.1531 | 0.43
64 | 4096 | 6144 | 0.0436 | 0.0377 | 1.16 | 0.0424 | 1.03 | 0.149 | 0.29
64 | 4096 | 4096 | 0.0342 | 0.0345 | 0.99 | 0.0423 | 0.81 | 0.1378 | 0.25
64 | 4096 | 28672 | 0.1064 | 0.0791 | 1.35 | 0.0808 | 1.32 | 0.1776 | 0.60
64 | 14336 | 4096 | 0.065 | 0.054 | 1.21 | 0.0608 | 1.07 | 0.1578 | 0.41
128 | 4096 | 6144 | 0.0509 | 0.0423 | 1.2 | 0.0511 | 1.00 | 0.1452 | 0.35
128 | 4096 | 4096 | 0.036 | 0.0388 | 0.93 | 0.0432 | 0.83 | 0.1395 | 0.26
128 | 4096 | 28672 | 0.1237 | 0.0894 | 1.38 | 0.0884 | 1.40 | 0.1874 | 0.66
128 | 14336 | 4096 | 0.0681 | 0.0544 | 1.25 | 0.0585 | 1.16 | 0.1538 | 0.44
256 | 4096 | 6144 | 0.0511 | 0.0428 | 1.19 | 0.0483 | 1.06 | 0.1462 | 0.35
256 | 4096 | 4096 | 0.0416 | 0.0372 | 1.12 | 0.0433 | 0.96 | 0.1416 | 0.29
256 | 4096 | 28672 | 0.1482 | 0.1095 | 1.35 | 0.111 | 1.34 | 0.2085 | 0.71
256 | 14336 | 4096 | 0.0758 | 0.0613 | 1.24 | 0.0651 | 1.16 | 0.1604 | 0.47
512 | 4096 | 6144 | 0.0668 | 0.0513 | 1.3 | 0.0568 | 1.18 | 0.1532 | 0.44
512 | 4096 | 4096 | 0.0493 | 0.0438 | 1.13 | 0.0491 | 1.00 | 0.1463 | 0.34
512 | 4096 | 28672 | 0.1957 | 0.1326 | 1.48 | 0.1361 | 1.44 | 0.2299 | 0.85
512 | 14336 | 4096 | 0.1239 | 0.1013 | 1.22 | 0.1011 | 1.23 | 0.1938 | 0.64
1024 | 4096 | 6144 | 0.102 | 0.0746 | 1.37 | 0.0788 | 1.29 | 0.1748 | 0.58
1024 | 4096 | 4096 | 0.072 | 0.0579 | 1.24 | 0.0641 | 1.12 | 0.1578 | 0.46
1024 | 4096 | 28672 | 0.3453 | 0.2298 | 1.5 | 0.2473 | 1.40 | 0.3565 | 0.97
1024 | 14336 | 4096 | 0.2188 | 0.1573 | 1.39 | 0.1589 | 1.38 | 0.2487 | 0.88
2048 | 4096 | 6144 | 0.1739 | 0.1162 | 1.5 | 0.1246 | 1.40 | 0.2125 | 0.82
2048 | 4096 | 4096 | 0.1239 | 0.0854 | 1.45 | 0.0931 | 1.33 | 0.1872 | 0.66
2048 | 4096 | 28672 | 0.7443 | 0.418 | 1.78 | 0.4744 | 1.57 | 0.6277 | 1.19
2048 | 14336 | 4096 | 0.401 | 0.251 | 1.6 | 0.2566 | 1.56 | 0.3548 | 1.13
---
### mgoin @ 2024-04-17T18:20:54Z

@comaniac I did some lower-level benchmarking just on `torch._scaled_mm` as you mentioned and got some fairly interesting results that I'm still investigating (EDIT: nice timing on the above 🥈 😄 ). I did this on an RTX 4090  so I did have to use the pytorch-nightly build in order to support CUDA capability 8.9. You can find my code here: https://github.com/mgoin/torch-fp8

Spreadsheet of data here for the following graphs: https://docs.google.com/spreadsheets/d/1X-N8h8_O-Z-S7tN5WRkAXXKkhCEniJciUpEzaUaksqs/edit?usp=sharing

Here is a fine-grained benchmark showing various M (num tokens dim, so either batch size or prefill length) for the layer shapes of Llama 7B:

![image](https://github.com/mgoin/torch-fp8/assets/3195154/fb769578-5539-4c37-93cb-e8f044b4f958)

Obviously some interesting stuff seeing more than 2x speedup, but I wouldn't take that so seriously since this is individual layer benchmarking and it is likely fitting into cache at small M (however I am unsure about the bump at M between 100-130, very interesting)

Here is a benchmark that is less fine-grained but showing the impact of dynamic activation quantization:

![image](https://github.com/vllm-project/vllm/assets/3195154/77904822-dac6-4f84-98a0-804877ff9fef)

I'll continue benchmarking and investigating how we can minimize overhead on the individual layer level and on different GPUs, but it does look like the performance of `scaled_mm` is highly variable for M<192 on an RTX 4090. This is bound to be worse on a GPU with more memory bandwidth like an H100.
---
### comaniac @ 2024-04-17T20:15:53Z

Thanks for the benchmarking and it basically aligns with my results on H100. I'll also do more investigations to see whether we could amortize this overhead.

btw my benchmark code is pretty much the same as yours. The only different is I added `@torch.compile` to `to_float8`. It helps a little.
---
### comaniac @ 2024-04-17T23:55:53Z

@WoosukKwon @zhuohan123 @robertgshaw2-neuralmagic @mgoin @pcmoritz 

Although we haven't achieved a decent performance with this FP8 support, I'd like to get this PR reviewed first, because it seems better to build follow-up optimizations on top of this PR so merging this PR could unblock others to try more interesting stuffs (unless you have concerns about the current interface and API implementation).

In short, this PR now offers:
1. Loading FP16 checkpoints and per-tensor quantizing linear weights to FP8 during the first forward pass.
2. Per-tensor quantizing activations to FP8 in each forward pass.
3. We also accept FP8 checkpoints. It has to store linear weights in `torch.float8_e4m3fn` along with an additional `w_scale` parameter. The major benefit of loading a FP8 checkpoint is lowering the memory footprint.

Some follow-ups (I probably don't have time to do all of these so you're welcome to take anything you interest):
1. Confirm with PyTorch and NVIDIA about the performance of `torch._scaled_mm`.
2. Benchmark TGI v2.0.0 with FP8, and confirm that we are on the same boat now.
3. If the above 2 points are true, we need a high-efficient GEMM kernel for small number of tokens. @pcmoritz suggested that we could try Triton (https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
4. Investigate about how can we reduce/amortize activation quantization overheads. There are two possible directions. First, compute more ops in FP8 so that one quantization can benefit more. Second, leverage delayed scaling (i.e. reuse scaling factor), but this may affect the result quality significantly.
5. Enable FP8 in attention layer: Flash attention should have supported FP8 so we should be good with prefill. For decoding, we better integrate FlashInfer directly as it supports FP8 too.

---
### comaniac @ 2024-04-18T02:31:52Z

Thanks for the review! Will add some tests tomorrow along with your suggestions.
---
### comaniac @ 2024-04-18T17:34:00Z

> In order to test the FP8 weight loading, could you include an example of how you save the FP8 model format? Something like the [FP8 KV Cache example](https://github.com/vllm-project/vllm/tree/main/examples/fp8) would be nice

Thanks for the pointer. I could actually make it compatible with the existing quantization flow.
---
### comaniac @ 2024-04-18T23:50:37Z

After more investigations, it seems not straightforward to support both FP16 and FP8 checkpoints in the same Linear implementation, so this PR would focus on FP16 checkpoints first (I'll explain the reasons in details but that's tl;dr). We will probably have an RFC discussing FP8 checkpoint standards later.

---
**[tl; dr] Issues with FP8 checkpoint**

The FP8 checkpoint should include
1. weights in `torch.float8_e4m3fn`.
2. The scaling factor of each weight parameter.

The major issue is most HF checkpoints save `proj_q`, `proj_k`, `proj_v`, `gate_proj`, `up_proj` to separate tensors; while we always merge them to `qkv` and `gate_up_proj`. Thus, how to merge their scaling factors becomes an issue, because the scaling factor should be based on the input tensor of `torch._scaled_mm` (and all future supported ops). There are two cases based on different quantization methods:

1. **The FP8 checkpoint was quantized without calibration data.** In this case, when we are loading q, k, and v incrementally, we dequantize partial loaded qkv tensor back to float16 using the current scaling factor, add the new loaded tensor, and re-quantize them together to get the new float8 tensor as well as the updated scaling factor. While this approach is doable, it requires some changes in the interface, because the current `weight_loader` is completely isolated from other parameters.
2. **The FP8 checkpoint was quantized with calibration data (e.g., AMMO, SmoothQuant, etc).** In this case, we cannot do anything unfortunately, unless we adopt per-channel scaling factor.

I guess supporting both cases is inevitable, so we should definitely focus more on (2) as its solution must cover (1). Just for reference, TensorRT-LLM solves this by first loading the weight to their model implementation (so qkv is already fused), performs quantization with calibration data, and save the FP8 checkpoint for their model (so the parameter names are diverse from the original HF model). We can probably do the same, but that means every model wants to run on vLLM with FP8 has to use our quantization flow. Anyways, we could have more discussions later.

---