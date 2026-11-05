# PR #33839: [Kernel][perf] optimize NCCL symm_mem vs custom_AR selection thresholds

- field: gpu_kernels  | area: distributed/device_communicators
- author: pkousha
- state: MERGED  created: 2026-02-04T22:39:30Z  merged: 2026-02-26T22:35:58Z
- url: https://github.com/vllm-project/vllm/pull/33839
- labels: performance, ready

## Body

## Purpose
Optimizes the decision logic for when to use NCCL symmetric memory allreduce vs custom allreduce based on H100 and GB200 benchmarks.

## Problem
The previous threshold-based approach only used NCCL symm_mem for large tensors (≥2MB for 4 GPUs, ≥1MB for 8 GPUs). However, benchmarks show that symm_mem also significantly outperforms custom_AR for **small tensors** (≤16KB), leaving performance on the table.

The old logic:
```python
# Only use symm_mem if tensor >= threshold
threshold = {"4": 2*MiB, "8": 1*MiB}
if input_tensor.nbytes >= threshold:
    return True  # use symm_mem
```

This missed the opportunity to use symm_mem for small tensors where it provides 1.21x-1.48x speedup.

## Solution
Replace the simple size threshold with a range-based approach that defines where custom_AR is preferred, enabling symm_mem for both small AND large tensor sizes:

```python
# Define mid-range where custom_AR wins
"custom_ar_preferred_ranges": {
    4: (16 * KiB, 512 * KiB),  # custom_AR wins for 32K-256K
    8: (16 * KiB, 128 * KiB),  # custom_AR wins for 32K-64K
}
# Use symm_mem outside these ranges (small and large tensors)
```

## Test Plan
Benchmarked allreduce performance using `benchmarks/kernels/benchmark_device_communicators.py` on:
- 8x H100 GPUs (single node, NVLink connected)
- 4x H100 GPUs (single node, NVLink connected)
- Tensor sizes: 2KB to 1GB
- 100 trials per configuration with CUDA graph enabled

```bash
NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 VLLM_USE_NCCL_SYMM_MEM=1 \
torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_device_communicators.py \
    --batch-sizes 1 --sequence-lengths 16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072 \
    --hidden-sizes 128 --num-trials 100 --use-cuda-graph
```

## Test Result

### 8 GPUs H100 Benchmark Results

| Size Range | Best Communicator | Speedup vs Custom AR-1stage |
|------------|-------------------|----------------------------|
| 2K - 16K | **PyNCCL-symm** | 1.35x - 1.48x faster |
| 32K - 64K | custom_AR-1stage | baseline |
| 128K - 1GB | **PyNCCL-symm** | 1.12x - 6.14x faster |

### 4 GPUs H100 Benchmark Results

| Size Range | Best Communicator | Performance |
|------------|-------------------|-------------|
| 2K - 16K | **PyNCCL-symm** | 1.21x - 1.30x faster than custom_AR |
| 32K - 256K | custom_AR-1stage | 1.07x - 1.35x faster than PyNCCL-symm |
| 512K - 1GB | **PyNCCL-symm** | 1.10x - 2.32x faster than custom_AR |

### Detailed Latency Comparison (8 GPUs)

| Tensor Size | custom_AR (μs) | PyNCCL-symm (μs) | Winner |
|-------------|----------------|------------------|--------|
| 2 KB | 6.8 | **4.6** | symm_mem (1.48x) |
| 4 KB | 6.9 | **4.8** | symm_mem (1.44x) |
| 8 KB | 7.0 | **5.0** | symm_mem (1.40x) |
| 16 KB | 7.2 | **5.3** | symm_mem (1.35x) |
| 32 KB | **7.5** | 8.1 | custom_AR |
| 64 KB | **8.2** | 9.4 | custom_AR |
| 128 KB | 12.1 | **10.8** | symm_mem (1.12x) |
| 256 KB | 18.5 | **14.2** | symm_mem (1.30x) |
| 1 MB | 58.3 | **32.1** | symm_mem (1.82x) |
| 16 MB | 812.4 | **132.3** | symm_mem (6.14x) |

### Summary of Changes

| World Size | Old Behavior | New Behavior |
|------------|--------------|--------------|
| 4 GPUs | symm_mem only for ≥2MB | symm_mem for ≤16K AND ≥512K |
| 8 GPUs | symm_mem only for ≥1MB | symm_mem for ≤16K AND ≥128K |
| >8 GPUs | Always symm_mem | Always symm_mem (unchanged) |

## Performance Figures
### H100 single node 
<img width="2083" height="1185" alt="4gpu-kernel_medians_plot_light" src="https://github.com/user-attachments/assets/8e3d0551-402b-4053-9ba3-2d272ef4abfe" />
<img width="2083" height="1185" alt="8gpu-kernel_medians_plot_light" src="https://github.com/user-attachments/assets/16801d96-b47c-4850-a311-ae5286e5f208" />

### B200 single node
<img width="5367" height="2072" alt="b200-8gpus-all-implementations" src="https://github.com/user-attachments/assets/275e70f5-b03a-4879-93d2-c286f3a67e05" />


## Comments

### pkousha @ 2026-02-24T06:57:27Z

@xinli-sw do you know whom should I ask reviews from for this issue?
---
### jasonlizhengjian @ 2026-02-24T16:34:43Z

Currently NCCL symm mem is guarded behind an env var that most users do not know about, should we make it one of the defaults based on these heuristics?
---
### Amir-19 @ 2026-02-24T21:38:13Z

> Currently NCCL symm mem is guarded behind an env var that most users do not know about, should we make it one of the defaults based on these heuristics?

I think we should. just change/add env var to disable it.
---
### pkousha @ 2026-02-26T23:08:31Z

> nccl is only installed in venv, not system

In nccl4py design, I still use C++ way of registering NCCL buffers to ensure compatibility with PyTorch mem pool. (see the chart below) Considering that, the dependency remains. Reading the code, the current flow of nccl_allocator finding the NCCL library is as follows:
1. Check VLLM_NCCL_SO_PATH environment variable 
    └─▶ If set, use that path directly 

  2. Otherwise, use standard library name: 
  └─▶CUDA: "libnccl.so.2"

  3. ctypes.CDLL(so_file) loads via system's dynamic linker: 
     └─▶ Searches LD_LIBRARY_PATH, /usr/lib, /usr/local/lib, etc. 

However @ilmarkov **raised a good point** that I wanted to discuss about handling NCCL versions in nccl4py vs default-nccl used for PyTorch. I'll discuss it in #33127

<section id="markdown-section-ca62a137-08ff-43e9-8677-a94bb21d6af0-10" class="markdown-section" data-markdown-raw="

### Why can't nccl4py do the allocation?" data-section-index="10" style="border-radius: 4px; line-height: 19.5px; margin: 6px 0px; position: relative; scroll-margin-bottom: 40px; scroll-margin-top: 40px; color: rgba(228, 228, 228, 0.92); font-family: -apple-system, &quot;system-ui&quot;, sans-serif; font-size: 13px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(20, 20, 20); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><h3 style="font-weight: 600 !important; font-size: 1.15em; line-height: 1.25; margin-bottom: 8px; margin-top: 18px;"><span>Why can't nccl4py do the allocation?</span></h3></section><div class="monaco-scrollable-element mac markdown-table-container" role="presentation" style="scrollbar-width: none; overflow: hidden; color: rgba(228, 228, 228, 0.92); font-family: -apple-system, &quot;system-ui&quot;, sans-serif; font-size: 13px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(20, 20, 20); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; position: relative; margin: 1em 0px; width: fit-content; max-width: 100%; border-color: color(srgb 0.894118 0.894118 0.894118 / 0.0737255); border-style: solid; border-width: 1px; border-image: none 100% / 1 / 0 stretch; border-radius: 6px;"><div class="markdown-table-wrapper" style="--fade-start: rgba(0,0,0,.25); --fade-end: #000; --markdown-table-fade-width: 10px; mask-image: none; overflow: hidden;">
Requirement | C++ Allocator | nccl4py
-- | -- | --
PyTorch CUDAPluggableAllocator integration | ✅ Returns void* | ❌ Returns cuda.core.Buffer
torch.cuda.use_mem_pool() support | ✅ | ❌
Automatic lifetime management | ✅ PyTorch handles | ❌ Manual

</div></div>
---