# PR #28597: [Perf] Optimize EAGLE prepare_inputs_padded with triton kernels

- field: gpu_kernels  | area: v1/spec_decode
- author: benchislett
- state: MERGED  created: 2025-11-12T22:33:09Z  merged: 2025-11-28T22:25:05Z
- url: https://github.com/vllm-project/vllm/pull/28597
- labels: speculative-decoding, ready, v1

## Body

## Purpose

Part 1 of a series of planned efforts to reduce the kernel call overhead in the EAGLE padded speculation pathway.

This PR tackles `next_token_ids_padded` and `prepare_inputs_padded` with a triton kernel for each that handles the GPU logic. The fused kernels are simple and cheap to run, reducing the overhead in these sections by over 10x.

This PR also inlines the positions slicing and removes `token_indices` from `prepare_inputs_padded`, since it is always an `arange` which can be more efficiently represented using an integer slice on the indexed tensor.

## Test Plan

Previous EAGLE tests should fully cover both functions already.

## Performance Validation

I constructed an artificial scenario by modifying the tests with an NVTX range and a spin-loop that allows the CPU to run ahead, as a realistic proxy for the model execution. Here is the measured overhead in each case. Note that the time scale in the screenshots (x axis) is not normalized.

See early commits on this PR for the setup for this testing.

### Next token ids (reference)
About **50us**
<img width="1351" height="241" alt="image" src="https://github.com/user-attachments/assets/cf670482-1cec-4e35-8cca-c3cbe75a3b28" />

### Next token ids (kernel)
Less than **5us (10x)**
<img width="1351" height="241" alt="image" src="https://github.com/user-attachments/assets/34f05afa-bb37-4e22-a170-85503bb228a8" />

### Prepare inputs (reference)
About **40us**
<img width="1351" height="241" alt="image" src="https://github.com/user-attachments/assets/7dafcff2-1754-4143-9c48-8b9981255530" />

### Prepare inputs (kernel)
Less than **2us (20x)**
<img width="1351" height="241" alt="image" src="https://github.com/user-attachments/assets/af9e31ae-4970-4d34-bd93-41e4e209d9cb" />


## Comments

### benchislett @ 2025-11-28T16:05:28Z

@LucasWilkinson we need to address https://github.com/vllm-project/vllm/pull/26498 (https://github.com/vllm-project/vllm/issues/26191) before removing the non-padded pathway.
---
### LucasWilkinson @ 2025-11-28T18:33:05Z

LGTM! Thanks for the updates

> @LucasWilkinson we need to address https://github.com/vllm-project/vllm/pull/26498 (https://github.com/vllm-project/vllm/issues/26191) before removing the non-padded pathway.

Makes sense, lets revisit after that is done  :+1:
---