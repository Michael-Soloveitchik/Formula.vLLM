# PR #37252: [Perf] Set Flashinfer sparse MLA as default backend for FP8 kv cache

- field: gpu_kernels  | area: rocm_amd
- author: wzhao18
- state: CLOSED  created: 2026-03-17T04:26:55Z  merged: 2026-03-17T20:09:21Z
- url: https://github.com/vllm-project/vllm/pull/37252
- labels: documentation, ready, nvidia

## Body

## Purpose
This PR sets Flashinfer sparse MLA as default backend for FP8 KV cache for better performance.

## Test Plan


## Test Result
Kernel microbenchmark results: https://github.com/vllm-project/vllm/issues/35807

E2E results with different TP (with EP enabled)
<img width="2384" height="1035" alt="nvidia_DeepSeek-V3 2-NVFP4_backend_cmp_isl8192_osl1024" src="https://github.com/user-attachments/assets/c0e284c1-b94f-43d4-bf50-e7f6f5b662d9" />
- Flashinfer shows significantly better performance across the pareto for TP=1, 4, 8.
- For TP=2, it shows slightly worse performance than flashMLA at higher concurrency. This is a known issue from kernel microbenchmark that the kernel has unusually bad performance at TP=2. For simplicity and the small E2E perf gap, we will set Flashinfer as the default backend for all TP sizes for FP8 KV types.

Command:
```
vllm serve \
    nvidia/DeepSeek-V3.2-NVFP4 \
    --trust-remote-code \
    --stream-interval 20 \
    --no-enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --attention-backend {FLASHMLA_SPARSE, FLASHINFER_MLA_SPARSE} \
    --data-parallel-size {1,2,4,8} \
    --tensor-parallel-size {8,4,2,1} \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.8
```

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

### mergify[bot] @ 2026-03-17T04:27:32Z

Documentation preview: https://vllm--37252.org.readthedocs.build/en/37252/

### wzhao18 @ 2026-03-17T04:40:05Z

cc @MatthewBonanni 

### wzhao18 @ 2026-03-17T16:54:01Z

@MatthewBonanni Done. Thanks!
