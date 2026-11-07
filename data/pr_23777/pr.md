# PR #23777: [LoRA] Much faster startup when LoRA is enabled

- field: scheduling  | area: v1/worker
- author: andylolu2
- state: MERGED  created: 2025-08-27T21:50:18Z  merged: 2025-08-30T15:37:39Z
- url: https://github.com/vllm-project/vllm/pull/23777
- labels: ready, v1

## Body


## Purpose

Starting a server with LoRA takes a very long time, especially if `--max-loras` is set to a large-ish value.
It takes a long time because every call to `dummy_run` destroys and re-creates dummy lora weights. 

In this PR, I make the lora destruction optional, such that consecutive runs of `dummy_run` during warmup / cuda graph capture can share the same dummy loras.

## Test Plan

```
vllm serve meta-llama/Meta-Llama-3-70B --enable-lora --tensor-parallel-size 4 --max-loras 32
```

## Test Result

Cuda graph capture is >20x faster with my test command.

Before
```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|████████████| 67/67 [06:09<00:00,  5.52s/it]
```

After
```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|████████████| 67/67 [00:15<00:00,  4.38it/s]
```

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>


## Comments

(no discussion comments)