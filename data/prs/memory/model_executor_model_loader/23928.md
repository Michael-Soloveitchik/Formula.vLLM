# PR #23928: [Model loader]: support multi-thread model weight loading

- field: memory  | area: model_executor/model_loader
- author: BraveY
- state: MERGED  created: 2025-08-29T10:34:05Z  merged: 2025-09-08T18:49:40Z
- url: https://github.com/vllm-project/vllm/pull/23928
- labels: ready

## Body

<!-- markdownlint-disable -->

## Purpose
This pull request introduces support for multi-threaded model weight loading in the `DefaultModelLoader`, allowing faster initialization of models by parallelizing the loading of safetensor and pt/bin weight files. It also adds configuration options to control threading behavior and validates extra config keys.

The code was modified from the [Sglang's PR].(https://github.com/sgl-project/sglang/pull/7277) 

### Multi-threaded weight loading support

* Added `multi_thread_safetensors_weights_iterator` and `multi_thread_pt_weights_iterator` functions to `weight_utils.py`, enabling parallel loading of safetensor and pt/bin files using Python's `ThreadPoolExecutor`. [[1]](diffhunk://#diff-c1cbb1d99298476e618b6fc564452c8cd9b3ecbf688c5cfb44506209547f7f3dR477-R502) [[2]](diffhunk://#diff-c1cbb1d99298476e618b6fc564452c8cd9b3ecbf688c5cfb44506209547f7f3dR583-R609)

* Updated `DefaultModelLoader` to use these new iterators when the `enable_multithread_load` extra config is set, with configurable `num_threads` (default 8). [[1]](diffhunk://#diff-ebfc523d6559bb3f30b9c05ae0f912120c22ddf17ae67f2a0a93e356245c0508R23) [[2]](diffhunk://#diff-ebfc523d6559bb3f30b9c05ae0f912120c22ddf17ae67f2a0a93e356245c0508R33-R35) [[3]](diffhunk://#diff-ebfc523d6559bb3f30b9c05ae0f912120c22ddf17ae67f2a0a93e356245c0508R190) [[4]](diffhunk://#diff-ebfc523d6559bb3f30b9c05ae0f912120c22ddf17ae67f2a0a93e356245c0508R210-R233)

### Configuration validation and error handling

* Added validation for extra config keys in `DefaultModelLoader`, raising an error if unexpected keys are provided to avoid misconfiguration. Only `enable_multithread_load` and `num_threads` are allowed.

### Dependency updates

* Imported `concurrent.futures` in `weight_utils.py` to support multi-threading for weight loading.
## Test Plan
I tested the default loading logic and multi-threaded loading logic on a single NVME SSD.
The loaded model is QwQ-32B, and the model size is 62G.
Test cmd:
```
python -m vllm.entrypoints.openai.api_server \
    --model=/data/model/QwQ-32B\
    --trust-remote-code \
    --served-model-name auto \
    --distributed-executor-backend=mp \
    --port 8006 \
    -tp=8 \
    --enforce-eager \
    --max-num-seqs 192 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --block-size 128 \
    --gpu-memory-utilization 0.9 \
    --enable-prompt-tokens-details \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
```
## Test Result
Single NVME ssd :
Module Loader | Load Time (s) | Average Bandwidth (GB/s)
-- | -- | --
disable multi thread loading(before) | 20 | 3.1
enable multi thread loading(after) | 14 | 4.42
Load Weight speed up 1.43x.

Test results of DeepSeek-R1-0528 safetensor loading in NVME ssd PCIE Gen4 (Speed 16GT/s, Width x4) could be found in this [PR](https://github.com/sgl-project/sglang/pull/7277).  This acceleration effect should also apply to vllm, because the loading logic of Sglang and vllm is similar.
<img width="1658" height="338" alt="image" src="https://github.com/user-attachments/assets/a8860739-d4f8-456a-b898-d22ac981ff26" />


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

### BraveY @ 2025-09-01T02:11:11Z

> Good enhancement! Please
> 
> * Take a look at Gemini comment
> * is there a downside to do this by default
> * seems like while the default loader mmap the weights file, so there won’t be excessive usage of RAM, this approach first copy the weights to cpu? I think this is indeed a behavior change and can cause thrashing (the model weights file might already be in disk page cache)

Thanks for the review. I will add further tests to observe the impact on RAM, as well as the status of weight files in PageCache.
---
### BraveY @ 2025-09-01T10:46:12Z

> Good enhancement! Please
> 
> * Take a look at Gemini comment
> * is there a downside to do this by default
> * seems like while the default loader mmap the weights file, so there won’t be excessive usage of RAM, this approach first copy the weights to cpu? I think this is indeed a behavior change and can cause thrashing (the model weights file might already be in disk page cache)

- The [load_file ](https://github.com/huggingface/safetensors/blob/54b35101e4fead20d5104468e1093250f267ead1/bindings/python/py_src/safetensors/torch.py#L310) still relies `safe_open`, which internally uses mmap under the hood, just like the default loader.
- In my tests loading the QwQ-32B model from the page cache, the memory usage peaked at around 58GB — consistent with the behavior of the default loader. No additional memory overhead was observed.
- I haven't identified any negative side effects so far. Tests show that the time spent loading weights from the Page Cache using multi-threading is still slightly faster than the default read method by about 0.2 seconds for the QwQ-32B model.
---
### lionelvillard @ 2025-09-02T12:47:03Z

@BraveY do you clear the linux page cache in your testing? I'm wondering if the number you get are from reading from the page cache vs nvmes, and whether multi-threading help in both cases.


 
---
### BraveY @ 2025-09-03T02:46:21Z

> @BraveY do you clear the linux page cache in your testing? I'm wondering if the number you get are from reading from the page cache vs nvmes, and whether multi-threading help in both cases.

Yes, i have dropped the page cache before starting test. The test result are both from nvmes. In scenarios where reading directly from disk, multi-threading can significantly improve disk bandwidth, thereby reducing model loading time. In the case of PageCache, it can slightly reduce loading time.
---
### mergify @ 2025-09-08T13:56:53Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @BraveY.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### pwschuurman @ 2025-09-10T00:18:20Z

This PR has broken test_tensorizer tests: (eg: https://buildkite.com/vllm/ci/builds/29967/steps/canvas?sid=01992ee2-d80a-4abb-83bd-c2f14f10f811)
---
### BraveY @ 2025-09-10T02:03:47Z

> This PR has broken test_tensorizer tests: (eg: https://buildkite.com/vllm/ci/builds/29967/steps/canvas?sid=01992ee2-d80a-4abb-83bd-c2f14f10f811)

I will fix this issue immediately.
---
### BraveY @ 2025-09-10T03:11:04Z

> > This PR has broken test_tensorizer tests: (eg: https://buildkite.com/vllm/ci/builds/29967/steps/canvas?sid=01992ee2-d80a-4abb-83bd-c2f14f10f811)
> 
> I will fix this issue immediately.

Already fix by https://github.com/vllm-project/vllm/pull/24545
---
### njhill @ 2025-09-10T03:52:39Z

I'm a bit confused why the Tensorizer test passed in the original CI run for this PR...
---
### pwschuurman @ 2025-09-10T04:12:15Z

> I'm a bit confused why the Tensorizer test passed in the original CI run for this PR...

Upon closer inspection, the old code was executed, which resulted in a passing test case. If you look at the test logs for Tensorizer Test: https://buildkite.com/vllm/ci/builds/29824/steps/canvas?sid=01992a3e-d822-4dde-979b-f0bed2541d69#01992a3e-d902-4cdf-89c9-a4a5a055df90/165-329

```
[2025-09-08T16:56:46Z] (EngineCore_0 pid=481) DEBUG 09-08 09:56:46 [base_loader.py:47] Loading weights on cuda ...
[2025-09-08T16:56:46Z] (EngineCore_0 pid=481) INFO 09-08 09:56:46 [weight_utils.py:346] Using model weights format ['*.safetensors', '*.bin', '*.pt']
Loading pt checkpoint shards: 100% 1/1 [00:00<00:00,  1.57it/s]
[2025-09-08T16:56:47Z] (EngineCore_0 pid=481) INFO 09-08 09:56:47 [default_loader.py:237] Loading weights took 0.64 seconds
[2025-09-08T16:56:48Z] (EngineCore_0 pid=481) INFO 09-08 09:56:48 [gpu_model_runner.py:2232] Model loading took 0.2389 GiB and 1.171350 
```

This indicates that the executed `default_loader.py` emitted a lot line of `Loading weights took` from line `237`. However, if we look at the diff of #23928, we see that this log message **should have been emitted from line 266**: https://github.com/vllm-project/vllm/pull/23928/files#diff-ebfc523d6559bb3f30b9c05ae0f912120c22ddf17ae67f2a0a93e356245c0508L237

So it seems like there are certain paths in Python that are not getting recompiled in the current CI model.
---
### CSWYF3634076 @ 2025-09-19T07:24:50Z

@BraveY May I ask how to run the feature
---
### BraveY @ 2025-09-19T08:46:03Z

> @BraveY May I ask how to run the feature

`--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'` add this option.

Full command.
```
python -m vllm.entrypoints.openai.api_server \
    --model=/data/model/QwQ-32B\
    --trust-remote-code \
    --served-model-name auto \
    --distributed-executor-backend=mp \
    --port 8006 \
    -tp=8 \
    --enforce-eager \
    --max-num-seqs 192 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --block-size 128 \
    --gpu-memory-utilization 0.9 \
    --enable-prompt-tokens-details \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
```
---
### CSWYF3634076 @ 2025-09-19T08:49:15Z

> > @BraveY May I ask how to run the feature
> 
> --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' add this option.
Do I need to use the following command to clear the page cache before testing?
`echo 1 > /proc/sys/vm/drop_caches`
---
### BraveY @ 2025-09-19T09:03:29Z

> > > @BraveY May I ask how to run the feature
> > 
> > 
> > --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' add this option.
> > Do I need to use the following command to clear the page cache before testing?
> > `echo 1 > /proc/sys/vm/drop_caches`

Yes，if you want to test without weights in pagecache.  `echo 1 > /proc/sys/vm/drop_caches` I used this command to clear cache.
---