# PR #6865: [core][misc] improve free_finished_seq_groups

- field: scheduling  | area: core/scheduler.py
- author: youkaichao
- state: MERGED  created: 2024-07-27T17:15:08Z  merged: 2024-07-30T21:32:12Z
- url: https://github.com/vllm-project/vllm/pull/6865
- labels: ready

## Body

`free_finished_seq_groups` is in the critical path, because the engine calls it after every model step.

the main branch searchs for all the requests to find finished requests, while new finished requests should only appear in running queue.

this simple optimization, turns out to be important, when we process large amount of offline batch inference, e.g. when users give `100k` prompts to the LLM engine. In this case, we are iterating all 100k data at every step.

I have heard users complaining that sending the whole dataset to `LLM` is slow, and he has to chunk the data into small batches (25 requests each) to make it fast. This PR might fix it.

Some micro benchmark to demonstrate the benefit when number of prompts are long and the model becomes small (thus overhead in scheduler becomes significant):

Note: main branch is commit https://github.com/vllm-project/vllm/commit/925de97e05dd4709fcd80691cb37da5e582c22e8

```shell
$ python benchmarks/benchmark_throughput.py --output-len 16 --input 16 --model facebook/opt-125m --num-prompts 10000

# main branch:
Throughput: 577.63 requests/s, 18484.31 tokens/s

# this PR:
Throughput: 674.46 requests/s, 21582.66 tokens/s

$ python benchmarks/benchmark_throughput.py --output-len 16 --input 16 --model facebook/opt-125m --num-prompts 100000

# main branch:
Throughput: 211.18 requests/s, 6757.85 tokens/s

# this PR:
Throughput: 522.91 requests/s, 16733.13 tokens/s
```

Conclusion:

When I increase the number of prompts from  10k to 100k, the main branch is slowed down by almost 3x. Improving the `free_finished_seq_groups` helps a lot, the slow down is about 40%. Apparently there are some other factors, but I think this might be the largest factor in processing with large datasets.

## Comments

### github-actions @ 2024-07-27T17:15:19Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which consists a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of default ones by unblocking the steps in your `fast-check` build on Buildkite UI. 

Once the PR is approved and ready to go, please make sure to run full CI as it is required to merge (or just use auto-merge).

 To run full CI, you can do one of these:
- Comment `/ready` on the PR
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### youkaichao @ 2024-07-27T17:47:56Z

Here is the flamechart of the main branch, with 100k prompts. It becomes clear that `free_finished_seq_groups` is the major source of overhead.

![flamechart](https://github.com/user-attachments/assets/adae061d-c70b-4b27-85cc-c6520b5c8e54)

---
### youkaichao @ 2024-07-27T18:51:46Z

Here is the flame chart of this PR, with various number of prompts:

10k prompts:

![flamechart_10k](https://github.com/user-attachments/assets/84fce574-41a7-4057-8e24-71c253dd1560)

100k prompts:

![flamechart_100k](https://github.com/user-attachments/assets/7eca542c-bc83-4a4e-a374-7612dd10d10a)

500k prompts:
![flamechart_500k](https://github.com/user-attachments/assets/de7c61a0-fdc6-4353-b060-e880e758e05c)

Even after this PR, there is still a slow down factor that is proportional to the number of waiting requests. The flamechat points to:

https://github.com/vllm-project/vllm/blob/fad5576c58864a6c2cf528f67e60e03a949b3dac/vllm/core/scheduler.py#L680 

which is indeed very inefficient.
---
### mgoin @ 2024-07-29T19:08:18Z

@youkaichao I think you must rebase to pass the failing Examples Test
```
[2024-07-29T19:01:27Z] python3: can't open file '/vllm-workspace/examples/llava_example.py': [Errno 2] No such file or directory
```
---
### youkaichao @ 2024-07-30T04:32:50Z

> @youkaichao I think you must rebase to pass the failing Examples Test
> 
> ```
> [2024-07-29T19:01:27Z] python3: can't open file '/vllm-workspace/examples/llava_example.py': [Errno 2] No such file or directory
> ```

If there's no more comments, I will force-merge to avoid one additional ci cost :)
---
### youkaichao @ 2024-07-30T04:39:53Z

> @youkaichao I think we need to understand the code before getting this PR merged.

LGTM. Please take your time. I think the scheduler logic is quite unclear though. Might have lots of bugs there.
---
### youkaichao @ 2024-07-30T04:50:04Z

@WoosukKwon please feel free to take this PR if you want some modifications!
---