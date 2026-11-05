# PR #12563: [Feature] Fix guided decoding blocking bitmask memcpy

- field: decoding  | area: model_executor/guided_decoding
- author: xpbowler
- state: MERGED  created: 2025-01-29T21:16:01Z  merged: 2025-01-31T23:37:30Z
- url: https://github.com/vllm-project/vllm/pull/12563
- labels: performance, structured-output, ready

## Body

**[Guided decoding performance optimization]** Sending the guided decoding bitmask in xgrammar to the GPU (`self.token_bitmask.to(scores.device)`) is a blocking operation that prevents the CPU from pre-launching the sampler kernels. The CPU waits until decode is complete, then copies the bitmask over. This PR changes the operation to async via setting `non-blocking=True`.

(Current) The CPU is blocked on a `cudaStreamSynchronize` and only pre-empts the sampling kernels after bitmask application. Below is the Nsys profile for one decode phase from Llama 3.1 8B.
![image](https://github.com/user-attachments/assets/8997eae1-b822-4f52-beb8-ef19a7c6b824)

With the optimization, this is no longer the case:
![image](https://github.com/user-attachments/assets/6d5ea83f-f169-4f98-a8c1-41c719b3e1e7)




## Comments

### github-actions @ 2025-01-29T21:16:13Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### xpbowler @ 2025-01-29T21:56:06Z

> This makes sense, thanks! LGTM pending green CI
> 
> Showing the profile is great, also showing an e2e speedup (even if small) would be nice

For single request benchmarks with `Llama 3.1 8B` running on H100, the improvement in tok/s was ~5%:

**Single request**
- 87.5tok/s, guided unoptimized
- 92 tok/s, guided optimized
---