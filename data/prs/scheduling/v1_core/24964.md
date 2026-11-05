# PR #24964: [Core] Reuse empty block lists whenever possible in KVCacheBlocks to mitigate GC costs

- field: scheduling  | area: v1/core
- author: Jialin
- state: MERGED  created: 2025-09-16T12:31:33Z  merged: 2025-10-14T19:58:43Z
- url: https://github.com/vllm-project/vllm/pull/24964
- labels: frontend, ready, v1, multi-modality

## Body


## Purpose
Majority of the the Generation 0 collect objects are related to KVCacheBlocks, and we found most of them are actually referring to empty blocks.
<img width="661" height="162" alt="Screenshot 2025-09-28 at 7 52 23 AM" src="https://github.com/user-attachments/assets/aae02ad1-8858-4444-80a1-b42e5d0d54e1" />

In this PR, we mainly made 2 changes
- change KVCacheBlocks.blocks from tuple[list] to tuple[tuple] to indicate immutable after creating
- pre-construct a empty KVCacheBlock member variable in KVCacheManager for reuse (in order to avoid GC)

## Test Plan & Test Result
Model: facebook/opt-125m
Prefill-Heavy workload: Input 2000, Output 48
Decode-Heavy workload: Input 48, Output 2000

We could see, GC costs are significantly smaller with the PR (especially for decode-heavy workload). It could roughly convert to 3-4% throughput improvements.
<img width="537" height="178" alt="Screenshot 2025-09-28 at 7 41 05 AM" src="https://github.com/user-attachments/assets/d5cd8855-327d-4220-a008-b2103b820481" />

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

### Jialin @ 2025-09-16T16:18:38Z

Resolve https://github.com/vllm-project/vllm/issues/24321
---
### Jialin @ 2025-09-19T18:49:43Z

Discussed with @heheda12345 offline. The main discussion point is to come up with solutions to minimize the code signature changes and avoid if else:
- A potential solution would be let the allocate_new_blocks interface to returned a static KVCacheBlocks contains ImmutableEmptyList (where ImmutableEmptyList extends list with all mutations got removed). In this way, all signatures are kept as is, and all object allocations around empty KVCacheBlocks are removed.

Change the PR to draft before addressing the concern.
---
### mergify @ 2025-10-02T18:50:08Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Jialin.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### heheda12345 @ 2025-10-06T03:34:48Z

@Jialin can you rebase the PR?
See https://vllm-dev.slack.com/archives/C07R5Q1Q2BB/p1759663228844749 for details.
---
### Jialin @ 2025-10-06T23:14:43Z

> @Jialin can you rebase the PR? See https://vllm-dev.slack.com/archives/C07R5Q1Q2BB/p1759663228844749 for details.

Rebased, PTAL. Thanks! @heheda12345 
---
### mergify @ 2025-10-12T16:52:40Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Jialin.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---