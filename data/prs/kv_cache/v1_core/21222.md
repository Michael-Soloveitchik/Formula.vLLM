# PR #21222: [Core] Introduce popleft_n and append_n in FreeKVCacheBlockQueue to further optimize block_pool

- field: scheduling  | area: v1/core
- author: Jialin
- state: MERGED  created: 2025-07-19T09:39:37Z  merged: 2025-07-22T13:17:47Z
- url: https://github.com/vllm-project/vllm/pull/21222
- labels: performance, ready, v1

## Body

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.


## Purpose
Most of the block_pool operators are on critical path
- forward path is hard blocked by kv block allocation
- decode cycle end is hard blocked by kv block free

In this PR, we're focusing on further optimization these 2 operators.

### Bulk popleft instead of popleft n times
Originally, in block_pool.get_new_blocks, we popped blocks one at a time, which would triggered the second block to fake head connections (which are unnecessary operations as the second block might be popped right after this).

As we knew total number of blocks to pop ahead, we could simply introduce popleft_n for buck popleft. Overall, the number link list operations to linked list of popleft_n would only be half of n popleft.

### Bulk append instead of append n times
Similar, in block_pool.free_blocks, we invoke append one at a time. Introducing bulk append would also cut link list operations by half.

## Test Plan
- Evaluate with benchmark scripts
- Evaluate with benchmark_blockpoll
- New Unit Test for append_n and popleft_n are added

## Test Result

### benchmark scripts
- Get new blocks improved from 0.15ms to 0.008ms
- Free new blocks improved from 33us to 9us

After
<img width="840" height="168" alt="Screenshot 2025-07-19 at 2 33 42 AM" src="https://github.com/user-attachments/assets/a22991d1-0f3a-4576-ba7e-a9e0cb9aebe4" />
<img width="603" height="173" alt="Screenshot 2025-07-19 at 2 34 21 AM" src="https://github.com/user-attachments/assets/8d986488-c278-4331-9186-d84aebcdb92d" />

Before
<img width="838" height="276" alt="Screenshot 2025-07-19 at 2 31 59 AM" src="https://github.com/user-attachments/assets/86bb6de6-e42b-4b38-b3c6-97d9d773e762" />
<img width="859" height="311" alt="Screenshot 2025-07-19 at 2 32 55 AM" src="https://github.com/user-attachments/assets/f7cefd94-b0c3-4f8d-b374-87951c864211" />

### benchmark_blockpool
As expected, get_blocks and free_blocks times are cut in half.

After
<img width="800" height="230" alt="Screenshot 2025-07-19 at 2 22 07 AM" src="https://github.com/user-attachments/assets/85ab0be7-90f0-4da8-97f4-b2079ac06433" />
Before
<img width="805" height="233" alt="Screenshot 2025-07-19 at 2 23 21 AM" src="https://github.com/user-attachments/assets/32a8ad40-33af-4585-877c-e8e4d9bd7749" />


## (Optional) Documentation Update

<!--- pyml disable-next-line no-emphasis-as-heading -->


## Comments

### github-actions @ 2025-07-19T09:39:45Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### Jialin @ 2025-07-19T09:40:11Z

resolve https://github.com/vllm-project/vllm/issues/21141
---