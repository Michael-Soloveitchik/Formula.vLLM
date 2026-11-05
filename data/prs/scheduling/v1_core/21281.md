# PR #21281: [Core] Minimize number of dict lookup in _maybe_evict_cached_block

- field: scheduling  | area: v1/core
- author: Jialin
- state: MERGED  created: 2025-07-21T05:24:50Z  merged: 2025-07-22T05:37:34Z
- url: https://github.com/vllm-project/vllm/pull/21281
- labels: ready, v1

## Body

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.


## Purpose
With https://github.com/vllm-project/vllm/pull/21222, we noticed that _maybe_evict_cached_block became the new bottleneck for allocate_new_block.
<img width="1070" height="518" alt="Screenshot 2025-07-20 at 9 47 09 PM" src="https://github.com/user-attachments/assets/91a468de-ab31-4f11-9e68-2db2cd10a39f" />

And we notice that, we do duplicated key lookup against the same dict 4 times, and we could easily cut it down into 2.

## Test Plan
Compare trace with and without the change on top of https://github.com/vllm-project/vllm/pull/21222

## Test Result
Quite significant improvements 0.13ms -> 0.03ms

After
<img width="930" height="545" alt="Screenshot 2025-07-20 at 9 46 23 PM" src="https://github.com/user-attachments/assets/822b55c8-cd18-4719-9802-f9b1310e145c" />

Before
<img width="1070" height="518" alt="Screenshot 2025-07-20 at 9 47 09 PM" src="https://github.com/user-attachments/assets/da000717-3099-488b-9757-b12953131edc" />

## (Optional) Documentation Update

<!--- pyml disable-next-line no-emphasis-as-heading -->


## Comments

### github-actions @ 2025-07-21T05:24:57Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### Jialin @ 2025-07-21T07:41:46Z

Resolve https://github.com/vllm-project/vllm/issues/21141
---