# PR #27673: [Core] Early return in SlidingWindowManager.remove_skipped_blocks

- field: scheduling  | area: v1/core
- author: Jialin
- state: MERGED  created: 2025-10-28T18:53:40Z  merged: 2025-10-29T01:32:34Z
- url: https://github.com/vllm-project/vllm/pull/27673
- labels: frontend, ready, v1

## Body

## Purpose
When SlidingWindowAttention is enabled, remove_skipped_blocks consumes ~25% of cost under allocate_slots.

In this PR, we leverage the fact that, most remove_skipped_blocks invocations does NOT remove any blocks. By introducing some lightweight checks ahead, we should be able to get rid of multiple function invocations and list comprehesions against empty list.

## Test Plan & Test Result

When early stop happened, the cost reduced from 6.3us to 1.6us now.

**After**
<img width="1085" height="152" alt="Screenshot 2025-10-28 at 11 48 39 AM" src="https://github.com/user-attachments/assets/06ded859-57c3-4dd7-8130-70c446ce638e" />

**Before**
<img width="1042" height="191" alt="Screenshot 2025-10-28 at 11 48 55 AM" src="https://github.com/user-attachments/assets/952779e5-18f2-49e9-ad5c-2171bd746f81" />

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

### heheda12345 @ 2025-10-28T21:24:02Z

@codex review
---
### chatgpt-codex-connector @ 2025-10-28T21:28:30Z

Codex Review: Didn't find any major issues. Nice work!

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

[Your team has set up Codex to review pull requests in this repo](http://chatgpt.com/codex/settings/general). Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.




Codex can also answer questions or update the PR. Try commenting "@codex address that feedback".
            
</details>
---