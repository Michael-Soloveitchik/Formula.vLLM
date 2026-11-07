# PR #10640: [V1] VLM - Run the mm_mapper preprocessor in the frontend process

- field: scheduling  | area: v1/engine
- author: alexm-redhat
- state: MERGED  created: 2024-11-25T15:55:50Z  merged: 2024-12-03T10:33:11Z
- url: https://github.com/vllm-project/vllm/pull/10640
- labels: frontend, ready

## Body

This PR adds support to run the multi-modal mapper/preprocessor (from huggingface) in the frontend process. Execution of 512 prompts with 64 output tokens results in 1.7X improvement. Command used: 

VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 python examples/offline_inference_vision_language.py -m llava --num-prompts 512 --modality image 

Without frontend generate() time is: 28.91 seconds
With frontend generate() time is: 16.84

## Comments

### github-actions @ 2024-11-25T15:56:02Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### mergify @ 2024-11-25T15:56:28Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @alexm-neuralmagic.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### alexm-redhat @ 2024-11-25T16:00:33Z

@rickyyx thanks for the suggestion on trying this!
---
### alexm-redhat @ 2024-11-26T15:28:37Z

@robertgshaw2-neuralmagic I think you right about P0 always running the mm_mapper, so I will remove the disable to simplify the code.
---
### njhill @ 2024-11-26T17:36:33Z

> @robertgshaw2-neuralmagic I think you right about P0 always running the mm_mapper, so I will remove the disable to simplify the code.

Does this mean we can then remove `mm_data` (and `mm_processor_kwargs`?) from `EngineCoreRequest`? :)
---
### ywang96 @ 2024-11-26T17:40:13Z

> > @robertgshaw2-neuralmagic I think you right about P0 always running the mm_mapper, so I will remove the disable to simplify the code.
> 
> Does this mean we can then remove `mm_data` (and `mm_processor_kwargs`?) from `EngineCoreRequest`? :)

Yea if P0 is always going to run the multimodal data processor (mm_mapper), then P1 should only need to receive `mm_inputs`
---
### alexm-redhat @ 2024-12-02T15:08:33Z

@njhill @ywang96 removed mm_data and mm_processor_kwargs from EngineCoreRequest.
---
### alexm-redhat @ 2024-12-02T18:08:30Z

Will revert changes to offline_inference_vision_language.py and see if I can use the other script.
---
### ywang96 @ 2024-12-03T10:20:56Z

Confirmed that the fix on tests [382fc0b](https://github.com/vllm-project/vllm/pull/10640/commits/382fc0be9e168fe8ae47176ba54fbdc126f36940) resolved the issue on CI, so I'm going to auto-merge this.
---