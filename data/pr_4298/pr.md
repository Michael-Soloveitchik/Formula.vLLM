# PR #4298: [Model] Adds Phi-3 support

- field: attention  | area: model_executor/models
- author: caiom
- state: MERGED  created: 2024-04-23T15:58:39Z  merged: 2024-04-25T03:06:57Z
- url: https://github.com/vllm-project/vllm/pull/4298
- labels: 

## Body

This PR adds support for [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) and [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct).

**Changes:**

1. Adds _PhiLongScaledRotaryEmbedding_ layer. This is a pure Pytorch implementation. Optimizations may come in the future.
2. Makes the layer name mapping in the llama model more robust to avoid layer name collision.

Models tested before this PR:
- microsoft/Phi-3-mini-4k-instruct
- microsoft/Phi-3-mini-128k-instruct
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Meta-Llama-3-8B
- mistralai/Mistral-7B-Instruct-v0.2

**BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE**

---

<details>
<!-- inside this <details> section, markdown rendering does not work, so we use raw html here. -->
<summary><b> PR Checklist (Click to Expand) </b></summary>

<p>Thank you for your contribution to vLLM! Before submitting the pull request, please ensure the PR meets the following criteria. This helps vLLM maintain the code quality and improve the efficiency of the review process.</p>

<h3>PR Title and Classification</h3>
<p>Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:</p>
<ul>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[Doc]</code> for documentation fixes and improvements.</li>
    <li><code>[Model]</code> for adding a new model or improving an existing model. Model name should appear in the title.</li>
    <li><code>[Frontend]</code> For changes on the vLLM frontend (e.g., OpenAI API server, <code>LLM</code> class, etc.) </li>
    <li><code>[Kernel]</code> for changes affecting CUDA kernels or other compute kernels.</li>
    <li><code>[Core]</code> for changes in the core vLLM logic (e.g., <code>LLMEngine</code>, <code>AsyncLLMEngine</code>, <code>Scheduler</code>, etc.)</li>
    <li><code>[Hardware][Vendor]</code> for hardware-specific changes. Vendor name should appear in the prefix (e.g., <code>[Hardware][AMD]</code>).</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If the PR spans more than one category, please include all relevant prefixes.</p>

<h3>Code Quality</h3>

<p>The PR need to meet the following code quality standards:</p>

<ul>
    <li>We adhere to <a href="https://google.github.io/styleguide/pyguide.html">Google Python style guide</a> and <a href="https://google.github.io/styleguide/cppguide.html">Google C++ style guide</a>.</li>
    <li>Pass all linter checks. Please use <a href="https://github.com/vllm-project/vllm/blob/main/format.sh"><code>format.sh</code></a> to format your code.</li>
    <li>The code need to be well-documented to ensure future contributors can easily understand the code.</li>
    <li>Include sufficient tests to ensure the project to stay correct and robust. This includes both unit tests and integration tests.</li>
    <li>Please add documentation to <code>docs/source/</code> if the PR modifies the user-facing behaviors of vLLM. It helps vLLM user understand and utilize the new features or changes.</li>
</ul>

<h3>Notes for Large Changes</h3>
<p>Please keep the changes as concise as possible. For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue (RFC) discussing the technical design and justification. Otherwise, we will tag it with <code>rfc-required</code> and might not go through the PR.</p>

<h3>What to Expect for the Reviews</h3>

<p>The goal of the vLLM team is to be a <i>transparent reviewing machine</i>. We would like to make the review process transparent and efficient and make sure no contributor feel confused or frustrated. However, the vLLM team is small, so we need to prioritize some PRs over others. Here is what you can expect from the review process: </p>

<ul>
    <li> After the PR is submitted, the PR will be assigned to a reviewer. Every reviewer will pick up the PRs based on their expertise and availability.</li>
    <li> After the PR is assigned, the reviewer will provide status update every 2-3 days. If the PR is not reviewed within 7 days, please feel free to ping the reviewer or the vLLM team.</li>
    <li> After the review, the reviewer will put an <code> action-required</code> label on the PR if there are changes required. The contributor should address the comments and ping the reviewer to re-review the PR.</li>
    <li> Please respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.
 </li>
</ul>

<h3>Thank You</h3>

<p> Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM. Your contributions make vLLM a great tool for everyone! </p>


</details>




## Comments

### simon-mo @ 2024-04-23T16:45:26Z

> This is a pure Pytorch implementation. Optimizations may come in the future.

Interested to see whether `torch.compile` can help here.  
---
### ArthurZucker @ 2024-04-23T17:11:26Z

Torch compile compatible version is [here](https://github.com/huggingface/transformers/blob/93cea8af5372e2532fd4f9e010a3a1527afcd0ea/src/transformers/models/gemma/modeling_gemma.py#L110) without scaling. @caiom would you mind syncronizing with https://github.com/huggingface/transformers/pull/30423 and @gugarosa to make sure we have a similar format? 

---
### caiom @ 2024-04-24T00:28:25Z

I've updated the naming to match the ones used in HF and the config changes. Let's wait until the HF PR is merged.
---
### pseudotensor @ 2024-04-24T17:25:05Z

HF PR is merged: https://github.com/huggingface/transformers/pull/30423
---
### caiom @ 2024-04-24T19:22:38Z

About _torch.compile_: The _Phi3SuScaledRotaryEmbedding_ is fully compatible with it and there are no graph breaks. However, I noticed a longer startup time and some recompilation warnings. Maybe someone with more compile experience can add this feature later?

The speedup is between 2-5x depending on the sequence length.
---
### esmeetu @ 2024-04-25T00:51:35Z

LGTM! Could you help adding this model to document and readme file?
---
### caiom @ 2024-04-25T02:13:46Z

> LGTM! Could you help adding this model to document and readme file?

Thanks for the review. I've added Phi3 to README and to the Supported Models doc.
---
### pseudotensor @ 2024-04-26T05:37:45Z

Hi @caiom , Phi-3 with this is relatively slow for a 3B model.  Is there a way to speed it up?

I get about 73 tokens/sec on H100 with no in-context stuff.  And about 35 tokens/sec with context filled.  But that's with streaming.

I get better performance at 125 tokens/s with no in-context stuff when not streaming.
---
### caiom @ 2024-04-27T02:58:33Z

> Hi @caiom , Phi-3 with this is relatively slow for a 3B model. Is there a way to speed it up?
> 
> I get about 73 tokens/sec on H100 with no in-context stuff. And about 35 tokens/sec with context filled. But that's with streaming.
> 
> I get better performance at 125 tokens/s with no in-context stuff when not streaming.

Will be working on that.
---
### davidgxue @ 2024-04-27T03:28:46Z

Hi sorry it's me again, I am actually getting an error with Phi-3 after installing from source
```
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/conda/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/david_xue/vllm/vllm/entrypoints/openai/api_server.py", line 159, in <module>
    engine = AsyncLLMEngine.from_engine_args(
  File "/home/david_xue/vllm/vllm/engine/async_llm_engine.py", line 361, in from_engine_args
    engine = cls(
  File "/home/david_xue/vllm/vllm/engine/async_llm_engine.py", line 319, in __init__
    self.engine = self._init_engine(*args, **kwargs)
  File "/home/david_xue/vllm/vllm/engine/async_llm_engine.py", line 437, in _init_engine
    return engine_class(*args, **kwargs)
  File "/home/david_xue/vllm/vllm/engine/llm_engine.py", line 214, in __init__
    self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)
  File "/home/david_xue/vllm/vllm/core/scheduler.py", line 265, in __init__
    self.block_manager = BlockSpaceManagerImpl(
  File "/home/david_xue/vllm/vllm/core/block_manager_v1.py", line 223, in __init__
    assert sliding_window % block_size == 0, (sliding_window,
AssertionError: (2047, 16)
```
Sliding window is 2047 and that seems correct based on phi 3 code. Block size 16 is also correct...? Am I missing something?

I am currently working on a custom quant for Phi 3 which maybe related. Should I open a separate issue? 

Oh also I am not sure how this was tested but Phi 3 mini 4k has sliding window of 2047 but I think 128k context length version has sliding window disabled or something different?

---
### caiom @ 2024-04-27T03:32:54Z

Hi @davidgxue, sorry about that! Have a look here: https://github.com/vllm-project/vllm/pull/4380
---