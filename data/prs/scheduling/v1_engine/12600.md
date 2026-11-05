# PR #12600: [V1] Bugfix: Validate Model Input Length

- field: scheduling  | area: v1/engine
- author: robertgshaw2-redhat
- state: MERGED  created: 2025-01-31T02:23:00Z  merged: 2025-02-01T02:32:04Z
- url: https://github.com/vllm-project/vllm/pull/12600
- labels: ready

## Body

SUMMARY:
* avoid crashing the engine when we get an input longer than max_model_len

FIX #12567(*link existing issues this PR will resolve*)



## Comments

### github-actions @ 2025-01-31T02:23:12Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### WoosukKwon @ 2025-01-31T05:20:01Z

@robertgshaw2-redhat Thanks for the PR. However, is this the desirable behavior?

This PR basically lets the `LLM` instance abort, but with a more informed message. Therefore, users will lose all the progress before the request.
IIUC, in V0, we didn't do this, but returned an empty output for the quest with a special finished reason. The `LLM` instance processes other requests normally.
---
### mgoin @ 2025-01-31T19:16:34Z

@WoosukKwon Within the server, this does simply fail the request and keep the server alive

Server log sending 1 good, 1 bad, and 1 good request:
```
INFO:     Started server process [3928685]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:40776 - "GET /v1/models HTTP/1.1" 200 OK
INFO 01-31 19:05:39 chat_utils.py:330] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
INFO 01-31 19:05:39 logger.py:37] Received request chatcmpl-573665c74ccb4f4090b71560fddb0860: prompt: '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=61, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 01-31 19:05:39 async_llm.py:159] Added request chatcmpl-573665c74ccb4f4090b71560fddb0860.
INFO 01-31 19:05:39 loggers.py:69] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs GPU KV cache usage: 0.0%.
INFO:     127.0.0.1:40776 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 01-31 19:05:44 loggers.py:69] Avg prompt throughput: 7.5 tokens/s, Avg generation throughput: 4.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs GPU KV cache usage: 0.0%.
INFO 01-31 19:05:49 loggers.py:69] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs GPU KV cache usage: 0.0%.
INFO 01-31 19:05:54 loggers.py:69] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs GPU KV cache usage: 0.0%.
INFO 01-31 19:05:59 loggers.py:69] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs GPU KV cache usage: 0.0%.
INFO:     127.0.0.1:47002 - "GET /v1/models HTTP/1.1" 200 OK
ERROR 01-31 19:06:01 serving_chat.py:191] Error in preprocessing prompt inputs
ERROR 01-31 19:06:01 serving_chat.py:191] Traceback (most recent call last):
ERROR 01-31 19:06:01 serving_chat.py:191]   File "/home/mgoin/code/vllm/vllm/entrypoints/openai/serving_chat.py", line 175, in create_chat_completion
ERROR 01-31 19:06:01 serving_chat.py:191]     ) = await self._preprocess_chat(
ERROR 01-31 19:06:01 serving_chat.py:191]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 01-31 19:06:01 serving_chat.py:191]   File "/home/mgoin/code/vllm/vllm/entrypoints/openai/serving_engine.py", line 432, in _preprocess_chat
ERROR 01-31 19:06:01 serving_chat.py:191]     prompt_inputs = await self._tokenize_prompt_input_async(
ERROR 01-31 19:06:01 serving_chat.py:191]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 01-31 19:06:01 serving_chat.py:191]   File "/home/mgoin/.local/share/uv/python/cpython-3.12.4-linux-x86_64-gnu/lib/python3.12/concurrent/futures/thread.py", line 58, in run
ERROR 01-31 19:06:01 serving_chat.py:191]     result = self.fn(*self.args, **self.kwargs)
ERROR 01-31 19:06:01 serving_chat.py:191]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 01-31 19:06:01 serving_chat.py:191]   File "/home/mgoin/code/vllm/vllm/entrypoints/openai/serving_engine.py", line 266, in _tokenize_prompt_input
ERROR 01-31 19:06:01 serving_chat.py:191]     return next(
ERROR 01-31 19:06:01 serving_chat.py:191]            ^^^^^
ERROR 01-31 19:06:01 serving_chat.py:191]   File "/home/mgoin/code/vllm/vllm/entrypoints/openai/serving_engine.py", line 289, in _tokenize_prompt_inputs
ERROR 01-31 19:06:01 serving_chat.py:191]     yield self._normalize_prompt_text_to_input(
ERROR 01-31 19:06:01 serving_chat.py:191]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 01-31 19:06:01 serving_chat.py:191]   File "/home/mgoin/code/vllm/vllm/entrypoints/openai/serving_engine.py", line 181, in _normalize_prompt_text_to_input
ERROR 01-31 19:06:01 serving_chat.py:191]     return self._validate_input(request, input_ids, input_text)
ERROR 01-31 19:06:01 serving_chat.py:191]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 01-31 19:06:01 serving_chat.py:191]   File "/home/mgoin/code/vllm/vllm/entrypoints/openai/serving_engine.py", line 238, in _validate_input
ERROR 01-31 19:06:01 serving_chat.py:191]     raise ValueError(
ERROR 01-31 19:06:01 serving_chat.py:191] ValueError: This model's maximum context length is 100 tokens. However, you requested 289 tokens in the messages, Please reduce the length of the messages.
INFO:     127.0.0.1:47002 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
INFO 01-31 19:06:04 loggers.py:69] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs GPU KV cache usage: 0.0%.
INFO 01-31 19:06:09 loggers.py:69] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs GPU KV cache usage: 0.0%.
INFO 01-31 19:06:14 loggers.py:69] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs GPU KV cache usage: 0.0%.
INFO:     127.0.0.1:48866 - "GET /v1/models HTTP/1.1" 200 OK
INFO 01-31 19:06:14 logger.py:37] Received request chatcmpl-d57c2926aa3b4b628f7b3cbf378588d7: prompt: '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=61, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 01-31 19:06:14 async_llm.py:159] Added request chatcmpl-d57c2926aa3b4b628f7b3cbf378588d7.
INFO:     127.0.0.1:48866 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

Error on the client for the failing length request:
```
Traceback (most recent call last):
  File "/home/mgoin/code/vllm/t.py", line 25, in <module>
    chat_response = client.chat.completions.create(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mgoin/venvs/vllm/lib/python3.12/site-packages/openai/_utils/_utils.py", line 279, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/mgoin/venvs/vllm/lib/python3.12/site-packages/openai/resources/chat/completions.py", line 859, in create
    return self._post(
           ^^^^^^^^^^^
  File "/home/mgoin/venvs/vllm/lib/python3.12/site-packages/openai/_base_client.py", line 1280, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mgoin/venvs/vllm/lib/python3.12/site-packages/openai/_base_client.py", line 957, in request
    return self._request(
           ^^^^^^^^^^^^^^
  File "/home/mgoin/venvs/vllm/lib/python3.12/site-packages/openai/_base_client.py", line 1061, in _request
    raise self._make_status_error_from_response(err.response) from None
openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': "This model's maximum context length is 100 tokens. However, you requested 289 tokens in the messages, Please reduce the length of the messages.", 'type': 'BadRequestError', 'param': None, 'code': 400}
```
---
### WoosukKwon @ 2025-01-31T19:19:07Z

@mgoin Thanks for testing. I've changed my comment from "`LLM` engine" to "`LLM` instance". I think the current PR works well with the API server, but does not solve the issue in our `LLM` interface.
---
### njhill @ 2025-01-31T19:34:47Z

@WoosukKwon you mean using `LLM.generate`? This should fast-fail I think because all sequences will first go through this code before the engine starts I think? (`LLM._validate_and_add_requests` method)

Another comment though, I think for the input validation class of errors we should avoid logging the whole stacktrace, should be a single line.
---
### WoosukKwon @ 2025-02-01T02:35:12Z

@njhill @robertgshaw2-redhat I merged the PR since it's better than what we have right now.

> Another comment though, I think for the input validation class of errors we should avoid logging the whole stacktrace, should be a single line.

Agreed. Can we have a followup PR on this?

> This should fast-fail I think because all sequences will first go through this code before the engine starts I think? (LLM._validate_and_add_requests method)

The case I'm worried about is when the user has a giant list of prompts, and the `LLM.generate` aborts in the middle of the execution (since we currently parallelize the input preprocessing and engine run). I think this is bad in terms of UX.

Also, I'm a bit worried about the backward compatibility, since in V0 we didn't raise an error for this case.
---