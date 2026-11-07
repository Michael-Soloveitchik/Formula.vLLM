# PR #10980: [V1] V1 engine implements parallel sampling (AsyncLLM and LLMEngine) (#10980)

# Issue

## Original PR description (captured source)

This PR adds support for parallel sampling to v1 AsyncLLM and LLMEngine.

Parallel sampling is implemented outside the engine. This does not impact the vLLM v0 parallel sampling implementation. 

Design doc: https://docs.google.com/document/d/1_fvbHVCfexkPAj2Vx0Q0gNvE-b53WwtrLrD6NldgBFU/edit?usp=sharing (message me if you need access permissions)

A request with `n>1` will spawn `n` requests with `n=1` and aggregate their outputs in accordance with the `output_kind`. ~~If prefix caching is enabled, an initial warmup request with `max_tokens=1` will be sent to the engine to fill the prefix cache.~~ **Update:** The vLLM v1 engine can exploit APC when a prompt repeats within a batch, even if that prompt was not seen in a previous batch. Therefore, no warmup request is required.

The abstractions are cleanest for async v1 engine because AsyncLLM presents a `generate()` method that handles adding requests and running the engine; parallel sampling is implemented by writing a wrapper which invokes this method `n` times against parallel sampling child requests.

v1 LLMEngine presents `add_request()` and `step()` methods, so parallel sampling is implemented by writing an `add_request()` wrapper which branches parallel sampling requests into `n` `add_request()` calls for child requests, and then having `step()` aggregate child request outputs into a parent request output.

FIX #13419 
FIX #13420

## Full discussion

See [10980.md](./10980.md) and [10980.json](./10980.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/10980
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/10980.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `12d020838e505db4eaabdfa678784ff9fb94b2e08c5a27d914b3d54d0a81950b`

Existing semantic notes in [10980.sem.json](./10980.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/v1/engine/test_llm_engine.py`
- `tests/v1/entrypoints/openai/test_completion.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/llm_engine.py`
- `vllm/v1/engine/parallel_sampling.py`

### Diff

```diff
diff --git a/tests/v1/engine/test_llm_engine.py b/tests/v1/engine/test_llm_engine.py
index 84b634316cb4..de2a39ee9c08 100644
--- a/tests/v1/engine/test_llm_engine.py
+++ b/tests/v1/engine/test_llm_engine.py
@@ -1,21 +1,114 @@
 # SPDX-License-Identifier: Apache-2.0
 
+import random
+from typing import Dict, List, Optional, Tuple
+
 import pytest
 
 from tests.v1.engine.utils import PLP_APC_UNSUPPORTED_MSG
 from vllm import LLM, SamplingParams
 
+MODEL = "facebook/opt-125m"
+DTYPE = "half"
 
-def test_llm_engine_refuses_prompt_logprobs_with_apc(monkeypatch):
-    """Test passes if LLMEngine raises an exception when it is configured
-    for automatic prefix caching and it receives a request with
-    prompt_logprobs enabled, which is incompatible."""
 
+def _vllm_model(apc: bool, vllm_runner, monkeypatch):
+    """Set up VllmRunner instance."""
     monkeypatch.setenv("VLLM_USE_V1", "1")
     # TODO(nick): Single-proc to work around a ZMQ shutdown hang for now.
     monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
+    return vllm_runner(
+        MODEL,
+        dtype=DTYPE,
+        max_model_len=128,
+        enforce_eager=True,
+        enable_prefix_caching=apc,
+        gpu_memory_utilization=0.5,
+    )
+
+
+@pytest.fixture(
+    # Function scope decouples tests & allows
+    # env var adjustment via monkeypatch
+    scope="function",
+    # Prefix caching
+    params=[False, True])
+def vllm_model(vllm_runner, request, monkeypatch):
+    """VllmRunner test fixture parameterized by APC True/False."""
+    with _vllm_model(request.param, vllm_runner, monkeypatch) as vllm_model:
+        yield vllm_model
+
+
+@pytest.fixture(scope="function")
+def vllm_model_apc(vllm_runner, monkeypatch):
+    """VllmRunner test fixture with APC."""
+    with _vllm_model(True, vllm_runner, monkeypatch) as vllm_model:
+        yield vllm_model
+
+
+def _get_test_sampling_params(
+    prompt_list: List[str],
+    seed: Optional[int] = 42,
+) -> Tuple[List[SamplingParams], List[int]]:
+    """Generate random sampling params for a batch."""
+
+    def get_mostly_n_gt1() -> int:
+        """Mostly n \in [2,20], ~1/3 n=1"""
+        x = random.randint(0, 28)
+        if x < 10:
+            return 1
+        else:
+            return x - 8
+
+    n_list = [get_mostly_n_gt1() for _ in range(len(prompt_list))]
+    # High temperature to maximize the chance of unique completions
+    return [
+        SamplingParams(temperature=0.95, top_p=0.95, n=n, seed=seed)
+        for n in n_list
+    ], n_list
+
+
+def test_parallel_sampling(vllm_model, example_prompts) -> None:
+    """Test passes if parallel sampling `n>1` yields `n` unique completions.
+    
+    Args:
+      vllm_model: VllmRunner instance under test.
+      example_prompt: test fixture providing prompts for testing.
+    """
+    sampling_params_list, n_list = _get_test_sampling_params(example_prompts)
+    model: LLM = vllm_model.model
+    outputs = model.generate(example_prompts, sampling_params_list)
+
+    # Validate each request response
+    for out, n in zip(outputs, n_list):
+        completion_counts: Dict[str, int] = {}
+        # Assert correct number of completions
+        assert len(out.outputs) == n, (
+            f"{len(out.outputs)} completions; {n} expected.")
+        for idx in range(n):
+            comp = out.outputs[idx]
+            # Assert correct completion indices
+            assert comp.index == idx, (f"Index {comp.index}; expected {idx}.")
+            text = comp.text
+            completion_counts[text] = completion_counts.get(text, 0) + 1
+        # Assert unique completions
+        if len(completion_counts) != n:
+            repeats = {
+                txt: num
+                for (txt, num) in completion_counts.items() if num > 1
+            }
+            raise AssertionError(
+                f"{len(completion_counts)} unique completions; expected"
+                f" {n}. Repeats: {repeats}")
+
+
+def test_llm_engine_refuses_prompt_logprobs_with_apc(vllm_model_apc):
+    """Test passes if LLMEngine raises an exception when it is configured
+    for automatic prefix caching and it receives a request with
+    prompt_logprobs enabled, which is incompatible."""
+    model: LLM = vllm_model_apc.model
     with pytest.raises(ValueError) as excinfo:
-        LLM(model="facebook/opt-125m", enable_prefix_caching=True).generate(
+        model.generate(
             "Hello, my name is",
             SamplingParams(temperature=0.8, top_p=0.95, prompt_logprobs=5))
 
diff --git a/tests/v1/entrypoints/openai/test_completion.py b/tests/v1/entrypoints/openai/test_completion.py
index ef46a16ef344..35e059ccb548 100644
--- a/tests/v1/entrypoints/openai/test_completion.py
+++ b/tests/v1/entrypoints/openai/test_completion.py
@@ -250,6 +250,108 @@ async def test_completion_streaming(client: openai.AsyncOpenAI,
     assert "".join(chunks) == single_output
 
 
+@pytest.mark.asyncio
+@pytest.mark.parametrize(
+    "model_name",
+    [MODEL_NAME],
+)
+async def test_parallel_no_streaming(client: openai.AsyncOpenAI,
+                                     model_name: str):
+    """Parallel sampling without streaming.
+    A single request output contains a list of completions.
+    """
+
+    prompt = "What is an LLM?"
+    n = 3
+    max_tokens = 5
+
+    # High temperature to maximize chance of unique completions.
+    completion = await client.completions.create(model=model_name,
+                                                 prompt=prompt,
+                                                 max_tokens=max_tokens,
+                                                 n=n,
+                                                 temperature=0.95,
+                                                 stream=False,
+                                                 seed=42)
+
+    # Assert `n` completions
+    num_completions = len(completion.choices)
+    assert num_completions == n, (
+        f"Num completions {num_completions} but expected {n}.")
+    completion_repeats: Dict[str, int] = {}
+    for idx, choice in enumerate(completion.choices):
+        # Assert correct completion index & some finish reason.
+        assert choice.index == idx, (
+            f"Index {choice.index} but expected {idx}.")
+        assert choice.finish_reason is not None, (
+            "None finish_reason is invalid.")
+        text = choice.text
+        completion_repeats[text] = completion_repeats.get(text, 0) + 1
+    # Assert `n` unique completions
+    num_unique = len(completion_repeats)
+    if num_unique != n:
+        repeats = {
+            txt: num
+            for (txt, num) in completion_repeats.items() if num > 1
+        }
+        raise AssertionError(
+            f"Expected {n} unique completions, got {num_unique};"
+            f" repeats: {repeats}.")
+
+
+@pytest.mark.asyncio
+@pytest.mark.parametrize(
+    "model_name",
+    [MODEL_NAME],
+)
+async def test_parallel_streaming(client: openai.AsyncOpenAI, model_name: str):
+    """Streaming for parallel sampling.
+    The tokens from multiple samples, are flattened into a single stream,
+    with an index to indicate which sample the token belongs to.
+    """
+
+    prompt = "What is an LLM?"
+    n = 3
+    max_tokens = 5
+
+    stream = await client.completions.create(model=model_name,
+                                             prompt=prompt,
+                                             max_tokens=max_tokens,
+                                             n=n,
+                                             temperature=0.95,
+                                             stream=True,
+                                             seed=42)
+    chunks: List[List[str]] = [[] for i in range(n)]
+    finish_reason_count = 0
+    async for chunk in stream:
+        index = chunk.choices[0].index
+        text = chunk.choices[0].text
+        chunks[index].append(text)
+        if chunk.choices[0].finish_reason is not None:
+            finish_reason_count += 1
+    # Assert `n` completions with correct finish reasons
+    assert finish_reason_count == n, (
+        f"Expected {n} completions with valid indices and finish_reason.")
+    completion_repeats: Dict[str, int] = {}
+    for chunk in chunks:
+        chunk_len = len(chunk)
+        # Assert correct number of completion tokens
+        assert chunk_len == max_tokens, (
+            f"max_tokens={max_tokens} but chunk len is {chunk_len}.")
+        text = "".join(chunk)
+        completion_repeats[text] = completion_repeats.get(text, 0) + 1
+        print(text)
+    # Assert `n` unique completions
+    num_unique = len(completion_repeats)
+    if num_unique != n:
+        repeats = {
+            txt: num
+            for (txt, num) in completion_repeats.items() if num > 1
+        }
+        raise AssertionError(f"{num_unique} unique completions, expected {n};"
+                             f" repeats: {repeats}")
+
+
 @pytest.mark.asyncio
 @pytest.mark.parametrize(
     "model_name",
diff --git a/vllm/v1/engine/async_llm.py b/vllm/v1/engine/async_llm.py
index 670454c283da..36a02628f405 100644
--- a/vllm/v1/engine/async_llm.py
+++ b/vllm/v1/engine/async_llm.py
@@ -24,6 +24,7 @@
 from vllm.utils import cdiv, kill_process_tree
 from vllm.v1.engine.core_client import EngineCoreClient
 from vllm.v1.engine.output_processor import OutputProcessor
+from vllm.v1.engine.parallel_sampling import generate_parallel_sampling_async
 from vllm.v1.engine.processor import Processor
 from vllm.v1.executor.abstract import Executor
 from vllm.v1.metrics.loggers import (LoggingStatLogger, PrometheusStatLogger,
@@ -170,7 +171,7 @@ async def add_request(
     # requests we don't need to send multiple messages to core proc,
     # and so we don't need multiple streams which then get
     # re-multiplexed in the API server anyhow.
-    async def generate(
+    async def _generate(
         self,
         prompt: PromptType,
         sampling_params: SamplingParams,
@@ -241,6 +242,30 @@ async def generate(
             await self.abort(request_id)
             raise
 
+    def generate(
+        self,
+        prompt: PromptType,
+        sampling_params: SamplingParams,
+        request_id: str,
+        lora_request: Optional[LoRARequest] = None,
+        trace_headers: Optional[Mapping[str, str]] = None,
+        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
+        priority: int = 0,
+    ) -> AsyncGenerator[RequestOutput, None]:
+        kwargs = dict(prompt=prompt,
+                      sampling_params=sampling_params,
+                      request_id=request_id,
+                      lora_request=lora_request,
+                      trace_headers=trace_headers,
+                      prompt_adapter_request=prompt_adapter_request,
+                      priority=priority)
+        if sampling_params.n is None or sampling_params.n == 1:
+            return self._generate(**kwargs)
+        else:
+            # Special handling for parallel sampling requests
+            return generate_parallel_sampling_async(generate=self._generate,
+                                                    **kwargs)
+
     async def _run_output_handler(self):
         """Background loop: pulls from EngineCore and pushes to AsyncStreams."""
 
diff --git a/vllm/v1/engine/llm_engine.py b/vllm/v1/engine/llm_engine.py
index 33b1ddc0f6fe..64fd8719c82e 100644
--- a/vllm/v1/engine/llm_engine.py
+++ b/vllm/v1/engine/llm_engine.py
@@ -21,6 +21,7 @@
 from vllm.usage.usage_lib import UsageContext
 from vllm.v1.engine.core_client import EngineCoreClient
 from vllm.v1.engine.output_processor import OutputProcessor
+from vllm.v1.engine.parallel_sampling import SyncParallelSamplingManager
 from vllm.v1.engine.processor import Processor
 from vllm.v1.executor.abstract import Executor
 
@@ -48,6 +49,9 @@ def __init__(
         self.model_config = vllm_config.model_config
         self.cache_config = vllm_config.cache_config
 
+        # Bookkeeping for parallel sampling requests
+        self.parallel_manager = SyncParallelSamplingManager()
+
         # important: init dp group before init the engine_core
         self.parallel_config = vllm_config.parallel_config
         self.dp_enabled = self.parallel_config.data_parallel_size > 1  # noqa
@@ -115,7 +119,8 @@ def from_engine_args(
                    multiprocess_mode=enable_multiprocessing)
 
     def get_num_unfinished_requests(self) -> int:
-        return self.output_processor.get_num_unfinished_requests()
+        return self.parallel_manager.get_num_unfinished_requests(
+            self.output_processor.get_num_unfinished_requests())
 
     def has_unfinished_requests(self) -> bool:
         has_unfinished = self.output_processor.has_unfinished_requests()
@@ -151,7 +156,36 @@ def add_request(
         prompt_adapter_request: Optional[PromptAdapterRequest] = None,
         priority: int = 0,
     ) -> None:
-
+        """Add request."""
+        kwargs = dict(request_id=request_id,
+                      prompt=prompt,
+                      params=params,
+                      arrival_time=arrival_time,
+                      lora_request=lora_request,
+                      trace_headers=trace_headers,
+                      prompt_adapter_request=prompt_adapter_request,
+                      priority=priority)
+        # Handle parallel sampling requests differently.
+        if params is None or isinstance(params,
+                                        PoolingParams) or params.n == 1:
+            self._add_request(**kwargs)
+        else:
+            # Special handling for parallel sampling requests
+            self.parallel_manager.add_request_parallel_sampling(
+                add_request=self._add_request, **kwargs)
+
+    def _add_request(
+        self,
+        request_id: str,
+        prompt: PromptType,
+        params: Union[SamplingParams, PoolingParams],
+        arrival_time: Optional[float] = None,
+        lora_request: Optional[LoRARequest] = None,
+        trace_headers: Optional[Mapping[str, str]] = None,
+        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
+        priority: int = 0,
+    ) -> None:
+        """Add request, `n=1`"""
         # 1) Process raw inputs into the request.
         request = self.processor.process_inputs(request_id, prompt, params,
                                                 arrival_time, lora_request,
@@ -182,7 +216,10 @@ def step(self) -> List[RequestOutput]:
         # 3) Abort any reqs that finished due to stop strings.
         self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
 
-        return processed_outputs.request_outputs
+        request_outputs = processed_outputs.request_outputs
+
+        # 4) Process unfinished parallel sampling requests
+        return self.parallel_manager.step(request_outputs)
 
     def get_model_config(self):
         return self.model_config
diff --git a/vllm/v1/engine/parallel_sampling.py b/vllm/v1/engine/parallel_sampling.py
new file mode 100644
index 000000000000..5d4ea111abfc
--- /dev/null
+++ b/vllm/v1/engine/parallel_sampling.py
@@ -0,0 +1,375 @@
+# SPDX-License-Identifier: Apache-2.0
+
+from copy import copy
+from typing import (AsyncGenerator, Dict, List, Mapping, Optional, Protocol,
+                    Tuple, Union)
+
+from vllm.inputs import PromptType
+from vllm.lora.request import LoRARequest
+from vllm.outputs import RequestOutput
+from vllm.pooling_params import PoolingParams
+from vllm.prompt_adapter.request import PromptAdapterRequest
+from vllm.sampling_params import RequestOutputKind, SamplingParams
+from vllm.utils import merge_async_iterators
+
+
+class AsyncGenerateMethodType(Protocol):
+
+    def __call__(self,
+                 prompt: PromptType,
+                 sampling_params: SamplingParams,
+                 request_id: str,
+                 lora_request: Optional[LoRARequest] = None,
+                 trace_headers: Optional[Mapping[str, str]] = None,
+                 prompt_adapter_request: Optional[PromptAdapterRequest] = None,
+                 priority: int = 0) -> AsyncGenerator[RequestOutput, None]:
+        ...
+
+
+class SyncAddRequestMethodType(Protocol):
+
+    def __call__(self,
+                 request_id: str,
+                 prompt: PromptType,
+                 params: Union[SamplingParams, PoolingParams],
+                 arrival_time: Optional[float] = None,
+                 lora_request: Optional[LoRARequest] = None,
+                 trace_headers: Optional[Mapping[str, str]] = None,
+                 prompt_adapter_request: Optional[PromptAdapterRequest] = None,
+                 priority: int = 0) -> None:
+        ...
+
+
+class ParallelSamplingRequest:
+    """Info, state & processing for parallel sampling request.
+    
+    Store parent request ID and sampling params.
+    Facilitate generating child request sampling params.
+    Transform child request outputs into parent request
+    outputs.
+    When stream mode is disabled, then `self.request_output`
+    aggregates child request completions.
+    """
+
+    request_id: str
+    sampling_params: SamplingParams
+    cached_child_sampling_params: Optional[SamplingParams]
+    request_output: Optional[RequestOutput]
+    num_finished_completions: int
+
+    def __init__(self, request_id: str,
+                 sampling_params: SamplingParams) -> None:
+        self.request_id = request_id
+        self.sampling_params = sampling_params
+        self.cached_child_sampling_params = None
+        self.request_output = None
+        self.num_finished_completions = 0
+
+    def _get_child_sampling_params(
+        self,
+        index: int,
+    ) -> SamplingParams:
+        """Efficiently obtain child `sampling_params`
+
+        If `sampling_params.seed` is not `None` then 
+        each child request requires a unique clone of
+        parent `sampling_params` with a unique seed.
+
+        Args:
+          index: index within `n` child requests
+
+        Returns:
+          Child `sampling_params` instance.
+        """
+        seed = self.sampling_params.seed
+        if self.cached_child_sampling_params:
+            # Reuse child sampling_params data structure
+            return self.cached_child_sampling_params
+        # Build child sampling_params
+        child_sampling_params = copy(self.sampling_params)
+        child_sampling_params.n = 1
+        if seed is None:
+            # Cache child sampling_params for later reuse
+            self.cached_child_sampling_params = child_sampling_params
+        else:
+            # Each child gets a clone with a unique seed
+            child_sampling_params.seed = seed + index
+        return child_sampling_params
+
+    def _add_output(
+        self,
+        child_req_output: RequestOutput,
+        index: int,
+    ) -> None:
+        """Aggregate a parallel sampling child
+        request output.
+        
+        Non-stream-mode (`output_kind == FINAL_ONLY`) 
+        only. Inject correct parent request ID and
+        completion index.
+
+        Args:
+          child_req_output: a single request output
+                            from a parallel sampling
+                            child request.   
+          index: index within `n` child    
+        """
+        self.num_finished_completions += 1
+        new_completion = child_req_output.outputs[0]
+        new_completion.index = index
+        if self.request_output is None:
+            # Save the first request output; reinstate
+            # original request ID; metrics are not
+            # supported for parallel sampling
+            child_req_output.request_id = self.request_id
+            child_req_output.metrics = None
+            self.request_output = child_req_output
+        else:
+            # Aggregate additional completion into request output
+            # Note: will be sorted by index later
+            self.request_output.outputs.append(new_completion)
+
+    def _get_final_request_output(self) -> RequestOutput:
+        """Invariant: parent completion outputs sorted by index"""
+        assert self.request_output is not None
+        self.request_output.finished = True
+        self.request_output.outputs = sorted(self.request_output.outputs,
+                                             key=lambda x: x.index)
+        return self.request_output
+
+    def get_child_info(self, index: int) -> Tuple[str, SamplingParams]:
+        """Get child request ID and sampling params.
+        
+        Args:
+          index: index within `n` child requests.
+        
+        Returns:
+          (request ID, sampling_params) tuple
+        """
+        return (f"{index}_{self.request_id}",
+                self._get_child_sampling_params(index))
+
+    def process_output(
+        self,
+        child_req_output: RequestOutput,
+        index: int,
+    ) -> Optional[RequestOutput]:
+        """Filter, aggregate and transform parallel sampling
+        child request outputs.
+
+        If the parent request has `stream=false`
+        (`output_kind == FINAL_ONLY`), each child will also have
+        `output_kind == FINAL_ONLY`. All child request outputs
+        must be aggregated into a single request output, with
+        multiple completions. This request output is only returned
+        once `n` completions are aggregated.
+
+        If the parent request has `stream=true`
+        (`output_kind == DELTA`), each child will also have
+        `output_kind == DELTA`. All child request outputs
+        must be streamed directly to the caller.
+
+        Args:
+          child_req_output: a single child request output
+          index: index within `n` child requests
+
+        Returns:
+          `None`, unless a processed request output is ready to
+          send back to the caller.
+        """
+        if self.output_kind != RequestOutputKind.FINAL_ONLY:
+            # stream=true: return child completions immediately
+            child_req_output.request_id = self.request_id
+            child_req_output.outputs[0].index = index
+            if child_req_output.finished:
+                # Parent request is complete if all child requests are
+                # complete.
+                self.num_finished_completions += 1
+                child_req_output.finished = (
+                    self.num_finished_completions == self.n)
+            return child_req_output
+
+        # stream=false: aggregate child completions
+        self._add_output(child_req_output, index)
+        if self.num_finished_completions == self.n:
+            # Return aggregated request output after obtaining
+            # all completions
+            return self._get_final_request_output()
+        return None
+
+    async def wrap_child_async_generator(
+        self,
+        child_gen: AsyncGenerator[RequestOutput, None],
+        index: int,
+    ) -> AsyncGenerator[RequestOutput, None]:
+        """Output generator for a single parallel sampling
+        child request.
+
+        Each parallel sampling request triggers at
+        least two child requests. This generator
+        yields zero or more request outputs to
+        return to the caller, as they become
+        available.
+
+        Args:
+          child_gen: generator for child request
+                     outputs.
+          index: index within the `n` child requests
+
+        Returns:
+          Yields zero or more request outputs to return
+          to the caller.
+        """
+        async for out in child_gen:
+            if req_out := self.process_output(out, index):
+                yield req_out
+
+    @property
+    def n(self) -> int:
+        return self.sampling_params.n
+
+    @property
+    def output_kind(self) -> RequestOutputKind:
+        return self.sampling_params.output_kind
+
+
+class SyncParallelSamplingManager:
+
+    def __init__(self):
+        # Parent req ID -> parent request manager
+        self.parent_reqs: Dict[str, ParallelSamplingRequest] = {}
+        # Child req ID -> (child req index, parent req ID)
+        self.child_reqs: Dict[str, Tuple[int, str]] = {}
+
+    def _register_parent_request(self, req: ParallelSamplingRequest) -> None:
+        """Register parallel sampling parent request."""
+        self.parent_reqs[req.request_id] = req
+
+    def _register_child_request(self, req_id: str, child_req_id: str,
+                                index: int) -> None:
+        """Register parallel sampling child request with parent.
+        
+        Args:
+          req_id: parent request ID
+          child_req_id: child request ID
+          index: child request index within `n` child requests
+        """
+        self.child_reqs[child_req_id] = (index, req_id)
+
+    def get_num_unfinished_requests(self, num_core_reqs: int) -> int:
+        """Get the number of unfinished requests, correcting for parallel
+           sampling.
+        
+        Args:
+          num_core_reqs: The number of unfinished requests in the engine core.
+        
+        Returns:
+          Number of unfinished requests, where each parallel sampling req 
+          counts as 1
+        """
+        return num_core_reqs + len(self.parent_reqs) - len(self.child_reqs)
+
+    def add_request_parallel_sampling(
+        self,
+        add_request: SyncAddRequestMethodType,
+        request_id: str,
+        prompt: PromptType,
+        params: Union[SamplingParams, PoolingParams],
+        arrival_time: Optional[float] = None,
+        lora_request: Optional[LoRARequest] = None,
+        trace_headers: Optional[Mapping[str, str]] = None,
+        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
+        priority: int = 0,
+    ) -> None:
+        """Add sync parallel sampling request."""
+        req = ParallelSamplingRequest(request_id, params)
+        self._register_parent_request(req)
+        # Add n child requests with unique request IDs & random seeds and n=1
+        for idx in range(req.n):
+            child_req_id, child_params = req.get_child_info(idx)
+            self._register_child_request(request_id, child_req_id, idx)
+            add_request(request_id=child_req_id,
+                        prompt=prompt,
+                        params=child_params,
+                        arrival_time=arrival_time,
+                        lora_request=lora_request,
+                        trace_headers=trace_headers,
+                        prompt_adapter_request=prompt_adapter_request,
+                        priority=priority)  # type: ignore
+
+    def step(
+        self,
+        outputs: List[RequestOutput],
+    ) -> List[RequestOutput]:
+        """Build parallel sampling request outputs.
+        
+        Extract child request outputs, aggregate them
+        into parent request output, and return parent
+        output when complete.
+
+        Do not modify `n=1` requests.
+
+        Args:
+          outputs: step request outputs. Mix of child request
+                   outputs & `n=1` request outputs.
+
+        Return:
+          List of parallel sampling parent request outputs &
+          unmodified `n=1` request outputs passed-thru from input.
+        """
+        if not (self.parent_reqs and outputs):
+            # Return unmodified
+            return outputs
+        agg_outputs = []
+        for output in outputs:
+            req_id = output.request_id
+            if child_req_entry := self.child_reqs.get(req_id, None):
+                # For each parallel sampling child request output:
+                (index, parent_req_id) = child_req_entry
+                req = self.parent_reqs[parent_req_id]
+                # Update parallel sampling request
+                if out := req.process_output(output, index):
+                    # Return parent request output if complete;
+                    # cleanup parent request bookkeeping.
+                    agg_outputs.append(out)
+                    del self.parent_reqs[parent_req_id]
+                # Cleanup child request bookkeeping.
+                del self.child_reqs[req_id]
+            else:
+                # Not a parallel sampling request output
+                agg_outputs.append(output)
+        return agg_outputs
+
+
+async def generate_parallel_sampling_async(
+    generate: AsyncGenerateMethodType,
+    prompt: PromptType,
+    sampling_params: SamplingParams,
+    request_id: str,
+    lora_request: Optional[LoRARequest] = None,
+    trace_headers: Optional[Mapping[str, str]] = None,
+    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
+    priority: int = 0,
+) -> AsyncGenerator[RequestOutput, None]:
+    """Generate completions for async parallel sampling requests."""
+    parent_req = ParallelSamplingRequest(request_id, sampling_params)
+
+    # Aggregate generators for n child requests
+    gens: List[AsyncGenerator[RequestOutput, None]] = []
+    for idx in range(parent_req.n):
+        child_req_id, child_params = parent_req.get_child_info(idx)
+        child_gen = generate(
+            prompt=prompt,
+            sampling_params=child_params,
+            request_id=child_req_id,
+            lora_request=lora_request,
+            trace_headers=trace_headers,
+            prompt_adapter_request=prompt_adapter_request,
+            priority=priority,
+        )  # type: ignore
+        gen = parent_req.wrap_child_async_generator(child_gen, idx)
+        gens.append(gen)
+
+    # Merge generators
+    async for _, out in merge_async_iterators(*gens):
+        yield out
```
