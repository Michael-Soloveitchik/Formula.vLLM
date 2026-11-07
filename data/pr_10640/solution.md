# PR #10640: [V1] VLM - Run the mm_mapper preprocessor in the frontend process

- field: scheduling  | area: v1_engine
- author: alexm-redhat
- state: MERGED  created: 2024-11-25T15:55:50Z  merged: 2024-12-03T10:33:11Z
- url: https://github.com/vllm-project/vllm/pull/10640
- labels: frontend, ready

# Issue

## Problem

In vLLM V1, the multi-modal input mapper (HuggingFace preprocessor that converts raw PIL images to model-ready tensors) ran inside the engine core process via `EngineCoreCore.add_request`. Because the engine core is a single busy process driving the scheduler and model execution, doing 10-50 ms of CPU-bound image preprocessing there serialized against the hot scheduling loop and blocked request throughput. The raw `mm_data` plus `mm_processor_kwargs` were shipped across the process boundary and mapped late, so preprocessing could not overlap with core work.

## Example

Running 512 prompts with 64 output tokens on LLaVA (`offline_inference_vision_language.py -m llava --num-prompts 512 --modality image`): generate() took 28.91 s with mapping in the core vs 16.84 s with mapping moved to the frontend — ~1.7x.

## Use case

Offline/online multi-modal serving (e.g. LLaVA vision-language inference) with V1 multiprocessing enabled, where each request carries image inputs.

## Resources

CPU time for the HF image preprocessor (PIL -> tensor), plus the engine-core kernel-launch/scheduling path it was blocking; also inter-process serialization bandwidth for raw image data.

## Type of users

Users serving multi-modal / vision-language models (VLMs) under V1 with `VLLM_ENABLE_V1_MULTIPROCESSING=1`.

## Kind of inputs

Requests containing multi-modal data (images), especially large batches (512 prompts) of image prompts where per-request preprocessing cost accumulates.

## Mechanism with the issue

`EngineCore.add_request` called `self.mm_input_mapper.process_inputs(req.mm_data, req.mm_processor_kwargs)` inline before scheduling, and `Request` carried raw `mm_data`/`mm_processor_kwargs`. The FIXME in core.py notes mapping takes 10-50 ms and spikes latency; running it in the single-threaded core process stalled the scheduler loop.

# Solution

## Fix in words

The mapper call is deleted from `EngineCore.add_request` and moved into the frontend `Processor.process_inputs`, which now computes `mm_inputs = self.mm_input_mapper.process_inputs(...)` (only when multi_modal_data is non-empty) before building the request. `EngineCoreRequest` is changed to carry preprocessed `mm_inputs: Optional[List[MultiModalKwargs]]` instead of raw `mm_data` + `mm_processor_kwargs`. `TokenInputs`/`token_inputs` gain a `multi_modal_inputs` field with a `multi_modal_inputs` cached_property, and `Request` now stores the already-mapped `mm_inputs` directly (with `has_encoder_inputs` checking `len(self.mm_inputs)` instead of `mm_data`). Effect: image preprocessing runs in the frontend process, overlapping with and no longer blocking the engine core scheduling loop, yielding ~1.7x throughput on the LLaVA benchmark.

## Author's solution

Author adds support to run the multi-modal mapper/preprocessor (from HuggingFace) in the frontend process rather than the engine core. Measured on 512 prompts / 64 output tokens with LLaVA: generate() time dropped from 28.91 s to 16.84 s, a 1.7x improvement.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/v1/engine/test_engine_core.py`
- `tests/v1/engine/test_engine_core_client.py`
- `vllm/inputs/data.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/processor.py`
- `vllm/v1/request.py`

### Diff
```diff
diff --git a/tests/v1/engine/test_engine_core.py b/tests/v1/engine/test_engine_core.py
index bd11ff187706..fef44ac29c41 100644
--- a/tests/v1/engine/test_engine_core.py
+++ b/tests/v1/engine/test_engine_core.py
@@ -27,9 +27,8 @@ def make_request() -> EngineCoreRequest:
         request_id=uuid.uuid4(),
         prompt=PROMPT,
         prompt_token_ids=PROMPT_TOKENS,
-        mm_data=None,
+        mm_inputs=None,
         mm_placeholders=None,
-        mm_processor_kwargs=None,
         sampling_params=SamplingParams(),
         eos_token_id=None,
         arrival_time=time.time(),
diff --git a/tests/v1/engine/test_engine_core_client.py b/tests/v1/engine/test_engine_core_client.py
index 582192196aaf..4e003a25e91d 100644
--- a/tests/v1/engine/test_engine_core_client.py
+++ b/tests/v1/engine/test_engine_core_client.py
@@ -29,9 +29,8 @@ def make_request(params: SamplingParams) -> EngineCoreRequest:
         request_id=str(uuid.uuid4()),
         prompt=PROMPT,
         prompt_token_ids=PROMPT_TOKENS,
-        mm_data=None,
+        mm_inputs=None,
         mm_placeholders=None,
-        mm_processor_kwargs=None,
         sampling_params=params,
         eos_token_id=None,
         arrival_time=time.time(),
diff --git a/vllm/inputs/data.py b/vllm/inputs/data.py
index fb7dbbebd7b9..554d96c0989d 100644
--- a/vllm/inputs/data.py
+++ b/vllm/inputs/data.py
@@ -7,7 +7,8 @@
 from typing_extensions import NotRequired, TypedDict, TypeVar, assert_never
 
 if TYPE_CHECKING:
-    from vllm.multimodal import MultiModalDataDict, MultiModalPlaceholderDict
+    from vllm.multimodal import (MultiModalDataDict, MultiModalKwargs,
+                                 MultiModalPlaceholderDict)
     from vllm.multimodal.inputs import MultiModalInputsV2
 
 
@@ -150,6 +151,12 @@ class TokenInputs(TypedDict):
     if the model supports it.
     """
 
+    multi_modal_inputs: NotRequired["MultiModalKwargs"]
+    """
+    Optional multi-modal inputs to pass to the model,
+    if the model supports it.
+    """
+
     multi_modal_placeholders: NotRequired["MultiModalPlaceholderDict"]
     """
     Placeholder ranges for the multi-modal data.
@@ -169,6 +176,7 @@ def token_inputs(
     token_type_ids: Optional[List[int]] = None,
     prompt: Optional[str] = None,
     multi_modal_data: Optional["MultiModalDataDict"] = None,
+    multi_modal_inputs: Optional["MultiModalKwargs"] = None,
     multi_modal_placeholders: Optional["MultiModalPlaceholderDict"] = None,
     mm_processor_kwargs: Optional[Dict[str, Any]] = None,
 ) -> TokenInputs:
@@ -181,6 +189,8 @@ def token_inputs(
         inputs["token_type_ids"] = token_type_ids
     if multi_modal_data is not None:
         inputs["multi_modal_data"] = multi_modal_data
+    if multi_modal_inputs is not None:
+        inputs["multi_modal_inputs"] = multi_modal_inputs
     if multi_modal_placeholders is not None:
         inputs["multi_modal_placeholders"] = multi_modal_placeholders
     if mm_processor_kwargs is not None:
@@ -273,6 +283,18 @@ def multi_modal_data(self) -> "MultiModalDataDict":
 
         assert_never(inputs)
 
+    @cached_property
+    def multi_modal_inputs(self) -> Union[Dict, "MultiModalKwargs"]:
+        inputs = self.inputs
+
+        if inputs["type"] == "token":
+            return inputs.get("multi_modal_inputs", {})
+
+        if inputs["type"] == "multimodal":
+            return inputs.get("mm_kwargs", {})
+
+        assert_never(inputs)
+
     @cached_property
     def multi_modal_placeholders(self) -> "MultiModalPlaceholderDict":
         inputs = self.inputs
diff --git a/vllm/v1/engine/__init__.py b/vllm/v1/engine/__init__.py
index 967124fd850e..3cf0e610ae7a 100644
--- a/vllm/v1/engine/__init__.py
+++ b/vllm/v1/engine/__init__.py
@@ -1,11 +1,11 @@
 import enum
 from dataclasses import dataclass
-from typing import Any, Dict, List, Optional, Union
+from typing import List, Optional, Union
 
 import msgspec
 
 from vllm.lora.request import LoRARequest
-from vllm.multimodal import MultiModalDataDict, MultiModalPlaceholderDict
+from vllm.multimodal import MultiModalKwargs, MultiModalPlaceholderDict
 from vllm.sampling_params import RequestOutputKind, SamplingParams
 
 
@@ -35,9 +35,8 @@ class EngineCoreRequest:
     # always be tokenized?
     prompt: Optional[str]
     prompt_token_ids: List[int]
-    mm_data: Optional[MultiModalDataDict]
+    mm_inputs: Optional[List[MultiModalKwargs]]
     mm_placeholders: Optional[MultiModalPlaceholderDict]
-    mm_processor_kwargs: Optional[Dict[str, Any]]
     sampling_params: SamplingParams
     eos_token_id: Optional[int]
     arrival_time: float
diff --git a/vllm/v1/engine/core.py b/vllm/v1/engine/core.py
index 34f99dd30ef2..397a33eed389 100644
--- a/vllm/v1/engine/core.py
+++ b/vllm/v1/engine/core.py
@@ -84,14 +84,7 @@ def _initialize_kv_caches(self,
 
     def add_request(self, request: EngineCoreRequest):
         """Add request to the scheduler."""
-
         req = Request.from_engine_core_request(request)
-        # FIXME(woosuk): The input mapping (e.g., PIL images to tensors) may
-        # take 10-50 ms, which can cause a spike in the latency. We should
-        # consider moving this to a separate thread.
-        if req.mm_data:
-            req.mm_inputs = self.mm_input_mapper.process_inputs(
-                req.mm_data, req.mm_processor_kwargs)
         self.scheduler.add_request(req)
 
     def abort_requests(self, request_ids: List[str]):
diff --git a/vllm/v1/engine/processor.py b/vllm/v1/engine/processor.py
index 5c1577190c75..7a1ea2530abd 100644
--- a/vllm/v1/engine/processor.py
+++ b/vllm/v1/engine/processor.py
@@ -14,6 +14,7 @@
 from vllm.transformers_utils.config import try_get_generation_config
 from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup
 from vllm.v1.engine import DetokenizerRequest, EngineCoreRequest
+from vllm.v1.engine.mm_input_mapper import MMInputMapper
 
 
 class Processor:
@@ -39,6 +40,9 @@ def __init__(
         self.input_processor = input_registry.create_input_processor(
             model_config)
 
+        # Multi-modal (huggingface) input mapper
+        self.mm_input_mapper = MMInputMapper(model_config)
+
     # TODO: run in an ThreadpoolExecutor or BackgroundProcess.
     # This ideally should releases the GIL, so we should not block the
     # asyncio loop while this is running.
@@ -96,6 +100,12 @@ def process_inputs(
         sampling_params.update_from_generation_config(
             self.generation_config_fields, eos_token_id)
 
+        # Preprocess multi-modal data
+        mm_inputs = self.mm_input_mapper.process_inputs(
+            decoder_inputs.multi_modal_data,
+            decoder_inputs.mm_processor_kwargs) if len(
+                decoder_inputs.multi_modal_data) > 0 else None
+
         # Make Request for Detokenizer.
         detokenizer_request = DetokenizerRequest(
             request_id,
@@ -113,9 +123,8 @@ def process_inputs(
             request_id,
             decoder_inputs.prompt,
             decoder_inputs.prompt_token_ids,
-            decoder_inputs.multi_modal_data,
+            mm_inputs,
             decoder_inputs.multi_modal_placeholders,
-            decoder_inputs.mm_processor_kwargs,
             sampling_params,
             eos_token_id,
             arrival_time,
diff --git a/vllm/v1/request.py b/vllm/v1/request.py
index 51fb4003e5fe..6bc1e4d5c769 100644
--- a/vllm/v1/request.py
+++ b/vllm/v1/request.py
@@ -45,9 +45,6 @@ def __init__(
         self._all_token_ids: List[int] = self.prompt_token_ids.copy()
         self.num_computed_tokens = 0
 
-        # Raw multimodal data before the mm input mapper (e.g., PIL images).
-        self.mm_data = self.inputs.multi_modal_data
-        self.mm_processor_kwargs = self.inputs.mm_processor_kwargs
         mm_positions = self.inputs.multi_modal_placeholders
         if mm_positions:
             # FIXME(woosuk): Support other modalities.
@@ -55,7 +52,10 @@ def __init__(
         else:
             self.mm_positions = []
         # Output of the mm input mapper (e.g., image tensors).
-        self.mm_inputs: List[MultiModalKwargs] = []
+        if self.inputs.multi_modal_inputs:
+            self.mm_inputs = self.inputs.multi_modal_inputs
+        else:
+            self.mm_inputs: List[MultiModalKwargs] = []
 
     @classmethod
     def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
@@ -64,9 +64,10 @@ def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
             inputs=token_inputs(
                 prompt_token_ids=request.prompt_token_ids,
                 prompt=request.prompt,
-                multi_modal_data=request.mm_data,
+                multi_modal_data=None,
+                multi_modal_inputs=request.mm_inputs,
                 multi_modal_placeholders=request.mm_placeholders,
-                mm_processor_kwargs=request.mm_processor_kwargs,
+                mm_processor_kwargs=None,
             ),
             sampling_params=request.sampling_params,
             eos_token_id=request.eos_token_id,
@@ -110,7 +111,7 @@ def get_finished_reason(self) -> Union[str, None]:
         return RequestStatus.get_finished_reason(self.status)
 
     def has_encoder_inputs(self) -> bool:
-        return len(self.mm_data) > 0
+        return len(self.mm_inputs) > 0
 
     @property
     def num_encoder_inputs(self) -> int:

```
