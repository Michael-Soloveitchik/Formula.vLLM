# PR #24219: [Core] Support async scheduling with uniproc executor  (#24219)

# Issue

## Original PR description (captured source)

Follow-on from https://github.com/vllm-project/vllm/pull/23569.

This provides most of the speedup of that PR, like +22% rather than +25%. We could experiment with a slightly more complicated version where the worker runs in a separate thread, but this seems like a good first implementation.

## Full discussion

See [24219.md](./24219.md) and [24219.json](./24219.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/24219
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/24219.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `fd6294053b67ce5bbd4075182a4836859b00b991a7642d2b70cb6369c416a9e6`

Existing semantic notes in [24219.sem.json](./24219.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/basic_correctness/test_basic_correctness.py`
- `tests/v1/engine/test_engine_core.py`
- `vllm/engine/arg_utils.py`
- `vllm/executor/uniproc_executor.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/executor/abstract.py`
- `vllm/v1/executor/multiproc_executor.py`
- `vllm/v1/executor/ray_distributed_executor.py`

### Diff

```diff
diff --git a/tests/basic_correctness/test_basic_correctness.py b/tests/basic_correctness/test_basic_correctness.py
index a3b09cc81791..fba18f197074 100644
--- a/tests/basic_correctness/test_basic_correctness.py
+++ b/tests/basic_correctness/test_basic_correctness.py
@@ -62,6 +62,8 @@ def _fix_prompt_embed_outputs(
 @pytest.mark.parametrize("backend", ["FLASH_ATTN"])
 @pytest.mark.parametrize("max_tokens", [5])
 @pytest.mark.parametrize("enforce_eager", [False])
+@pytest.mark.parametrize("async_scheduling", [True, False])
+@pytest.mark.parametrize("model_executor", ["uni", "mp"])
 @pytest.mark.parametrize("enable_prompt_embeds", [True, False])
 def test_models(
     monkeypatch: pytest.MonkeyPatch,
@@ -70,6 +72,8 @@ def test_models(
     backend: str,
     max_tokens: int,
     enforce_eager: bool,
+    async_scheduling: bool,
+    model_executor: str,
     enable_prompt_embeds: bool,
 ) -> None:
 
@@ -77,6 +81,12 @@ def test_models(
             "VLLM_USE_V1") and envs.VLLM_USE_V1:
         pytest.skip("enable_prompt_embeds is not supported in v1.")
 
+    if not envs.VLLM_USE_V1:
+        if async_scheduling:
+            pytest.skip("async_scheduling only supported in v1.")
+        if model_executor != "uni":
+            pytest.skip("only test uniproc executor for v0.")
+
     if backend == "XFORMERS" and model == "google/gemma-2-2b-it":
         pytest.skip(
             f"{backend} does not support gemma2 with full context length.")
@@ -98,11 +108,15 @@ def test_models(
                     prompt_embeds = hf_model.get_prompt_embeddings(
                         example_prompts)
 
-        with VllmRunner(model,
-                        max_model_len=8192,
-                        enforce_eager=enforce_eager,
-                        enable_prompt_embeds=enable_prompt_embeds,
-                        gpu_memory_utilization=0.7) as vllm_model:
+        with VllmRunner(
+                model,
+                max_model_len=8192,
+                enforce_eager=enforce_eager,
+                enable_prompt_embeds=enable_prompt_embeds,
+                gpu_memory_utilization=0.7,
+                async_scheduling=async_scheduling,
+                distributed_executor_backend=model_executor,
+        ) as vllm_model:
             if enable_prompt_embeds:
                 vllm_outputs = vllm_model.generate_greedy(
                     prompt_embeds, max_tokens)
diff --git a/tests/v1/engine/test_engine_core.py b/tests/v1/engine/test_engine_core.py
index 98265c634957..17b136aa4273 100644
--- a/tests/v1/engine/test_engine_core.py
+++ b/tests/v1/engine/test_engine_core.py
@@ -257,9 +257,13 @@ def initialize_from_config(
         def execute_model(
             self,
             scheduler_output,
+            non_block=False,
         ) -> Future[ModelRunnerOutput]:
             """Make execute_model non-blocking."""
 
+            # DummyExecutor used only for testing async case.
+            assert non_block
+
             def _execute():
                 output = self.collective_rpc("execute_model",
                                              args=(scheduler_output, ))
diff --git a/vllm/engine/arg_utils.py b/vllm/engine/arg_utils.py
index bc7ec0f065db..ab43c0edc98d 100644
--- a/vllm/engine/arg_utils.py
+++ b/vllm/engine/arg_utils.py
@@ -1296,11 +1296,8 @@ def create_engine_config(
             # Async scheduling does not work with the uniprocess backend.
             if self.distributed_executor_backend is None:
                 self.distributed_executor_backend = "mp"
-                logger.info("Using mp-based distributed executor backend "
-                            "for async scheduling.")
-            if self.distributed_executor_backend == "uni":
-                raise ValueError("Async scheduling is not supported with "
-                                 "uni-process backend.")
+                logger.info("Defaulting to mp-based distributed executor "
+                            "backend for async scheduling.")
             if self.pipeline_parallel_size > 1:
                 raise ValueError("Async scheduling is not supported with "
                                  "pipeline-parallel-size > 1.")
diff --git a/vllm/executor/uniproc_executor.py b/vllm/executor/uniproc_executor.py
index 2e456ecd6de0..3b566e88a9ec 100644
--- a/vllm/executor/uniproc_executor.py
+++ b/vllm/executor/uniproc_executor.py
@@ -1,7 +1,8 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-
 import os
+from concurrent.futures import Future, ThreadPoolExecutor
+from functools import cached_property
 from multiprocessing import Lock
 from typing import Any, Callable, Dict, List, Optional, Tuple, Union
 
@@ -17,6 +18,7 @@
                         run_method)
 from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
 from vllm.v1.executor.utils import get_and_update_mm_cache
+from vllm.v1.outputs import AsyncModelRunnerOutput
 from vllm.worker.worker_base import WorkerWrapperBase
 
 logger = init_logger(__name__)
@@ -31,15 +33,7 @@ def _init_executor(self) -> None:
         """
         self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                                rpc_rank=0)
-        distributed_init_method = get_distributed_init_method(
-            get_ip(), get_open_port())
-        local_rank = 0
-        # set local rank as the device index if specified
-        device_info = self.vllm_config.device_config.device.__str__().split(
-            ":")
-        if len(device_info) > 1:
-            local_rank = int(device_info[1])
-        rank = 0
+        distributed_init_method, rank, local_rank = self._distributed_args()
         is_driver_worker = True
         kwargs = dict(
             vllm_config=self.vllm_config,
@@ -50,21 +44,56 @@ def _init_executor(self) -> None:
         )
         self.mm_receiver_cache = worker_receiver_cache_from_config(
             self.vllm_config, MULTIMODAL_REGISTRY, Lock())
+
+        self.async_output_thread: Optional[ThreadPoolExecutor] = None
+        if self.max_concurrent_batches > 1:
+            self.async_output_thread = ThreadPoolExecutor(
+                max_workers=1, thread_name_prefix="WorkerAsyncOutput")
+
         self.collective_rpc("init_worker", args=([kwargs], ))
         self.collective_rpc("init_device")
         self.collective_rpc("load_model")
 
+    def _distributed_args(self) -> tuple[str, int, int]:
+        """Return (distributed_init_method, rank, local_rank)."""
+        distributed_init_method = get_distributed_init_method(
+            get_ip(), get_open_port())
+        # set local rank as the device index if specified
+        device_info = self.vllm_config.device_config.device.__str__().split(
+            ":")
+        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
+        return distributed_init_method, 0, local_rank
+
+    @cached_property
+    def max_concurrent_batches(self) -> int:
+        return 2 if self.scheduler_config.async_scheduling else 1
+
     def collective_rpc(self,
                        method: Union[str, Callable],
                        timeout: Optional[float] = None,
                        args: Tuple = (),
-                       kwargs: Optional[Dict] = None) -> List[Any]:
+                       kwargs: Optional[Dict] = None,
+                       non_block: bool = False) -> List[Any]:
         if kwargs is None:
             kwargs = {}
         if self.mm_receiver_cache is not None and method == "execute_model":
             get_and_update_mm_cache(self.mm_receiver_cache, args)
-        answer = run_method(self.driver_worker, method, args, kwargs)
-        return [answer]
+
+        if not non_block:
+            return [run_method(self.driver_worker, method, args, kwargs)]
+
+        try:
+            result = run_method(self.driver_worker, method, args, kwargs)
+            if isinstance(result, AsyncModelRunnerOutput):
+                if (async_thread := self.async_output_thread) is not None:
+                    return [async_thread.submit(result.get_output)]
+                result = result.get_output()
+            future = Future[Any]()
+            future.set_result(result)
+        except Exception as e:
+            future = Future[Any]()
+            future.set_exception(e)
+        return [future]
 
     def check_health(self) -> None:
         # UniProcExecutor will always be healthy as long as
@@ -116,8 +145,9 @@ def _init_executor(self) -> None:
             assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, \
             ("To get deterministic execution in V1, "
             "please set VLLM_ENABLE_V1_MULTIPROCESSING=0")
-        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
-                                               rpc_rank=0)
+        super()._init_executor()
+
+    def _distributed_args(self) -> tuple[str, int, int]:
         # engines are launched in torchrun-compatible launchers
         # so we can use the env:// method.
         # required env vars:
@@ -128,19 +158,7 @@ def _init_executor(self) -> None:
         distributed_init_method = "env://"
         rank = int(os.environ["RANK"])
         local_rank = int(os.environ["LOCAL_RANK"])
-        is_driver_worker = True
-        kwargs = dict(
-            vllm_config=self.vllm_config,
-            local_rank=local_rank,
-            rank=rank,
-            distributed_init_method=distributed_init_method,
-            is_driver_worker=is_driver_worker,
-        )
-        self.mm_receiver_cache = worker_receiver_cache_from_config(
-            self.vllm_config, MULTIMODAL_REGISTRY, Lock())
-        self.collective_rpc("init_worker", args=([kwargs], ))
-        self.collective_rpc("init_device")
-        self.collective_rpc("load_model")
+        return distributed_init_method, rank, local_rank
 
     def determine_num_available_blocks(self) -> Tuple[int, int]:
         """
diff --git a/vllm/v1/engine/core.py b/vllm/v1/engine/core.py
index 27ee146c4f6f..995e70385be8 100644
--- a/vllm/v1/engine/core.py
+++ b/vllm/v1/engine/core.py
@@ -159,6 +159,9 @@ def __init__(self,
             self.request_block_hasher = get_request_block_hasher(
                 block_size, caching_hash_fn)
 
+        self.step_fn = (self.step if self.batch_queue is None else
+                        self.step_with_batch_queue)
+
     def _initialize_kv_caches(
             self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
         start = time.time()
@@ -331,7 +334,8 @@ def step_with_batch_queue(
         model_executed = False
         if self.scheduler.has_requests():
             scheduler_output = self.scheduler.schedule()
-            future = self.model_executor.execute_model(scheduler_output)
+            future = self.model_executor.execute_model(scheduler_output,
+                                                       non_block=True)
             batch_queue.appendleft(
                 (future, scheduler_output))  # type: ignore[arg-type]
 
@@ -534,9 +538,6 @@ def __init__(
                 assert addresses.coordinator_input is not None
                 logger.info("Waiting for READY message from DP Coordinator...")
 
-        self.step_fn = (self.step if self.batch_queue is None else
-                        self.step_with_batch_queue)
-
         # Mark the startup heap as static so that it's ignored by GC.
         # Reduces pause times of oldest generation collections.
         gc.collect()
diff --git a/vllm/v1/engine/core_client.py b/vllm/v1/engine/core_client.py
index 605bedaf10e6..bb0f37c6e026 100644
--- a/vllm/v1/engine/core_client.py
+++ b/vllm/v1/engine/core_client.py
@@ -245,8 +245,8 @@ def __init__(self, *args, **kwargs):
         self.engine_core = EngineCore(*args, **kwargs)
 
     def get_output(self) -> EngineCoreOutputs:
-        outputs, _ = self.engine_core.step()
-        return outputs.get(0) or EngineCoreOutputs()
+        outputs, _ = self.engine_core.step_fn()
+        return outputs and outputs.get(0) or EngineCoreOutputs()
 
     def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
         return self.engine_core.get_supported_tasks()
diff --git a/vllm/v1/executor/abstract.py b/vllm/v1/executor/abstract.py
index 68408a0b8a3d..625017d52fff 100644
--- a/vllm/v1/executor/abstract.py
+++ b/vllm/v1/executor/abstract.py
@@ -2,7 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
 from concurrent.futures import Future
-from typing import Callable, Optional, Union
+from typing import Any, Callable, Optional, Union
 
 import torch
 import torch.distributed as dist
@@ -14,6 +14,7 @@
 from vllm.executor.uniproc_executor import (  # noqa
     UniProcExecutor as UniProcExecutorV0)
 from vllm.utils import resolve_obj_by_qualname
+from vllm.v1.core.sched.output import SchedulerOutput
 from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
 from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
 
@@ -86,12 +87,22 @@ def determine_available_memory(self) -> list[int]:  # in bytes
     def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
         return self.collective_rpc("get_kv_cache_spec")
 
+    def collective_rpc(self,
+                       method: Union[str, Callable],
+                       timeout: Optional[float] = None,
+                       args: tuple = (),
+                       kwargs: Optional[dict] = None,
+                       non_block: bool = False) -> list[Any]:
+        raise NotImplementedError
+
     def execute_model(
         self,
-        scheduler_output,
+        scheduler_output: SchedulerOutput,
+        non_block: bool = False,
     ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
         output = self.collective_rpc("execute_model",
-                                     args=(scheduler_output, ))
+                                     args=(scheduler_output, ),
+                                     non_block=non_block)
         return output[0]
 
     def execute_dummy_batch(self) -> None:
diff --git a/vllm/v1/executor/multiproc_executor.py b/vllm/v1/executor/multiproc_executor.py
index 2f0e5aa383b1..f566c9aee0c5 100644
--- a/vllm/v1/executor/multiproc_executor.py
+++ b/vllm/v1/executor/multiproc_executor.py
@@ -11,7 +11,7 @@
 from concurrent.futures import Future, ThreadPoolExecutor
 from dataclasses import dataclass
 from enum import Enum, auto
-from functools import partial
+from functools import cached_property, partial
 from multiprocessing.connection import Connection
 from multiprocessing.process import BaseProcess
 from multiprocessing.synchronize import Lock as LockType
@@ -37,6 +37,7 @@
 from vllm.utils import (decorate_logs, get_distributed_init_method,
                         get_loopback_ip, get_mp_context, get_open_port,
                         set_process_title)
+from vllm.v1.core.sched.output import SchedulerOutput
 from vllm.v1.executor.abstract import Executor, FailureCallback
 from vllm.v1.executor.utils import get_and_update_mm_cache
 from vllm.v1.outputs import (AsyncModelRunnerOutput, DraftTokenIds,
@@ -174,9 +175,9 @@ def register_failure_callback(self, callback: FailureCallback):
 
     def execute_model(
         self,
-        scheduler_output,
+        scheduler_output: SchedulerOutput,
+        non_block: bool = False,
     ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
-        non_block = self.max_concurrent_batches > 1
 
         if not self.has_connector:
             # get output only from a single worker (output_rank)
@@ -328,7 +329,7 @@ def check_health(self) -> None:
         self.collective_rpc("check_health", timeout=10)
         return
 
-    @property
+    @cached_property
     def max_concurrent_batches(self) -> int:
         if self.scheduler_config.async_scheduling:
             return 2
@@ -632,7 +633,8 @@ def enqueue_output(self, output: Any):
             result = (WorkerProc.ResponseStatus.FAILURE, str(output))
         else:
             result = (WorkerProc.ResponseStatus.SUCCESS, output)
-        self.worker_response_mq.enqueue(result)
+        if (response_mq := self.worker_response_mq) is not None:
+            response_mq.enqueue(result)
 
     def handle_output(self, output: Any):
         """Handles output from the worker. If async scheduling is enabled,
diff --git a/vllm/v1/executor/ray_distributed_executor.py b/vllm/v1/executor/ray_distributed_executor.py
index 8394ae788ab0..59c9b56625a9 100644
--- a/vllm/v1/executor/ray_distributed_executor.py
+++ b/vllm/v1/executor/ray_distributed_executor.py
@@ -66,11 +66,13 @@ def max_concurrent_batches(self) -> int:
     def execute_model(
         self,
         scheduler_output: SchedulerOutput,
+        non_block: bool = False,
     ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
         """Execute the model on the Ray workers.
 
         Args:
             scheduler_output: The scheduler output to execute.
+            non_block: If True, the method will return a Future.
 
         Returns:
             The model runner output.
@@ -84,7 +86,7 @@ def execute_model(
         if not self.has_connector:
             # Get output only from a single worker (output_rank)
             # When PP is not used, we block here until the result is available.
-            if self.max_concurrent_batches == 1:
+            if not non_block:
                 return refs[0].get()
 
             # When PP is used, we return a FutureWrapper immediately so that
@@ -92,7 +94,7 @@ def execute_model(
             return FutureWrapper(refs)
 
         # Get output from all workers when connector is present
-        if self.max_concurrent_batches == 1:
+        if not non_block:
             # Block and get results from all workers
             outputs = [ref.get() for ref in refs]
             return self.kv_output_aggregator.aggregate(outputs)
@@ -106,4 +108,3 @@ def reinitialize_distributed(
         if reconfig_request.new_data_parallel_rank == \
         ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
             self.shutdown()
-        return
\ No newline at end of file
```
