# PR #11870: [perf]fix current stream

- field: parallelism  | area: distributed_device_communicators
- author: youkaichao
- state: MERGED  created: 2025-01-09T02:00:36Z  merged: 2025-01-09T07:18:22Z
- url: https://github.com/vllm-project/vllm/pull/11870
- labels: ready

# Issue

## Problem

Every collective call in the pynccl path invokes `torch.cuda.current_stream()` to resolve the active CUDA stream. This call is expensive because PyTorch constructs a fresh Python `Stream` object on each invocation rather than returning a cached handle. On the hot forward path, per-step all_reduce triggers this allocation repeatedly, adding CPU overhead that shows up as a measurable latency regression.

## Example

Benchmarking Meta-Llama-3-70B with tp=4, enforce-eager, dummy weights: per-step forward time at batchsize 8 drops from 20.77ms to 20.45ms, and avg latency from 2.946s to 2.852s (~3%).

## Use case

Tensor-parallel inference where each decode step issues NCCL all_reduce across GPUs, so the stream lookup runs once per collective per step.

## Resources

CPU / kernel-launch overhead on the forward hot path — the cost is Python object construction, not GPU compute or interconnect bandwidth.

## Type of users

Multi-GPU users running tensor parallelism (TP>1), where pynccl collectives execute on every forward step.

## Kind of inputs

Small per-step decode batches (e.g. batchsize 8) where fixed per-call overhead dominates relative to actual compute.

## Mechanism with the issue

`pynccl.py` collectives (all_reduce, all_gather, reduce_scatter, send, recv, broadcast, warmup) and `parallel_state._all_reduce_out_place` all called `torch.cuda.current_stream()`, which rebuilds a Stream object each time. Called once per collective per step, this allocation cost accumulates on the critical decode path.

# Solution

## Fix in words

Adds `vllm.utils.current_stream()` which monkey-patches `torch.cuda.set_stream` to record the active stream in a module-global `_current_stream`, so lookups return the cached object instead of constructing a new one each call. All `torch.cuda.current_stream()` sites in `pynccl.py` and `multi_step_model_runner.py` are replaced with the cheap cached `current_stream()`. In `parallel_state._all_reduce_out_place`, the explicit `stream=torch.cuda.current_stream()` arg is dropped so pynccl resolves the stream itself via the cached path. First-call fallback lazily populates `_current_stream` from `torch.cuda.current_stream()`; hypothesis is no stream set happens from C/C++ bypassing the patch.

## Author's solution

Author found `torch.cuda.current_stream()` is expensive because it builds a new stream object on every call. Fix patches `torch.cuda.set_stream` to track the current stream directly in a global, exposing `vllm.utils.current_stream()` as a cheap replacement, and swaps all call sites over to it. Local benchmark shows ~3% latency improvement and per-step forward time dropping from 20.77ms to 20.45ms at batchsize 8.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/distributed/device_communicators/pynccl.py`
- `vllm/distributed/parallel_state.py`
- `vllm/utils.py`
- `vllm/worker/multi_step_model_runner.py`

### Diff
```diff
diff --git a/vllm/distributed/device_communicators/pynccl.py b/vllm/distributed/device_communicators/pynccl.py
index fda4d007ceb5..efc59987195f 100644
--- a/vllm/distributed/device_communicators/pynccl.py
+++ b/vllm/distributed/device_communicators/pynccl.py
@@ -10,6 +10,7 @@
     ncclRedOpTypeEnum, ncclUniqueId)
 from vllm.distributed.utils import StatelessProcessGroup
 from vllm.logger import init_logger
+from vllm.utils import current_stream
 
 logger = init_logger(__name__)
 
@@ -96,7 +97,7 @@ def __init__(
             self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                 self.world_size, self.unique_id, self.rank)
 
-            stream = torch.cuda.current_stream()
+            stream = current_stream()
             # A small all_reduce for warmup.
             data = torch.zeros(1, device=device)
             self.all_reduce(data)
@@ -119,7 +120,7 @@ def all_reduce(self,
         out_tensor = torch.empty_like(in_tensor)
 
         if stream is None:
-            stream = torch.cuda.current_stream()
+            stream = current_stream()
         self.nccl.ncclAllReduce(buffer_type(in_tensor.data_ptr()),
                                 buffer_type(out_tensor.data_ptr()),
                                 in_tensor.numel(),
@@ -141,7 +142,7 @@ def all_gather(self,
             f"this nccl communicator is created to work on {self.device}, "
             f"but the input tensor is on {input_tensor.device}")
         if stream is None:
-            stream = torch.cuda.current_stream()
+            stream = current_stream()
         self.nccl.ncclAllGather(
             buffer_type(input_tensor.data_ptr()),
             buffer_type(output_tensor.data_ptr()), input_tensor.numel(),
@@ -162,7 +163,7 @@ def reduce_scatter(self,
             f"this nccl communicator is created to work on {self.device}, "
             f"but the input tensor is on {input_tensor.device}")
         if stream is None:
-            stream = torch.cuda.current_stream()
+            stream = current_stream()
         self.nccl.ncclReduceScatter(
             buffer_type(input_tensor.data_ptr()),
             buffer_type(output_tensor.data_ptr()), output_tensor.numel(),
@@ -177,7 +178,7 @@ def send(self, tensor: torch.Tensor, dst: int, stream=None):
             f"this nccl communicator is created to work on {self.device}, "
             f"but the input tensor is on {tensor.device}")
         if stream is None:
-            stream = torch.cuda.current_stream()
+            stream = current_stream()
         self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                            ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                            self.comm, cudaStream_t(stream.cuda_stream))
@@ -189,7 +190,7 @@ def recv(self, tensor: torch.Tensor, src: int, stream=None):
             f"this nccl communicator is created to work on {self.device}, "
             f"but the input tensor is on {tensor.device}")
         if stream is None:
-            stream = torch.cuda.current_stream()
+            stream = current_stream()
         self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                            ncclDataTypeEnum.from_torch(tensor.dtype), src,
                            self.comm, cudaStream_t(stream.cuda_stream))
@@ -201,7 +202,7 @@ def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
             f"this nccl communicator is created to work on {self.device}, "
             f"but the input tensor is on {tensor.device}")
         if stream is None:
-            stream = torch.cuda.current_stream()
+            stream = current_stream()
         if src == self.rank:
             sendbuff = buffer_type(tensor.data_ptr())
             # NCCL requires the sender also to have a receive buffer
diff --git a/vllm/distributed/parallel_state.py b/vllm/distributed/parallel_state.py
index a837c1dc5953..be7f16ef52a4 100644
--- a/vllm/distributed/parallel_state.py
+++ b/vllm/distributed/parallel_state.py
@@ -357,10 +357,7 @@ def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
             return out
         pynccl_comm = self.pynccl_comm
         assert pynccl_comm is not None
-        # TODO: pynccl should not use `stream=`
-        # it can just always use the current stream.
-        out = pynccl_comm.all_reduce(input_,
-                                     stream=torch.cuda.current_stream())
+        out = pynccl_comm.all_reduce(input_)
         if out is None:
             # fall back to the default all-reduce using PyTorch.
             # this usually happens during testing.
diff --git a/vllm/utils.py b/vllm/utils.py
index c09cae70e9af..99c4288af8cb 100644
--- a/vllm/utils.py
+++ b/vllm/utils.py
@@ -956,6 +956,39 @@ def find_nccl_library() -> str:
     return so_file
 
 
+prev_set_stream = torch.cuda.set_stream
+
+_current_stream = None
+
+
+def _patched_set_stream(stream: torch.cuda.Stream) -> None:
+    global _current_stream
+    _current_stream = stream
+    prev_set_stream(stream)
+
+
+torch.cuda.set_stream = _patched_set_stream
+
+
+def current_stream() -> torch.cuda.Stream:
+    """
+    replace `torch.cuda.current_stream()` with `vllm.utils.current_stream()`.
+    it turns out that `torch.cuda.current_stream()` is quite expensive,
+    as it will construct a new stream object at each call.
+    here we patch `torch.cuda.set_stream` to keep track of the current stream
+    directly, so that we can avoid calling `torch.cuda.current_stream()`.
+
+    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
+    from C/C++ code.
+    """
+    global _current_stream
+    if _current_stream is None:
+        # when this function is called before any stream is set,
+        # we return the default stream.
+        _current_stream = torch.cuda.current_stream()
+    return _current_stream
+
+
 def enable_trace_function_call_for_thread(vllm_config: "VllmConfig") -> None:
     """Set up function tracing for the current thread,
     if enabled via the VLLM_TRACE_FUNCTION environment variable
diff --git a/vllm/worker/multi_step_model_runner.py b/vllm/worker/multi_step_model_runner.py
index a2c2cebf8d1f..acce923498d7 100644
--- a/vllm/worker/multi_step_model_runner.py
+++ b/vllm/worker/multi_step_model_runner.py
@@ -14,7 +14,7 @@
                                                 get_pythonized_sample_results)
 from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                            Logprob, SequenceGroupMetadata, SequenceOutput)
-from vllm.utils import PyObjectCache, async_tensor_h2d
+from vllm.utils import PyObjectCache, async_tensor_h2d, current_stream
 from vllm.worker.model_runner import (GPUModelRunnerBase,
                                       ModelInputForGPUWithSamplingMetadata)
 from vllm.worker.model_runner_base import (
@@ -498,7 +498,7 @@ def execute_model(
         #   appended sampler output from last iteration
         #   - also maybe pythonize if CPU is ahead of GPU
 
-        current_stream = torch.cuda.current_stream()
+        stream = current_stream()
         if not model_input.is_first_multi_step:
             # Explicitly block on the previous step's forward to make sure we
             # don't clobber any GPU tensors still in use.
@@ -541,7 +541,7 @@ def execute_model(
                                                        num_steps=1)
 
         # record the event for the current step so that the next step can sync
-        model_input.record_step_event(current_stream)
+        model_input.record_step_event(stream)
 
         if get_pp_group().is_last_rank and self.is_driver_worker:
             assert isinstance(output, list)
@@ -552,7 +552,7 @@ def execute_model(
             # event for the pythonization so that we only pythonize if the
             # tensors are ready. May be able to be combined with the step event
             output_ready_event = torch.cuda.Event()
-            output_ready_event.record(current_stream)
+            output_ready_event.record(stream)
             if self.parallel_config.pipeline_parallel_size > 1:
                 output[0].sampled_token_ids_cpu = output[
                     0].sampled_token_ids.cpu()

```
