# PR #6695: PP comm optimization: replace send with partial send + allgather

- field: parallelism  | area: distributed_parallel_state.py
- author: aurickq
- state: MERGED  created: 2024-07-23T16:27:43Z  merged: 2024-08-01T03:15:42Z
- url: https://github.com/vllm-project/vllm/pull/6695
- labels: ready

# Issue

## Problem

Pipeline-parallel stages pass the full `IntermediateTensors` (hidden states) between adjacent PP ranks with a single point-to-point `torch.distributed.send`/`recv`. When tensor-parallelism is also enabled, every TP rank in the sending stage redundantly sends the entire tensor, even though the data is identical/partitionable across the TP group. This wastes cross-stage communication bandwidth and inflates PP transfer latency, capping throughput.

## Example

Serving Llama 3.1 405B with combined PP and TP, where each PP boundary ships a large hidden-state activation tensor; the full-tensor send dominated inter-stage comm and left ~5% throughput on the table.

## Use case

Multi-GPU inference of very large models split across both pipeline and tensor parallel dimensions, where activations must hop between PP stages each forward step.

## Resources

Inter-GPU interconnect bandwidth (NVLink/PCIe/network) for the PP send/recv, and the latency it adds to each pipeline step.

## Type of users

Users serving large models (e.g. 405B-class) with simultaneous tensor + pipeline parallelism across many GPUs/nodes.

## Kind of inputs

Large hidden-state activation tensors at PP stage boundaries whose element count is divisible by the TP world size; bigger batch/sequence makes the transferred tensor larger and the waste worse.

## Mechanism with the issue

`send_tensor_dict`/`recv_tensor_dict` in `parallel_state.py` do a plain `torch.distributed.send`/`recv` of the whole tensor over the PP device group. The full activation crosses the PP link once per tensor, with no coordination with the TP group, so the entire payload traverses the slow inter-stage link instead of being split and reconstructed cheaply over fast intra-stage TP links.

# Solution

## Fix in words

Adds an optional `all_gather_group` arg to `send_tensor_dict` and `recv_tensor_dict`. Before: full tensor sent/received over PP link. After: when a TP group is passed and `tensor.numel() % all_gather_size == 0`, the tensor is reshaped to `(all_gather_size, -1)` and only the sender's slice `[all_gather_rank]` is sent over PP; the receiver recvs just its slice and then does `all_gather(dim=0)` over the TP group and reshapes back to `orig_shape`. `worker_base.py` wires `get_tp_group()` into both the PP send and recv calls. Net effect: each PP rank ships only 1/TP of the activation across the slow inter-stage link, reconstructing the full tensor via a cheap intra-stage allgather.

## Author's solution

Replace the full point-to-point send across PP stages with a partial send of a per-TP-rank slice followed by an allgather within the TP group to reconstruct the full tensor. Author reports ~5% throughput improvement on Llama 3.1 405B.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/distributed/parallel_state.py`
- `vllm/worker/worker_base.py`

### Diff
```diff
diff --git a/vllm/distributed/parallel_state.py b/vllm/distributed/parallel_state.py
index bf7a7de0724a..d7ca8fd82e1a 100644
--- a/vllm/distributed/parallel_state.py
+++ b/vllm/distributed/parallel_state.py
@@ -569,7 +569,8 @@ def broadcast_tensor_dict(
     def send_tensor_dict(
         self,
         tensor_dict: Dict[str, Union[torch.Tensor, Any]],
-        dst: Optional[int] = None
+        dst: Optional[int] = None,
+        all_gather_group: Optional["GroupCoordinator"] = None,
     ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
         """Send the input tensor dictionary.
         NOTE: `dst` is the local rank of the source rank.
@@ -578,6 +579,11 @@ def send_tensor_dict(
         if not torch.distributed.is_initialized() or self.world_size == 1:
             return tensor_dict
 
+        all_gather_size = (1 if all_gather_group is None else
+                           all_gather_group.world_size)
+        all_gather_rank = (0 if all_gather_group is None else
+                           all_gather_group.rank_in_group)
+
         group = self.device_group
         metadata_group = self.cpu_group
 
@@ -598,6 +604,12 @@ def send_tensor_dict(
             if tensor.numel() == 0:
                 # Skip sending empty tensors.
                 continue
+
+            # send-allgather: send only a slice, then do allgather.
+            if (all_gather_group is not None
+                    and tensor.numel() % all_gather_size == 0):
+                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
+
             if tensor.is_cpu:
                 # use metadata_group for CPU tensors
                 torch.distributed.send(tensor,
@@ -612,7 +624,8 @@ def send_tensor_dict(
 
     def recv_tensor_dict(
         self,
-        src: Optional[int] = None
+        src: Optional[int] = None,
+        all_gather_group: Optional["GroupCoordinator"] = None,
     ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
         """Recv the input tensor dictionary.
         NOTE: `src` is the local rank of the source rank.
@@ -621,6 +634,11 @@ def recv_tensor_dict(
         if not torch.distributed.is_initialized() or self.world_size == 1:
             return None
 
+        all_gather_size = (1 if all_gather_group is None else
+                           all_gather_group.world_size)
+        all_gather_rank = (0 if all_gather_group is None else
+                           all_gather_group.rank_in_group)
+
         group = self.device_group
         metadata_group = self.cpu_group
 
@@ -639,6 +657,16 @@ def recv_tensor_dict(
                     # Skip broadcasting empty tensors.
                     tensor_dict[key] = tensor
                     continue
+
+                # send-allgather: send only a slice, then do allgather.
+                use_all_gather = (all_gather_group is not None
+                                  and tensor.numel() % all_gather_size == 0)
+
+                if use_all_gather:
+                    orig_shape = tensor.shape
+                    tensor = tensor.reshape(all_gather_size,
+                                            -1)[all_gather_rank]
+
                 if tensor.is_cpu:
                     # use metadata_group for CPU tensors
                     torch.distributed.recv(tensor,
@@ -649,6 +677,12 @@ def recv_tensor_dict(
                     torch.distributed.recv(tensor,
                                            src=self.ranks[src],
                                            group=group)
+                if use_all_gather:
+                    # do the allgather
+                    tensor = all_gather_group.all_gather(  # type: ignore
+                        tensor, dim=0)
+                    tensor = tensor.reshape(orig_shape)
+
                 tensor_dict[key] = tensor
             else:
                 tensor_dict[key] = value
diff --git a/vllm/worker/worker_base.py b/vllm/worker/worker_base.py
index 03e3857e23c4..8a4d1958c65a 100644
--- a/vllm/worker/worker_base.py
+++ b/vllm/worker/worker_base.py
@@ -6,7 +6,7 @@
 
 import torch
 
-from vllm.distributed import broadcast_tensor_dict, get_pp_group
+from vllm.distributed import broadcast_tensor_dict, get_pp_group, get_tp_group
 from vllm.logger import init_logger
 from vllm.lora.request import LoRARequest
 from vllm.platforms import current_platform
@@ -267,7 +267,8 @@ def execute_model(
         intermediate_tensors = None
         if not get_pp_group().is_first_rank:
             intermediate_tensors = IntermediateTensors(
-                get_pp_group().recv_tensor_dict())
+                get_pp_group().recv_tensor_dict(
+                    all_gather_group=get_tp_group()))
 
         output = self.model_runner.execute_model(
             model_input, self.kv_cache[worker_input.virtual_engine]
@@ -276,7 +277,8 @@ def execute_model(
 
         if not get_pp_group().is_last_rank:
             # output is IntermediateTensors
-            get_pp_group().send_tensor_dict(output.tensors)
+            get_pp_group().send_tensor_dict(output.tensors,
+                                            all_gather_group=get_tp_group())
             return [None]
 
         # output is List[SamplerOutput]

```
