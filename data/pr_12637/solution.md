# PR #12637: Apply torch.compile to fused_moe/grouped_topk

- field: gpu_kernels  | area: model_executor_layers
- author: mgoin
- state: MERGED  created: 2025-02-01T03:16:21Z  merged: 2025-02-01T16:16:19Z
- url: https://github.com/vllm-project/vllm/pull/12637
- labels: ready

# Issue

## Problem

The `grouped_topk` routing function used by DeepSeek-V2/V3 MoE layers runs as eager PyTorch, executing many small elementwise and reduction ops (reshape, group max, topk, softmax, masking) as separate kernel launches on every decode step. With a large number of experts and expert groups, this per-op dispatch overhead dominates the tiny actual compute at low batch size. Additionally the full DeepSeekV3 model was not wrapped in `support_torch_compile`, leaving the graph un-fused.

## Example

For `DeepSeek-Coder-V2-Lite-Instruct` at batch size 1 on an H100, decode throughput was 154 tokens/s because each grouped-topk routing step fired a chain of tiny kernels instead of one fused graph.

## Use case

Low-batch / single-request decode serving of DeepSeek MoE models, where generation latency per token matters and routing overhead is a measurable fraction of step time.

## Resources

Kernel-launch overhead and GPU dispatch latency — many small ops stress the launch path, not raw compute or HBM bandwidth.

## Type of users

Users serving DeepSeek-family MoE models (V2/V3, Coder-V2-Lite) at small batch sizes on single GPUs like H100.

## Kind of inputs

Small batch size (bs=1), short sequences in decode phase, models with large expert counts and grouped expert routing.

## Mechanism with the issue

`grouped_topk` in `fused_moe.py` performs group-wise reshape, per-group max, group topk mask, scatter back, then `topk` + softmax as independent eager ops. At bs=1 the compute per op is trivial so launch overhead per kernel dominates, making routing slow. The DeepSeekV3 model class also lacked `@support_torch_compile`.

# Solution

## Fix in words

Adds `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)` on `grouped_topk` so the whole routing op chain fuses into one compiled graph with dynamic shapes, cutting per-op kernel-launch overhead. Enables `@support_torch_compile` on `DeepseekV3Model` (removing the TODO-disabled decorator) so the model graph gets compiled too. No change to routing math — only compilation. Result: decode throughput on DeepSeek-Coder-V2-Lite-Instruct rose 154 → 163 tokens/s (~5%) at bs=1 on H100.

## Author's solution

Apply `torch.compile` to the `grouped_topk`/fused_moe routing path, inspired by an equivalent SGLang commit, to reduce overhead from DeepSeek's large expert count. Also enable `support_torch_compile` on the DeepSeek-V3 model. Benchmarked a 5% generation-throughput gain (154 → 163 tokens/s) at bs=1 on H100 with DeepSeek-Coder-V2-Lite-Instruct.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/fused_moe.py`
- `vllm/model_executor/models/deepseek_v3.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/fused_moe.py b/vllm/model_executor/layers/fused_moe/fused_moe.py
index c966be99ed24..c80e6bf07468 100644
--- a/vllm/model_executor/layers/fused_moe/fused_moe.py
+++ b/vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -759,6 +759,7 @@ def fused_topk(
 
 
 # This is used by the Deepseek-V2 and Deepseek-V3 model
+@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
 def grouped_topk(hidden_states: torch.Tensor,
                  gating_output: torch.Tensor,
                  topk: int,
diff --git a/vllm/model_executor/models/deepseek_v3.py b/vllm/model_executor/models/deepseek_v3.py
index f6ab53c85faa..06ea3dab93e1 100644
--- a/vllm/model_executor/models/deepseek_v3.py
+++ b/vllm/model_executor/models/deepseek_v3.py
@@ -27,6 +27,7 @@
 from transformers import PretrainedConfig
 
 from vllm.attention import Attention, AttentionMetadata
+from vllm.compilation.decorators import support_torch_compile
 from vllm.config import CacheConfig, ModelConfig, VllmConfig
 from vllm.distributed import (get_pp_group,
                               get_tensor_model_parallel_world_size,
@@ -566,8 +567,7 @@ def forward(
         return hidden_states, residual
 
 
-# TODO(simon): check whether we support torch compile for Deepseek V3
-# @support_torch_compile
+@support_torch_compile
 class DeepseekV3Model(nn.Module):
 
     fall_back_to_pt_during_load = False

```
