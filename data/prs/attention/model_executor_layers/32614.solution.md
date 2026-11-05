# PR #32614: fix: Add glm4_moe_lite to MLA detection (#32614)

# Issue

## Original PR description (captured source)

## Summary

- Add `glm4_moe_lite` and `glm4_moe_lite_mtp` to `is_deepseek_mla()` check in `model_arch_config_convertor.py`

GLM-4.7-Flash (`glm4_moe_lite`) uses Multi-head Latent Attention (MLA) via `Glm4MoeLiteMLAAttention` (which inherits from `DeepseekV2MLAAttention`) but was missing from the MLA detection.

Without this fix, vLLM falls back to standard KV caching instead of efficient MLA caching, resulting in ~4x higher KV cache memory usage.

Note: `glm4_moe` is intentionally NOT included as it uses standard attention (`Glm4MoeAttention` with `vllm.attention.layer.Attention`).

Co-authored with @mgoin 

NOTE: SM100 has issues with support for this model with various MLA decode and prefill kernels, so the following changes were made to support default inference there:
* Disable trtllm prefill and flashinfer prefill if we didn't find DeepSeek R1 compatible MLA dimensions
* Disable flashinfer mla if we didn't find DeepSeek R1 compatible MLA dimensions
* Explicitly enable cutlass mla so the block_size=128 gets enforced
* This means that on SM100 we will run with CUTLASS_MLA decode and FA2 prefill by default for this model, I also tested that TRITON_MLA works.

## Test Plan

- [x] Tested with `marksverdhei/GLM-4.7-Flash-fp8` on 2x RTX 3090
- [x] Verified MLA is detected and efficient KV caching is used
- [x] Model runs with 14.7 GB VRAM per GPU


## Full discussion

See [32614.md](./32614.md) and [32614.json](./32614.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/32614
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/32614.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `36845fef9d378c1e677e56739b3386ca834a9d4ccfe8beb6b5c24ddb2ec9a458`

Existing semantic notes in [32614.sem.json](./32614.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/platforms/cuda.py`
- `vllm/transformers_utils/model_arch_config_convertor.py`
- `vllm/v1/attention/backends/mla/flashinfer_mla.py`

### Diff

```diff
diff --git a/vllm/model_executor/layers/attention/mla_attention.py b/vllm/model_executor/layers/attention/mla_attention.py
index f57b825a924d..55c4c0e187bd 100755
--- a/vllm/model_executor/layers/attention/mla_attention.py
+++ b/vllm/model_executor/layers/attention/mla_attention.py
@@ -440,18 +440,34 @@ def __post_init__(self):
 A = TypeVar("A", bound=AttentionMetadata)
 
 
+def is_deepseek_r1_mla_compatible(vllm_config: VllmConfig) -> bool:
+    # Check if model has DeepSeek R1 compatible MLA dimensions:
+    # qk_nope_head_dim = 128, qk_rope_head_dim = 64, v_head_dim = 128
+    # which results in query/key head dim = 192.
+    if vllm_config.model_config is None:
+        return False
+    hf_text_config = vllm_config.model_config.hf_text_config
+    qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
+    qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 1)
+    v_head_dim = getattr(hf_text_config, "v_head_dim", 1)
+    return qk_nope_head_dim == 128 and qk_rope_head_dim == 64 and v_head_dim == 128
+
+
 def use_flashinfer_prefill() -> bool:
     # For blackwell default to flashinfer prefill if it's available since
     # it is faster than FA2.
     from vllm.config import get_current_vllm_config
 
     vllm_config = get_current_vllm_config()
-    return (
+    if not (
         not vllm_config.attention_config.disable_flashinfer_prefill
         and flashinfer_available
         and not vllm_config.attention_config.use_cudnn_prefill
         and current_platform.is_device_capability_family(100)
-    )
+    ):
+        return False
+
+    return is_deepseek_r1_mla_compatible(vllm_config)
 
 
 def use_cudnn_prefill() -> bool:
@@ -471,11 +487,14 @@ def use_trtllm_ragged_deepseek_prefill() -> bool:
     from vllm.config import get_current_vllm_config
 
     vllm_config = get_current_vllm_config()
-    return (
+    if not (
         flashinfer_available
         and vllm_config.attention_config.use_trtllm_ragged_deepseek_prefill
         and current_platform.is_device_capability_family(100)
-    )
+    ):
+        return False
+
+    return is_deepseek_r1_mla_compatible(vllm_config)
 
 
 @dataclass
diff --git a/vllm/platforms/cuda.py b/vllm/platforms/cuda.py
index e4af7136d553..725bf65063cb 100644
--- a/vllm/platforms/cuda.py
+++ b/vllm/platforms/cuda.py
@@ -180,19 +180,34 @@ def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
             use_cutlass_mla = False
             use_flashinfer_mla = False
 
+            from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported
+
             if vllm_config.attention_config.backend is None:
                 # Default case
-                if cls.is_device_capability_family(100) and not use_sparse:
-                    # Blackwell => Force FlashInfer MLA (unless sparse, i.e. DSv3.2).
+                hf_text_config = model_config.hf_text_config
+                qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
+                if (
+                    cls.is_device_capability_family(100)
+                    and not use_sparse
+                    and qk_nope_head_dim == 128
+                ):
+                    # Blackwell => Force FlashInfer MLA (unless sparse, i.e. DSv3.2)
+                    # and only if qk_nope_head_dim == 128 (kernel constraint)
                     use_flashinfer_mla = True
                     # Set the backend in AttentionConfig so it's used during
                     # backend selection
                     vllm_config.attention_config.backend = (
                         AttentionBackendEnum.FLASHINFER_MLA
                     )
-                else:
-                    # Not Blackwell
+                elif cls.is_device_capability_family(100) and not use_sparse:
+                    # Fall back to CUTLASS_MLA as 2nd priority on Blackwell
+                    use_cutlass_mla = True
+                elif is_flashmla_dense_supported()[0]:
+                    # Non-Blackwell with FlashMLA support
                     use_flashmla = True
+                else:
+                    # Fallback: will use Triton MLA or other compatible backend
+                    pass
             else:
                 # Forced case
                 backend = vllm_config.attention_config.backend
@@ -200,8 +215,6 @@ def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
                 use_cutlass_mla = backend == AttentionBackendEnum.CUTLASS_MLA
                 use_flashinfer_mla = backend == AttentionBackendEnum.FLASHINFER_MLA
 
-            from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported
-
             if (
                 use_flashmla
                 and is_flashmla_dense_supported()[0]
diff --git a/vllm/transformers_utils/model_arch_config_convertor.py b/vllm/transformers_utils/model_arch_config_convertor.py
index 6df4bb64dceb..d569c99ca214 100644
--- a/vllm/transformers_utils/model_arch_config_convertor.py
+++ b/vllm/transformers_utils/model_arch_config_convertor.py
@@ -189,6 +189,8 @@ def is_deepseek_mla(self) -> bool:
             "deepseek_v3",
             "deepseek_v32",
             "deepseek_mtp",
+            "glm4_moe_lite",
+            "glm4_moe_lite_mtp",
             "kimi_k2",
             "kimi_linear",
             "longcat_flash",
diff --git a/vllm/v1/attention/backends/mla/flashinfer_mla.py b/vllm/v1/attention/backends/mla/flashinfer_mla.py
index 236e293adf53..d1314ccf2955 100644
--- a/vllm/v1/attention/backends/mla/flashinfer_mla.py
+++ b/vllm/v1/attention/backends/mla/flashinfer_mla.py
@@ -63,6 +63,32 @@ def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
     def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
         return capability.major == 10
 
+    @classmethod
+    def supports_combination(
+        cls,
+        head_size: int,
+        dtype: torch.dtype,
+        kv_cache_dtype: CacheDType | None,
+        block_size: int,
+        use_mla: bool,
+        has_sink: bool,
+        use_sparse: bool,
+        device_capability: DeviceCapability,
+    ) -> str | None:
+        # FlashInfer MLA kernel requires qk_nope_head_dim == 128
+        from vllm.config import get_current_vllm_config
+
+        vllm_config = get_current_vllm_config()
+        if vllm_config.model_config is not None:
+            hf_text_config = vllm_config.model_config.hf_text_config
+            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
+            if qk_nope_head_dim != 128:
+                return (
+                    f"FlashInfer MLA kernel requires qk_nope_head_dim == 128, "
+                    f"but got {qk_nope_head_dim}"
+                )
+        return None
+
     @classmethod
     def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
         return "HND"
```
