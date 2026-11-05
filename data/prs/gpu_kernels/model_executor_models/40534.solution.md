# PR #40534: [Model] Gemma4: add bidirectional vision attention for sliding layers with window guard (#40534)

# Issue

## Original PR description (captured source)

## Purpose

Implement `use_bidirectional_attention="vision"` support for Gemma4 models
(gemma-4-31B-it, gemma-4-26B-A4B-it), addressing #40106.

Gemma4's architecture applies bidirectional attention to vision tokens in
sliding_attention layers only, allowing image patches to attend to each
other regardless of causal ordering. This matches the HF transformers
reference (`create_causal_mask_mapping` applies an OR mask to
`create_sliding_window_causal_mask` for sliding layers, while full_attention
layers use plain causal).

### Changes

- Register `gemma4` in `MM_PREFIX_LM_MODELS` to enable `mm_prefix_range`
  population for vision token ranges during prefill
- Add `_clear_mm_prefix_for_full_attn_layers` in
  `Gemma4ForConditionalGeneration.forward()` to clear `mm_prefix_range` on
  full_attention layers. This runs outside the `@support_torch_compile`
  boundary since `_run_decoder_layers` is inside a compiled graph where
  Python side effects are eliminated
- Add per-range sliding window guard in `gpu_model_runner`: skip bidi for
  image ranges where `num_tokens > sliding_window`

### Why the sliding window guard

Gemma4 uses `sliding_window=1024`. At the default `max_soft_tokens=280`,
images produce ~256 tokens (within the window) and bidi works correctly.
At `max_soft_tokens=1120`, images produce ~1092 tokens (exceeding the
window). 

The guard filters per-range, so in multi-image or video prompts:
- Small images (≤1024 tokens): bidi enabled
- Large images (>1024 tokens): bidi skipped
- Video frames (~64 tokens each): always enabled

This mirrors the spirit of the `use_bidirectional_attention="all"` config
which halves `sliding_window` to compensate for bidirectional span doubling.

## Test Result

**MMMU-Pro** (gemma-4-31B-it, 1730 samples):

| max_soft_tokens | Image tokens | Within window? | No bidi | With bidi | Delta |
|----------------|-------------|---------------|---------|-----------|-------|
| 280 | 256 | Yes | 72.14% | 72.95% | **+0.81%** |
| 560 | 529 | Yes | 73.29% | 74.39% | **+1.10%** |
| 1120 | 1092 | No (guard skips) | 74.28% | 74.28% | 0% (guarded) |

**ScreenSpot-Pro** (gemma-4-31B-it, 1580 samples, max_soft_tokens=1120):

| Config | Click Accuracy |
|--------|---------------|
| No bidi | 37.26% |
| With bidi + guard | 37.26% (guarded, no regression) |

Bidi improves MMMU-Pro by +0.81–1.10% when images fit within the sliding
window. The guard prevents the 3–4% ScreenSpot-Pro regression observed
without it (caused by attention explosion at max_soft_tokens=1120).

## Full discussion

See [40534.md](./40534.md) and [40534.json](./40534.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/40534
- Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/40534
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `f372628ac7d8fd9959dd7fc3cd8af49761103f5bebb4535cad690682a3c803e6`

Existing semantic notes in [40534.sem.json](./40534.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/model_executor/models/gemma4_mm.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Diff

```diff
diff --git a/vllm/model_executor/models/gemma4_mm.py b/vllm/model_executor/models/gemma4_mm.py
index 46d0308f4c86..cdc54609a652 100644
--- a/vllm/model_executor/models/gemma4_mm.py
+++ b/vllm/model_executor/models/gemma4_mm.py
@@ -969,6 +969,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
             self.language_model.make_empty_intermediate_tensors
         )
 
+        # --- Precompute full-attention layer indices for bidi clearing ---
+        self._full_attn_layer_idxs: frozenset[int] = frozenset()
+        text_config = config.text_config
+        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
+            layer_types = getattr(text_config, "layer_types", None)
+            if layer_types:
+                self._full_attn_layer_idxs = frozenset(
+                    i for i, lt in enumerate(layer_types) if lt != "sliding_attention"
+                )
+
         # --- MixtureOfExperts delegation to language_model ---
         self.expert_weights = self.language_model.expert_weights
         self.moe_layers = self.language_model.moe_layers
@@ -1310,6 +1320,12 @@ def forward(
             else None
         )
 
+        # Gemma4 bidi: clear mm_prefix_range for full_attention layers.
+        # Must run here (outside @support_torch_compile boundary) because
+        # _run_decoder_layers is inside a compiled graph where Python
+        # side effects are eliminated.
+        self._clear_mm_prefix_for_full_attn_layers()
+
         hidden_states = self.language_model.model(
             input_ids,
             positions,
@@ -1327,6 +1343,49 @@ def compute_logits(
     ) -> torch.Tensor | None:
         return self.language_model.compute_logits(hidden_states)
 
+    # ------------------------------------------------------------------ #
+    # Bidirectional attention helpers
+    # ------------------------------------------------------------------ #
+
+    def _clear_mm_prefix_for_full_attn_layers(self) -> None:
+        """Clear mm_prefix_range for non-sliding layers.
+
+        Gemma4 with use_bidirectional_attention='vision' applies
+        bidirectional attention only to sliding_attention layers.
+        Full attention layers use plain causal masking.
+
+        Uses _full_attn_layer_idxs (precomputed in __init__) for O(1)
+        lookup instead of per-call regex parsing.
+        """
+        if not self._full_attn_layer_idxs:
+            return
+
+        from vllm.forward_context import get_forward_context
+
+        attn_metadata = get_forward_context().attn_metadata
+        if attn_metadata is None:
+            return
+
+        def _process(metadata_dict: dict) -> None:
+            for layer_name, metadata in metadata_dict.items():
+                if ".layers." not in layer_name:
+                    continue
+                try:
+                    layer_idx = int(layer_name.split(".layers.")[1].split(".")[0])
+                except (ValueError, IndexError):
+                    continue
+                if layer_idx in self._full_attn_layer_idxs:
+                    if hasattr(metadata, "mm_prefix_range"):
+                        metadata.mm_prefix_range = None
+                    if hasattr(metadata, "mm_prefix_range_tensor"):
+                        metadata.mm_prefix_range_tensor = None
+
+        if isinstance(attn_metadata, list):
+            for ub_metadata in attn_metadata:
+                _process(ub_metadata)
+        elif isinstance(attn_metadata, dict):
+            _process(attn_metadata)
+
     # ------------------------------------------------------------------ #
     # Weight loading
     # ------------------------------------------------------------------ #
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 0b0fed4824ab..0362011a6e6d 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -2314,13 +2314,26 @@ def _build_attn_group_metadata(
 
         if self.is_mm_prefix_lm:
             req_doc_ranges = {}
+
+            # Gemma4 bidi: skip ranges that exceed the sliding
+            # window. When image tokens > sliding_window, bidi causes
+            # early image tokens to attend to the entire image
+            # (e.g. 6 → 1092 targets), degrading spatial precision.
+            # Per-range filtering keeps bidi for small images/video
+            # frames while skipping oversized images.
+            hf_text_config = self.model_config.hf_text_config
+            _bidi_sw = getattr(hf_text_config, "sliding_window", None)
+
             for req_id in self.input_batch.req_ids:
                 image_doc_ranges = []
                 req_state = self.requests[req_id]
                 for mm_feature in req_state.mm_features:
                     pos_info = mm_feature.mm_position
                     img_doc_range = pos_info.extract_embeds_range()
-                    image_doc_ranges.extend(img_doc_range)
+                    for r in img_doc_range:
+                        if _bidi_sw is not None and (r[1] - r[0] + 1) > _bidi_sw:
+                            continue
+                        image_doc_ranges.append(r)
                 req_idx = self.input_batch.req_id_to_index[req_id]
                 req_doc_ranges[req_idx] = image_doc_ranges
 
```
