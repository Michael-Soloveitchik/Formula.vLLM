# PR #32133: [QeRL] Layerwise Reloading

- field: quantization  | area: model_executor/model_loader
- author: kylesayrs
- state: MERGED  created: 2026-01-12T00:20:08Z  merged: 2026-01-30T15:50:05Z
- url: https://github.com/vllm-project/vllm/pull/32133
- labels: ready, v1

## Body

## Purpose ##
* Support weight reloading of quantized models
  * Improves upon https://github.com/vllm-project/vllm/pull/28480
    * Changes are more maintainable
      * FP8 can now do arbitrary kernel processing (padding, renaming, deepgemm, ect.)
      * FP8 does not need to reattach weight loaders after processing
      * Can remove complications around renaming weight_scale_inv
    * Support reloading with FP8 marlin
  * Improves upon https://github.com/vllm-project/vllm/pull/23014
    * Reduce memory requirements by 3x for FP8
    * Expand support to all quant configs, not just torchao
  * Support reloading of (theoretically any quantization scheme)
    * INT4 has been tested
    * FP8_BLOCK has been tested
    * FP8_DYNAMIC has been tested
    * NVFP4_A16 has been tested

### Nomenclature ##
<details>

- “Checkpoint format” refers to the format in which weights are loaded from disk or provided by a user.
- “Model format” refers to the state of the model after `init` but before weights are processed with `process_weights_after_loading` . The mapping between “checkpoint format” and “model format” is implemented by `model.load_weights`.
- “Kernel format” refers to the state of the model after `process_weights_after_loading`
- In the case that checkpoint format is unquantized, but the kernel format is quantized, we call this “online quantization”, where unquantized weights are quantized by vLLM during/after loading.
</details>

## Design ##
1. Weight metadata (shape, attributes) are captured before kernel processing
2. Kernel processing occurs
3. User triggers reloading
4. Weight metadata is used to restore the layer to original load-ready model format
5. Loaded weights are cached
6. Once all loaded weights are ready for a given layer
  a. the layer is materialized on device
  b. the weights are loaded into the layer
  c. the layer is processed into kernel format
  d. the new tensors are copied into the original tensor storage
7. Any remaining layers which did not load all weights are cleaned up

## Changes ##
* Implement 4 public functions
  * `record_metadata_for_reloading` is called at the end of `initialize_model`
  * `initialize_layerwise_reload` is called before weights are loaded a second time
  * `finalize_layerwise_reload` is called after weights are loaded a second
  * `support_quantized_model_reload_from_hp_weights` is a decorator which calls `initialize_layerwise_reload` and `finalize_layerwise_reload`, preserves backwards compatibility for torchao users
* Track layer information (restore metadata, kernel tensors, loading progresss) via the global dictionary `LAYER_RELOADING_INFO`
  * Before and after weight reloading, this dictionary only holds meta tensors. This dictionary is a weak key dictionary, so those meta tensors are garbage collected if the model is garbage collected.
* Expand capabilities of `lm.collective_rpc("reload_weights")`
  * Can now pass `weights_iterator` directly
  * Alternatively, can pass `weights_path` to point to another checkpoint on disk
  * `checkpoint_format` controls whether the weights are provided in model format (default) or kernel format (quantized, renamed, and sharded)
* Misc
  * Improve `generate_prompt_perplexity` by allowing users to pass a mask

## Testing ##
* Add unit tests for materializing and restoring layers
* Add model tests for full precision, int4, and fp8 weight updating
* Add model test for MoE weight updating
* Modify `test_online_quant_config_dict_json` to actually test online quantization via the `model.load_weights` pathway

## Future Work ##
* Implement composing with the following features
  * Attention/MLA quantization
  * Expert-parallel load balancing (all that should be needed is to pause async EPLB during reload)
  * Reuse code and compose with https://github.com/vllm-project/vllm/pull/31914
  * Distributed weight transfer and updating: #31943
  * Remove logic from https://github.com/vllm-project/vllm/pull/28480

## Comments

### mergify @ 2026-01-22T16:37:56Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @kylesayrs.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---