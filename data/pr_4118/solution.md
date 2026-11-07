# PR #4118: [Kernel][FP8] Initial support with dynamic per-tensor scaling (#4118)

# Issue

## Original PR description (captured source)

Provide an initial support to FP8 computation. This PR is inspired by HuggingFace TGI: https://github.com/huggingface/text-generation-inference/pull/1726

This feature can be enabled with `--quantization fp8` or `-q fp8` when launching an engine.

## Algorithm
We still load a model checkpoint in FP16/BF16. After the weights are loaded, Fp8LinearMethod calculates the per-tensor scaling factor of weights and quantizes the weights accordingly. The scaling factor will then be stored for future use. Meanwhile, the per-tensor scaling factor for activations is calculated in every forward pass.

## Initial Results

Currently tested Mistral-7B on 1xH100. With prompt length ~5 and decoding length 128:
* BF16: 1.47s
* FP8: 1.66s

I'll try to use larger models and try to find more performance bottleneck. Meanwhile, you're welcome to try this PR.

## To Do
- [ ] Unit tests
- [ ] Comprehensive benchmarking
- [ ] Refine the interface

cc @WoosukKwon @zhuohan123 @robertgshaw2-neuralmagic @mgoin 

**BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE**

---

<details>
<!-- inside this <details> section, markdown rendering does not work, so we use raw html here. -->
<summary><b> PR Checklist (Click to Expand) </b></summary>

<p>Thank you for your contribution to vLLM! Before submitting the pull request, please ensure the PR meets the following criteria. This helps vLLM maintain the code quality and improve the efficiency of the review process.</p>

<h3>PR Title and Classification</h3>
<p>Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:</p>
<ul>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[Doc]</code> for documentation fixes and improvements.</li>
    <li><code>[Model]</code> for adding a new model or improving an existing model. Model name should appear in the title.</li>
    <li><code>[Frontend]</code> For changes on the vLLM frontend (e.g., OpenAI API server, <code>LLM</code> class, etc.) </li>
    <li><code>[Kernel]</code> for changes affecting CUDA kernels or other compute kernels.</li>
    <li><code>[Core]</code> for changes in the core vLLM logic (e.g., <code>LLMEngine</code>, <code>AsyncLLMEngine</code>, <code>Scheduler</code>, etc.)</li>
    <li><code>[Hardware][Vendor]</code> for hardware-specific changes. Vendor name should appear in the prefix (e.g., <code>[Hardware][AMD]</code>).</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If the PR spans more than one category, please include all relevant prefixes.</p>

<h3>Code Quality</h3>

<p>The PR need to meet the following code quality standards:</p>

<ul>
    <li>We adhere to <a href="https://google.github.io/styleguide/pyguide.html">Google Python style guide</a> and <a href="https://google.github.io/styleguide/cppguide.html">Google C++ style guide</a>.</li>
    <li>Pass all linter checks. Please use <a href="https://github.com/vllm-project/vllm/blob/main/format.sh"><code>format.sh</code></a> to format your code.</li>
    <li>The code need to be well-documented to ensure future contributors can easily understand the code.</li>
    <li>Include sufficient tests to ensure the project to stay correct and robust. This includes both unit tests and integration tests.</li>
    <li>Please add documentation to <code>docs/source/</code> if the PR modifies the user-facing behaviors of vLLM. It helps vLLM user understand and utilize the new features or changes.</li>
</ul>

<h3>Notes for Large Changes</h3>
<p>Please keep the changes as concise as possible. For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue (RFC) discussing the technical design and justification. Otherwise, we will tag it with <code>rfc-required</code> and might not go through the PR.</p>

<h3>What to Expect for the Reviews</h3>

<p>The goal of the vLLM team is to be a <i>transparent reviewing machine</i>. We would like to make the review process transparent and efficient and make sure no contributor feel confused or frustrated. However, the vLLM team is small, so we need to prioritize some PRs over others. Here is what you can expect from the review process: </p>

<ul>
    <li> After the PR is submitted, the PR will be assigned to a reviewer. Every reviewer will pick up the PRs based on their expertise and availability.</li>
    <li> After the PR is assigned, the reviewer will provide status update every 2-3 days. If the PR is not reviewed within 7 days, please feel free to ping the reviewer or the vLLM team.</li>
    <li> After the review, the reviewer will put an <code> action-required</code> label on the PR if there are changes required. The contributor should address the comments and ping the reviewer to re-review the PR.</li>
    <li> Please respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.
 </li>
</ul>

<h3>Thank You</h3>

<p> Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM. Your contributions make vLLM a great tool for everyone! </p>


</details>




## Full discussion

See [4118.md](./4118.md) and [4118.json](./4118.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/4118
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/4118.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `826d7def37329ce0d0359a6ef1b30c140d0739a774ba0e390649abf95f70a631`

Existing semantic notes in [4118.sem.json](./4118.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/quantization/test_fp8.py`
- `vllm/entrypoints/llm.py`
- `vllm/model_executor/layers/linear.py`
- `vllm/model_executor/layers/quantization/__init__.py`
- `vllm/model_executor/layers/quantization/fp8.py`
- `vllm/model_executor/model_loader/loader.py`
- `vllm/model_executor/model_loader/weight_utils.py`

### Diff

```diff
diff --git a/tests/quantization/test_fp8.py b/tests/quantization/test_fp8.py
new file mode 100644
index 000000000000..fa10e60de10a
--- /dev/null
+++ b/tests/quantization/test_fp8.py
@@ -0,0 +1,24 @@
+"""Tests whether FP8 computation is enabled correctly.
+
+Run `pytest tests/quantization/test_fp8.py --forked`.
+"""
+import pytest
+import torch
+
+from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
+from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
+
+capability = torch.cuda.get_device_capability()
+capability = capability[0] * 10 + capability[1]
+
+
+@pytest.mark.skipif(
+    capability < QUANTIZATION_METHODS["fp8"].get_min_capability(),
+    reason="FP8 is not supported on this GPU type.")
+def test_load_fp16_model(vllm_runner) -> None:
+    llm = vllm_runner("facebook/opt-125m", quantization="fp8")
+
+    model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model
+    fc1 = model.model.decoder.layers[0].fc1
+    assert isinstance(fc1.linear_method, Fp8LinearMethod)
+    assert fc1.weight.dtype == torch.float8_e4m3fn
diff --git a/vllm/entrypoints/llm.py b/vllm/entrypoints/llm.py
index 9e08c253dc53..961de5d5063f 100644
--- a/vllm/entrypoints/llm.py
+++ b/vllm/entrypoints/llm.py
@@ -42,10 +42,11 @@ class LLM:
             However, if the `torch_dtype` in the config is `float32`, we will
             use `float16` instead.
         quantization: The method used to quantize the model weights. Currently,
-            we support "awq", "gptq" and "squeezellm". If None, we first check
-            the `quantization_config` attribute in the model config file. If
-            that is None, we assume the model weights are not quantized and use
-            `dtype` to determine the data type of the weights.
+            we support "awq", "gptq", "squeezellm", and "fp8" (experimental).
+            If None, we first check the `quantization_config` attribute in the
+            model config file. If that is None, we assume the model weights are
+            not quantized and use `dtype` to determine the data type of
+            the weights.
         revision: The specific model version to use. It can be a branch name,
             a tag name, or a commit id.
         tokenizer_revision: The specific tokenizer version to use. It can be a
diff --git a/vllm/model_executor/layers/linear.py b/vllm/model_executor/layers/linear.py
index 3ca870742efc..d466d8807fc6 100644
--- a/vllm/model_executor/layers/linear.py
+++ b/vllm/model_executor/layers/linear.py
@@ -3,6 +3,7 @@
 
 import torch
 import torch.nn.functional as F
+from torch import nn
 from torch.nn.parameter import Parameter
 
 from vllm.distributed import (divide, get_tensor_model_parallel_rank,
@@ -48,6 +49,13 @@ def apply_weights(self,
         Expects create_weights to have been called before on the layer."""
         raise NotImplementedError
 
+    def process_weights_after_loading(self, layer: nn.Module) -> None:
+        """Process the weight after loading.
+
+        This can be used for example, to transpose weights for computation.
+        """
+        return
+
 
 class UnquantizedLinearMethod(LinearMethodBase):
     """Linear method without quantization.
diff --git a/vllm/model_executor/layers/quantization/__init__.py b/vllm/model_executor/layers/quantization/__init__.py
index a3b89a66469e..0344d6e4e3e4 100644
--- a/vllm/model_executor/layers/quantization/__init__.py
+++ b/vllm/model_executor/layers/quantization/__init__.py
@@ -3,12 +3,14 @@
 from vllm.model_executor.layers.quantization.awq import AWQConfig
 from vllm.model_executor.layers.quantization.base_config import (
     QuantizationConfig)
+from vllm.model_executor.layers.quantization.fp8 import FP8Config
 from vllm.model_executor.layers.quantization.gptq import GPTQConfig
 from vllm.model_executor.layers.quantization.marlin import MarlinConfig
 from vllm.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig
 
 QUANTIZATION_METHODS = {
     "awq": AWQConfig,
+    "fp8": FP8Config,
     "gptq": GPTQConfig,
     "squeezellm": SqueezeLLMConfig,
     "marlin": MarlinConfig,
diff --git a/vllm/model_executor/layers/quantization/fp8.py b/vllm/model_executor/layers/quantization/fp8.py
new file mode 100644
index 000000000000..9dc0e86e1243
--- /dev/null
+++ b/vllm/model_executor/layers/quantization/fp8.py
@@ -0,0 +1,138 @@
+from typing import Any, Dict, List, Optional
+
+import torch
+from torch.nn import Module
+from torch.nn.parameter import Parameter
+
+from vllm.model_executor.layers.linear import (LinearMethodBase,
+                                               set_weight_attrs)
+from vllm.model_executor.layers.quantization.base_config import (
+    QuantizationConfig)
+
+
+class FP8Config(QuantizationConfig):
+    """Config class for FP8."""
+
+    @classmethod
+    def get_name(cls) -> str:
+        return "fp8"
+
+    @classmethod
+    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
+        return [torch.bfloat16, torch.half]
+
+    @classmethod
+    def get_min_capability(cls) -> int:
+        # TODO: PyTorch 2.3.0+ is required to run FP8 on
+        # SM 89 (e.g. Ada) GPUs. Specifically, this PR has to
+        # be included: https://github.com/pytorch/pytorch/pull/118881
+        return 90
+
+    @classmethod
+    def get_config_filenames(cls) -> List[str]:
+        return []
+
+    @classmethod
+    def from_config(cls, config: Dict[str, Any]) -> "FP8Config":
+        return cls()
+
+    def get_linear_method(self) -> "Fp8LinearMethod":
+        return Fp8LinearMethod(self)
+
+    def get_scaled_act_names(self) -> List[str]:
+        return []
+
+
+class Fp8LinearMethod(LinearMethodBase):
+    """Linear method for FP8.
+    We now support common FP16/BF16 model checkpoints ONLY. The weight
+    scaling factor will be initialized after the model weights are loaded.
+
+    Limitations:
+    1. Only support per-tensor quantization due to torch._scaled_mm support.
+    2. Only support float8_e4m3fn data type due to the limitation of
+       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)
+       
+    Args:
+        quant_config: The quantization config.
+    """
+
+    def __init__(self, quant_config: FP8Config):
+        self.quant_config = quant_config
+
+    def create_weights(
+        self,
+        layer: torch.nn.Module,
+        input_size_per_partition: int,
+        output_size_per_partition: int,
+        input_size: int,
+        output_size: int,
+        params_dtype: torch.dtype,
+        **extra_weight_attrs,
+    ):
+        weight = Parameter(torch.empty(output_size_per_partition,
+                                       input_size_per_partition,
+                                       dtype=params_dtype),
+                           requires_grad=False)
+        layer.register_parameter("weight", weight)
+        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
+        set_weight_attrs(weight, extra_weight_attrs)
+
+        w_scale = Parameter(
+            torch.empty(1, dtype=torch.float32),
+            requires_grad=False,
+        )
+        layer.register_parameter("weight_scaling_factor", w_scale)
+
+    def process_weights_after_loading(self, layer: Module) -> None:
+        # Although the linear_method is propagated to all layers,
+        # only linear layers invoke "create_weights". So we check
+        # whether "weight_scaling_facor" is registered to determine
+        # whether the layer is a linear layer that requires quantization.
+        if not hasattr(layer, "weight_scaling_factor"):
+            return
+
+        qweight, weight_scale = per_tensor_quantize(layer.weight)
+        # torch._scaled_mm requires column-major in the second
+        # input (weight), so we transpose the quantized weight.
+        layer.weight = Parameter(qweight.t(), requires_grad=False)
+        layer.weight_scaling_factor.data.copy_(weight_scale)
+
+    def apply_weights(self,
+                      layer: torch.nn.Module,
+                      x: torch.Tensor,
+                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
+        qinput, x_scale = per_tensor_quantize(x)
+        output, _ = torch._scaled_mm(
+            qinput,
+            layer.weight,
+            out_dtype=x.dtype,
+            scale_a=x_scale,
+            scale_b=layer.weight_scaling_factor,
+            bias=bias,
+        )
+        return output
+
+
+def per_tensor_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
+    """Quantize a tensor using per-tensor static scaling factor.
+
+    Args:
+        tensor: The input tensor.
+    """
+    finfo = torch.finfo(torch.float8_e4m3fn)
+    # Calculate the scale as dtype max divided by absmax.
+    # Since .abs() creates a new tensor, we use aminmax to get
+    # the min and max first and then calculate the absmax.
+    min_val, max_val = tensor.aminmax()
+    amax = min_val.abs().max(max_val.abs())
+    scale = finfo.max / amax.clamp(min=1e-12)
+    # scale and clamp the tensor to bring it to
+    # the representative range of float8 data type
+    # (as default cast is unsaturated)
+    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
+    # Return both float8 data and the inverse scale (as float),
+    # as both required as inputs to torch._scaled_mm
+    qweight = qweight.to(torch.float8_e4m3fn)
+    scale = scale.float().reciprocal()
+    return qweight, scale
diff --git a/vllm/model_executor/model_loader/loader.py b/vllm/model_executor/model_loader/loader.py
index 3b1d125ef8a6..6c8cb2935f37 100644
--- a/vllm/model_executor/model_loader/loader.py
+++ b/vllm/model_executor/model_loader/loader.py
@@ -228,6 +228,10 @@ def load_model(self, *, model_config: ModelConfig,
                                                model,
                                                "fall_back_to_pt_during_load",
                                                True)), )
+            for _, module in model.named_modules():
+                linear_method = getattr(module, "linear_method", None)
+                if linear_method is not None:
+                    linear_method.process_weights_after_loading(module)
         return model.eval()
 
 
diff --git a/vllm/model_executor/model_loader/weight_utils.py b/vllm/model_executor/model_loader/weight_utils.py
index 1798db013686..9995f2afe3cf 100644
--- a/vllm/model_executor/model_loader/weight_utils.py
+++ b/vllm/model_executor/model_loader/weight_utils.py
@@ -134,11 +134,18 @@ def get_quant_config(model_config: ModelConfig,
                                           tqdm_class=DisabledTqdm)
     else:
         hf_folder = model_name_or_path
+
+    possible_config_filenames = quant_cls.get_config_filenames()
+
+    # If the quantization config is not found, use the default config.
+    if not possible_config_filenames:
+        return quant_cls()
+
     config_files = glob.glob(os.path.join(hf_folder, "*.json"))
 
     quant_config_files = [
         f for f in config_files if any(
-            f.endswith(x) for x in quant_cls.get_config_filenames())
+            f.endswith(x) for x in possible_config_filenames)
     ]
     if len(quant_config_files) == 0:
         raise ValueError(
```
