# PR #18745: [Core] Support inplace model weights loading (#18745)

# Issue

## Original PR description (captured source)

In RL, we sometimes initialize a model with dummy weights at the beginning of rollout. At a later stage, real weights are loaded with a `load_model` RPC call. Currently this is not well supported because `load_model` API always re-initialize the model. Three main changes in this PR:
- Model loader refactoring: Most model loaders now share a common `load_model` function defined in `BaseModelLoader`
- Add `load_weights` API on `BaseModelLoader`
- In V1 GPU model runner, if a model has already been initialized, call `load_weights` instead of `load_model`

E2E test:
```
# Save https://gist.github.com/foreverlms/7e3ead8118db578bcff454256336e391
torchrun  --standalone  --nnodes=1   --nproc-per-node=4 dummy_load_test.py
```

Before this PR:
> ValueError: Duplicate layer name: model.layers.0.self_attn.attn
Full log: https://gist.github.com/22quinn/6819a91091f3be252e5fba231716e59a

After this PR:
> [gpu_model_runner.py:1537] Model was already initialized. Loading weights inplace...

cc @jeejeelee @DarkLight1337 @houseroad 

## Full discussion

See [18745.md](./18745.md) and [18745.json](./18745.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/18745
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/18745.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `6d8d57e26f445e741f664fa775bd0bdf982090b8877444ced50aaa75f70de4cf`

Existing semantic notes in [18745.sem.json](./18745.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/tensorizer_loader/test_tensorizer.py`
- `tests/v1/worker/test_gpu_model_runner.py`
- `vllm/model_executor/model_loader/base_loader.py`
- `vllm/model_executor/model_loader/bitsandbytes_loader.py`
- `vllm/model_executor/model_loader/default_loader.py`
- `vllm/model_executor/model_loader/dummy_loader.py`
- `vllm/model_executor/model_loader/gguf_loader.py`
- `vllm/model_executor/model_loader/runai_streamer_loader.py`
- `vllm/model_executor/model_loader/sharded_state_loader.py`
- `vllm/model_executor/model_loader/tensorizer.py`
- `vllm/model_executor/model_loader/tensorizer_loader.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/tpu_model_runner.py`

### Diff

```diff
diff --git a/tests/tensorizer_loader/test_tensorizer.py b/tests/tensorizer_loader/test_tensorizer.py
index b6286e148397..747ec56ad629 100644
--- a/tests/tensorizer_loader/test_tensorizer.py
+++ b/tests/tensorizer_loader/test_tensorizer.py
@@ -4,7 +4,6 @@
 import os
 import pathlib
 import subprocess
-from unittest.mock import MagicMock, patch
 
 import pytest
 import torch
@@ -16,7 +15,6 @@
 from vllm.model_executor.model_loader.tensorizer import (TensorizerConfig,
                                                          TensorSerializer,
                                                          is_vllm_tensorized,
-                                                         load_with_tensorizer,
                                                          open_stream,
                                                          tensorize_vllm_model)
 # yapf: enable
@@ -61,21 +59,6 @@ def write_keyfile(keyfile_path: str):
         f.write(encryption_params.key)
 
 
-@patch('vllm.model_executor.model_loader.tensorizer.TensorizerAgent')
-def test_load_with_tensorizer(mock_agent, tensorizer_config):
-    mock_linear_method = MagicMock()
-    mock_agent_instance = mock_agent.return_value
-    mock_agent_instance.deserialize.return_value = MagicMock()
-
-    result = load_with_tensorizer(tensorizer_config,
-                                  quant_method=mock_linear_method)
-
-    mock_agent.assert_called_once_with(tensorizer_config,
-                                       quant_method=mock_linear_method)
-    mock_agent_instance.deserialize.assert_called_once()
-    assert result == mock_agent_instance.deserialize.return_value
-
-
 @pytest.mark.skipif(not is_curl_installed(), reason="cURL is not installed")
 def test_can_deserialize_s3(vllm_runner):
     model_ref = "EleutherAI/pythia-1.4b"
diff --git a/tests/v1/worker/test_gpu_model_runner.py b/tests/v1/worker/test_gpu_model_runner.py
index c38eb486646f..6ba6d1f6f131 100644
--- a/tests/v1/worker/test_gpu_model_runner.py
+++ b/tests/v1/worker/test_gpu_model_runner.py
@@ -94,6 +94,9 @@ def model_runner():
     return runner
 
 
+model_runner_2 = model_runner
+
+
 def _schedule_new_request(*req_ids: str) -> SchedulerOutput:
     new_reqs = []
     num_scheduled_tokens = {}
@@ -366,3 +369,18 @@ def rnd_stride_order():
         assert all(kv.is_contiguous() for kv in model_runner.kv_caches)
     else:
         assert all(not kv.is_contiguous() for kv in model_runner.kv_caches)
+
+
+def test_load_model_weights_inplace(dist_init, model_runner, model_runner_2):
+    # In this test, model_runner loads model + weights in one go, while
+    # model_runner_2 loads dummy weights first then load real weights inplace
+    model_runner.load_model()
+    original_load_format = model_runner_2.load_config.load_format
+    model_runner_2.load_config.load_format = "dummy"
+    model_runner_2.load_model()  # Initial model loading with dummy weights
+    assert str(model_runner.get_model().state_dict()) != str(
+        model_runner_2.get_model().state_dict())
+    model_runner_2.load_config.load_format = original_load_format
+    model_runner_2.load_model()  # Load real weights inplace
+    assert str(model_runner.get_model().state_dict()) == str(
+        model_runner_2.get_model().state_dict())
diff --git a/vllm/model_executor/model_loader/base_loader.py b/vllm/model_executor/model_loader/base_loader.py
index 010dd515784a..d619d9f25e08 100644
--- a/vllm/model_executor/model_loader/base_loader.py
+++ b/vllm/model_executor/model_loader/base_loader.py
@@ -1,9 +1,12 @@
 # SPDX-License-Identifier: Apache-2.0
 from abc import ABC, abstractmethod
 
+import torch
 import torch.nn as nn
 
 from vllm.config import LoadConfig, ModelConfig, VllmConfig
+from vllm.model_executor.model_loader.utils import (
+    initialize_model, process_weights_after_loading, set_default_torch_dtype)
 
 
 class BaseModelLoader(ABC):
@@ -18,7 +21,22 @@ def download_model(self, model_config: ModelConfig) -> None:
         raise NotImplementedError
 
     @abstractmethod
-    def load_model(self, *, vllm_config: VllmConfig,
+    def load_weights(self, model: nn.Module,
+                     model_config: ModelConfig) -> None:
+        """Load weights into a model. This standalone API allows 
+        inplace weights loading for an already-initialized model"""
+        raise NotImplementedError
+
+    def load_model(self, vllm_config: VllmConfig,
                    model_config: ModelConfig) -> nn.Module:
         """Load a model with the given configurations."""
-        raise NotImplementedError
+        device_config = vllm_config.device_config
+        target_device = torch.device(device_config.device)
+        with set_default_torch_dtype(model_config.dtype):
+            with target_device:
+                model = initialize_model(vllm_config=vllm_config,
+                                         model_config=model_config)
+            # Quantization does not happen in `load_weights` but after it
+            self.load_weights(model, model_config)
+            process_weights_after_loading(model, model_config, target_device)
+        return model.eval()
diff --git a/vllm/model_executor/model_loader/bitsandbytes_loader.py b/vllm/model_executor/model_loader/bitsandbytes_loader.py
index 8996ea266ac4..3df835a93896 100644
--- a/vllm/model_executor/model_loader/bitsandbytes_loader.py
+++ b/vllm/model_executor/model_loader/bitsandbytes_loader.py
@@ -14,7 +14,7 @@
 from torch import nn
 from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
 
-from vllm.config import LoadConfig, ModelConfig, VllmConfig
+from vllm.config import LoadConfig, ModelConfig
 from vllm.distributed import (get_tensor_model_parallel_rank,
                               get_tensor_model_parallel_world_size)
 # yapf: enable
@@ -28,7 +28,6 @@
                                                RowParallelLinear)
 from vllm.model_executor.model_loader.base_loader import BaseModelLoader
 from vllm.model_executor.model_loader.utils import (ParamMapping,
-                                                    initialize_model,
                                                     set_default_torch_dtype)
 from vllm.model_executor.model_loader.weight_utils import (
     download_safetensors_index_file_from_hf, download_weights_from_hf,
@@ -408,8 +407,7 @@ def _get_bnb_target_modules(self, model: nn.Module) -> None:
                 ), "vllm currently does not support BNB quantization for"
         f" {type(model).__name__}"
 
-    def _load_weights(self, model_config: ModelConfig,
-                      model: nn.Module) -> None:
+    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
         if not hasattr(model, "load_weights"):
             raise AttributeError(
                 "The required method 'load_weights' is not defined in class"
@@ -568,15 +566,3 @@ def _load_weights(self, model_config: ModelConfig,
 
     def download_model(self, model_config: ModelConfig) -> None:
         self._prepare_weights(model_config.model, model_config.revision)
-
-    def load_model(self, vllm_config: VllmConfig,
-                   model_config: ModelConfig) -> nn.Module:
-        device_config = vllm_config.device_config
-        with set_default_torch_dtype(model_config.dtype):
-            with torch.device(device_config.device):
-
-                model = initialize_model(vllm_config=vllm_config)
-
-                self._load_weights(model_config, model)
-
-        return model.eval()
diff --git a/vllm/model_executor/model_loader/default_loader.py b/vllm/model_executor/model_loader/default_loader.py
index 29a6e0af4bc6..6946627a54d2 100644
--- a/vllm/model_executor/model_loader/default_loader.py
+++ b/vllm/model_executor/model_loader/default_loader.py
@@ -12,11 +12,9 @@
 from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
 
 from vllm import envs
-from vllm.config import LoadConfig, LoadFormat, ModelConfig, VllmConfig
+from vllm.config import LoadConfig, LoadFormat, ModelConfig
 from vllm.logger import init_logger
 from vllm.model_executor.model_loader.base_loader import BaseModelLoader
-from vllm.model_executor.model_loader.utils import (
-    initialize_model, process_weights_after_loading, set_default_torch_dtype)
 from vllm.model_executor.model_loader.weight_utils import (
     download_safetensors_index_file_from_hf, download_weights_from_hf,
     fastsafetensors_weights_iterator, filter_duplicate_safetensors_files,
@@ -264,32 +262,20 @@ def download_model(self, model_config: ModelConfig) -> None:
                               fall_back_to_pt=True,
                               allow_patterns_overrides=None)
 
-    def load_model(self, vllm_config: VllmConfig,
-                   model_config: ModelConfig) -> nn.Module:
-        device_config = vllm_config.device_config
-        target_device = torch.device(device_config.device)
-        with set_default_torch_dtype(model_config.dtype):
-            with target_device:
-                model = initialize_model(vllm_config=vllm_config,
-                                         model_config=model_config)
-
-            weights_to_load = {name for name, _ in model.named_parameters()}
-            loaded_weights = model.load_weights(
-                self.get_all_weights(model_config, model))
-            self.counter_after_loading_weights = time.perf_counter()
-            logger.info(
-                "Loading weights took %.2f seconds",
-                self.counter_after_loading_weights -
-                self.counter_before_loading_weights)
-            # We only enable strict check for non-quantized models
-            # that have loaded weights tracking currently.
-            if model_config.quantization is None and loaded_weights is not None:
-                weights_not_loaded = weights_to_load - loaded_weights
-                if weights_not_loaded:
-                    raise ValueError(
-                        "Following weights were not initialized from "
-                        f"checkpoint: {weights_not_loaded}")
-
-            process_weights_after_loading(model, model_config, target_device)
-
-        return model.eval()
+    def load_weights(self, model: nn.Module,
+                     model_config: ModelConfig) -> None:
+        weights_to_load = {name for name, _ in model.named_parameters()}
+        loaded_weights = model.load_weights(
+            self.get_all_weights(model_config, model))
+        self.counter_after_loading_weights = time.perf_counter()
+        logger.info(
+            "Loading weights took %.2f seconds",
+            self.counter_after_loading_weights -
+            self.counter_before_loading_weights)
+        # We only enable strict check for non-quantized models
+        # that have loaded weights tracking currently.
+        if model_config.quantization is None and loaded_weights is not None:
+            weights_not_loaded = weights_to_load - loaded_weights
+            if weights_not_loaded:
+                raise ValueError("Following weights were not initialized from "
+                                 f"checkpoint: {weights_not_loaded}")
diff --git a/vllm/model_executor/model_loader/dummy_loader.py b/vllm/model_executor/model_loader/dummy_loader.py
index 0e2f0be1ec26..64fa2be76d08 100644
--- a/vllm/model_executor/model_loader/dummy_loader.py
+++ b/vllm/model_executor/model_loader/dummy_loader.py
@@ -1,11 +1,8 @@
 # SPDX-License-Identifier: Apache-2.0
-import torch
 import torch.nn as nn
 
-from vllm.config import LoadConfig, ModelConfig, VllmConfig
+from vllm.config import LoadConfig, ModelConfig
 from vllm.model_executor.model_loader.base_loader import BaseModelLoader
-from vllm.model_executor.model_loader.utils import (
-    initialize_model, process_weights_after_loading, set_default_torch_dtype)
 from vllm.model_executor.model_loader.weight_utils import (
     initialize_dummy_weights)
 
@@ -22,16 +19,8 @@ def __init__(self, load_config: LoadConfig):
     def download_model(self, model_config: ModelConfig) -> None:
         pass  # Nothing to download
 
-    def load_model(self, vllm_config: VllmConfig,
-                   model_config: ModelConfig) -> nn.Module:
-        device_config = vllm_config.device_config
-        target_device = torch.device(device_config.device)
-        with set_default_torch_dtype(model_config.dtype):
-            with target_device:
-                model = initialize_model(vllm_config=vllm_config)
-            # NOTE(woosuk): For accurate performance evaluation, we assign
-            # random values to the weights.
-            initialize_dummy_weights(model)
-
-            process_weights_after_loading(model, model_config, target_device)
-        return model.eval()
+    def load_weights(self, model: nn.Module,
+                     model_config: ModelConfig) -> None:
+        # NOTE(woosuk): For accurate performance evaluation, we assign
+        # random values to the weights.
+        initialize_dummy_weights(model)
diff --git a/vllm/model_executor/model_loader/gguf_loader.py b/vllm/model_executor/model_loader/gguf_loader.py
index 806004bf9604..1eac504227e2 100644
--- a/vllm/model_executor/model_loader/gguf_loader.py
+++ b/vllm/model_executor/model_loader/gguf_loader.py
@@ -92,6 +92,13 @@ def _get_weights_iterator(
     def download_model(self, model_config: ModelConfig) -> None:
         self._prepare_weights(model_config.model)
 
+    def load_weights(self, model: nn.Module,
+                     model_config: ModelConfig) -> None:
+        local_model_path = self._prepare_weights(model_config.model)
+        gguf_weights_map = self._get_gguf_weights_map(model_config)
+        model.load_weights(
+            self._get_weights_iterator(local_model_path, gguf_weights_map))
+
     def load_model(self, vllm_config: VllmConfig,
                    model_config: ModelConfig) -> nn.Module:
         device_config = vllm_config.device_config
@@ -106,8 +113,7 @@ def load_model(self, vllm_config: VllmConfig,
         with set_default_torch_dtype(model_config.dtype):
             with target_device:
                 model = initialize_model(vllm_config=vllm_config)
-            model.load_weights(
-                self._get_weights_iterator(local_model_path, gguf_weights_map))
+            self.load_weights(model, model_config)
 
             process_weights_after_loading(model, model_config, target_device)
         return model
diff --git a/vllm/model_executor/model_loader/runai_streamer_loader.py b/vllm/model_executor/model_loader/runai_streamer_loader.py
index 9f1022c25925..a39e26c6da50 100644
--- a/vllm/model_executor/model_loader/runai_streamer_loader.py
+++ b/vllm/model_executor/model_loader/runai_streamer_loader.py
@@ -9,10 +9,8 @@
 from torch import nn
 from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
 
-from vllm.config import LoadConfig, ModelConfig, VllmConfig
+from vllm.config import LoadConfig, ModelConfig
 from vllm.model_executor.model_loader.base_loader import BaseModelLoader
-from vllm.model_executor.model_loader.utils import (
-    initialize_model, process_weights_after_loading, set_default_torch_dtype)
 from vllm.model_executor.model_loader.weight_utils import (
     download_safetensors_index_file_from_hf, download_weights_from_hf,
     runai_safetensors_weights_iterator)
@@ -100,21 +98,11 @@ def download_model(self, model_config: ModelConfig) -> None:
         """Download model if necessary"""
         self._prepare_weights(model_config.model, model_config.revision)
 
-    def load_model(self, vllm_config: VllmConfig,
-                   model_config: ModelConfig) -> nn.Module:
-        """Perform streaming of the model to destination"""
-        device_config = vllm_config.device_config
-        target_device = torch.device(device_config.device)
-        with set_default_torch_dtype(model_config.dtype):
-            with target_device:
-                model = initialize_model(vllm_config=vllm_config)
-
-            model_weights = model_config.model
-            if hasattr(model_config, "model_weights"):
-                model_weights = model_config.model_weights
-            model.load_weights(
-                self._get_weights_iterator(model_weights,
-                                           model_config.revision))
-
-            process_weights_after_loading(model, model_config, target_device)
-        return model.eval()
+    def load_weights(self, model: nn.Module,
+                     model_config: ModelConfig) -> None:
+        """Load weights into a model."""
+        model_weights = model_config.model
+        if hasattr(model_config, "model_weights"):
+            model_weights = model_config.model_weights
+        model.load_weights(
+            self._get_weights_iterator(model_weights, model_config.revision))
diff --git a/vllm/model_executor/model_loader/sharded_state_loader.py b/vllm/model_executor/model_loader/sharded_state_loader.py
index 78bca89f0015..b5a5031bb6f9 100644
--- a/vllm/model_executor/model_loader/sharded_state_loader.py
+++ b/vllm/model_executor/model_loader/sharded_state_loader.py
@@ -9,11 +9,9 @@
 import torch
 from torch import nn
 
-from vllm.config import LoadConfig, ModelConfig, VllmConfig
+from vllm.config import LoadConfig, ModelConfig
 from vllm.logger import init_logger
 from vllm.model_executor.model_loader.base_loader import BaseModelLoader
-from vllm.model_executor.model_loader.utils import (
-    initialize_model, process_weights_after_loading, set_default_torch_dtype)
 from vllm.model_executor.model_loader.weight_utils import (
     download_weights_from_hf, runai_safetensors_weights_iterator)
 from vllm.transformers_utils.s3_utils import glob as s3_glob
@@ -100,11 +98,8 @@ def _prepare_weights(self, model_name_or_path: str,
     def download_model(self, model_config: ModelConfig) -> None:
         self._prepare_weights(model_config.model, model_config.revision)
 
-    def load_model(self, vllm_config: VllmConfig,
-                   model_config: ModelConfig) -> nn.Module:
-        device_config = vllm_config.device_config
-        target_device = torch.device(device_config.device)
-
+    def load_weights(self, model: nn.Module,
+                     model_config: ModelConfig) -> None:
         from vllm.distributed import get_tensor_model_parallel_rank
 
         model_weights = model_config.model
@@ -112,53 +107,47 @@ def load_model(self, vllm_config: VllmConfig,
             model_weights = model_config.model_weights
         local_model_path = model_weights
 
-        with set_default_torch_dtype(model_config.dtype):
-            with target_device:
-                model = initialize_model(vllm_config=vllm_config)
-                process_weights_after_loading(model, model_config,
-                                              target_device)
-            rank = get_tensor_model_parallel_rank()
-            pattern = os.path.join(
-                local_model_path,
-                self.pattern.format(rank=rank, part="*"),
-            )
-
-            filepaths = []
-            if is_s3(local_model_path):
-                file_pattern = f"*{self.pattern.format(rank=rank, part=' * ')}"
-                filepaths = s3_glob(path=local_model_path,
-                                    allow_pattern=[file_pattern])
-            else:
-                filepaths = glob.glob(pattern)
-            if not filepaths:
-                # TODO: support un-sharded checkpoints too
-                raise ValueError(
-                    f"Could not find checkpoint files '{pattern}', only "
-                    f"pre-sharded checkpoints are currently supported!")
-            state_dict = self._filter_subtensors(model.state_dict())
-            for key, tensor in self.iterate_over_files(filepaths):
-                # If loading with LoRA enabled, additional padding may
-                # be added to certain parameters. We only load into a
-                # narrowed view of the parameter data.
-                param_data = state_dict[key].data
-                param_shape = state_dict[key].shape
-                for dim, size in enumerate(tensor.shape):
-                    if size < param_shape[dim]:
-                        param_data = param_data.narrow(dim, 0, size)
-                if tensor.shape != param_shape:
-                    logger.warning(
-                        "loading tensor of shape %s into "
-                        "parameter '%s' of shape %s",
-                        tensor.shape,
-                        key,
-                        param_shape,
-                    )
-                param_data.copy_(tensor)
-                state_dict.pop(key)
-            if state_dict:
-                raise ValueError(
-                    f"Missing keys {tuple(state_dict)} in loaded state!")
-        return model.eval()
+        rank = get_tensor_model_parallel_rank()
+        pattern = os.path.join(
+            local_model_path,
+            self.pattern.format(rank=rank, part="*"),
+        )
+
+        filepaths = []
+        if is_s3(local_model_path):
+            file_pattern = f"*{self.pattern.format(rank=rank, part=' * ')}"
+            filepaths = s3_glob(path=local_model_path,
+                                allow_pattern=[file_pattern])
+        else:
+            filepaths = glob.glob(pattern)
+        if not filepaths:
+            # TODO: support un-sharded checkpoints too
+            raise ValueError(
+                f"Could not find checkpoint files '{pattern}', only "
+                f"pre-sharded checkpoints are currently supported!")
+        state_dict = self._filter_subtensors(model.state_dict())
+        for key, tensor in self.iterate_over_files(filepaths):
+            # If loading with LoRA enabled, additional padding may
+            # be added to certain parameters. We only load into a
+            # narrowed view of the parameter data.
+            param_data = state_dict[key].data
+            param_shape = state_dict[key].shape
+            for dim, size in enumerate(tensor.shape):
+                if size < param_shape[dim]:
+                    param_data = param_data.narrow(dim, 0, size)
+            if tensor.shape != param_shape:
+                logger.warning(
+                    "loading tensor of shape %s into "
+                    "parameter '%s' of shape %s",
+                    tensor.shape,
+                    key,
+                    param_shape,
+                )
+            param_data.copy_(tensor)
+            state_dict.pop(key)
+        if state_dict:
+            raise ValueError(
+                f"Missing keys {tuple(state_dict)} in loaded state!")
 
     def iterate_over_files(
             self, paths) -> Generator[tuple[str, torch.Tensor], None, None]:
diff --git a/vllm/model_executor/model_loader/tensorizer.py b/vllm/model_executor/model_loader/tensorizer.py
index 4c4502284a6a..90c0bdf08ef8 100644
--- a/vllm/model_executor/model_loader/tensorizer.py
+++ b/vllm/model_executor/model_loader/tensorizer.py
@@ -21,7 +21,8 @@
 from transformers import PretrainedConfig
 
 import vllm.envs as envs
-from vllm.config import ModelConfig, ParallelConfig, set_current_vllm_config
+from vllm.config import (ModelConfig, ParallelConfig, VllmConfig,
+                         set_current_vllm_config)
 from vllm.engine.arg_utils import EngineArgs
 from vllm.logger import init_logger
 from vllm.model_executor.layers.vocab_parallel_embedding import (
@@ -208,12 +209,6 @@ def open_stream(self, tensorizer_args: Optional["TensorizerArgs"] = None):
                            **tensorizer_args.stream_params)
 
 
-def load_with_tensorizer(tensorizer_config: TensorizerConfig,
-                         **extra_kwargs) -> nn.Module:
-    tensorizer = TensorizerAgent(tensorizer_config, **extra_kwargs)
-    return tensorizer.deserialize()
-
-
 @dataclass
 class TensorizerArgs:
     tensorizer_uri: Union[io.BufferedIOBase, io.RawIOBase, BinaryIO, str,
@@ -366,100 +361,72 @@ def from_cli_args(cls, args: argparse.Namespace) -> "TensorizerArgs":
         return tensorizer_args
 
 
-class TensorizerAgent:
-    """
-    A class for performing tensorizer deserializations specifically for
-    vLLM models using plaid_mode. Uses TensorizerArgs to configure the
-    behavior of the TensorDeserializer when loading tensors from a serialized
-    model. For deserializations of HuggingFace models, TensorDeserializer is
-    instead used as an iterator directly in the func hf_model_weights_iterator
-    in vllm/model_executor/model_loader/weight_utils.py
-    """
-
-    def __init__(self, tensorizer_config: TensorizerConfig, vllm_config):
-        self.tensorizer_config = tensorizer_config
-        self.tensorizer_args = (
-            self.tensorizer_config._construct_tensorizer_args())
-        self.vllm_config = vllm_config
-        self.model = self._init_model()
-
-    def _init_model(self):
-        assert self.tensorizer_config.hf_config is not None
-        model_args = self.tensorizer_config.hf_config
-        model_args.torch_dtype = self.tensorizer_config.dtype
-        assert self.tensorizer_config.model_class is not None
-        # TODO: Do we need to consider old-style model class?
-        with meta_tensor_mode(), set_current_vllm_config(self.vllm_config,
-                                                         check_compile=True):
-            return self.tensorizer_config.model_class(
-                vllm_config=self.vllm_config)
-
-    def _resize_lora_embeddings(self):
-        """Modify LoRA embedding layers to use bigger tensors
-        to allow for adapter added tokens."""
-        for child in self.model.modules():
-            if (isinstance(child, VocabParallelEmbedding)
-                    and child.weight.shape[0]
-                    < child.num_embeddings_per_partition):
-                new_weight = torch.empty(child.num_embeddings_per_partition,
-                                         child.embedding_dim,
-                                         dtype=child.weight.dtype,
-                                         device=child.weight.device)
-                new_weight[:child.weight.shape[0]].copy_(child.weight.data)
-                new_weight[child.weight.shape[0]:].fill_(0)
-                child.weight.data = new_weight
-
-    def _check_tensors_on_meta_device(self):
-        for tensor in self.model.state_dict().values():
-            if tensor.device.type == 'meta':
-                raise ValueError(
-                    "The serialized model contains tensors on the meta device,"
-                    " indicating that some tensors were not loaded properly."
-                    " Please check that the parameters of the model being"
-                    " specified match that of the serialized model, such as"
-                    " its quantization.")
-
-    def deserialize(self):
-        """
-        Deserialize the model using the TensorDeserializer. This method is
-        specifically for vLLM models using tensorizer's plaid_mode.
-
-        The deserializer makes use of tensorizer_args.stream_params
-        to configure the behavior of the stream when loading tensors from a
-        serialized model. The deserializer_params are used to configure the
-        behavior of the TensorDeserializer when loading tensors themselves.
-        Documentation on these params can be found in TensorizerArgs
-
-        Returns:
-            nn.Module: The deserialized model.
-        """
-        before_mem = get_mem_usage()
-        start = time.perf_counter()
-        with _read_stream(
-                self.tensorizer_config.tensorizer_uri,
-                **self.tensorizer_args.stream_params
-        ) as stream, TensorDeserializer(
+def _check_tensors_on_meta_device(model: nn.Module) -> None:
+    for tensor in model.state_dict().values():
+        if tensor.device.type == 'meta':
+            raise ValueError(
+                "The serialized model contains tensors on the meta device,"
+                " indicating that some tensors were not loaded properly."
+                " Please check that the parameters of the model being"
+                " specified match that of the serialized model, such as"
+                " its quantization.")
+
+
+def _resize_lora_embeddings(model: nn.Module):
+    """Modify LoRA embedding layers to use bigger tensors
+    to allow for adapter added tokens."""
+    for child in model.modules():
+        if (isinstance(child, VocabParallelEmbedding) and child.weight.shape[0]
+                < child.num_embeddings_per_partition):
+            new_weight = torch.empty(child.num_embeddings_per_partition,
+                                     child.embedding_dim,
+                                     dtype=child.weight.dtype,
+                                     device=child.weight.device)
+            new_weight[:child.weight.shape[0]].copy_(child.weight.data)
+            new_weight[child.weight.shape[0]:].fill_(0)
+            child.weight.data = new_weight
+
+
+def init_tensorizer_model(tensorizer_config: TensorizerConfig,
+                          vllm_config: VllmConfig) -> nn.Module:
+    assert tensorizer_config.hf_config is not None
+    model_args = tensorizer_config.hf_config
+    model_args.torch_dtype = tensorizer_config.dtype
+    assert tensorizer_config.model_class is not None
+    # TODO: Do we need to consider old-style model class?
+    with meta_tensor_mode(), set_current_vllm_config(vllm_config,
+                                                     check_compile=True):
+        return tensorizer_config.model_class(vllm_config=vllm_config)
+
+
+def deserialize_tensorizer_model(model: nn.Module,
+                                 tensorizer_config: TensorizerConfig) -> None:
+    tensorizer_args = tensorizer_config._construct_tensorizer_args()
+    before_mem = get_mem_usage()
+    start = time.perf_counter()
+    with _read_stream(
+            tensorizer_config.tensorizer_uri,
+            **tensorizer_args.stream_params) as stream, TensorDeserializer(
                 stream,
-                dtype=self.tensorizer_config.dtype,
+                dtype=tensorizer_config.dtype,
                 device=f'cuda:{torch.cuda.current_device()}',
-                **self.tensorizer_args.deserializer_params) as deserializer:
-            deserializer.load_into_module(self.model)
-            end = time.perf_counter()
-
-        total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
-        duration = end - start
-        per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
-        after_mem = get_mem_usage()
-        deserializer.close()
-        logger.info("Deserialized %s in %0.2fs, %s/s", total_bytes_str,
-                    end - start, per_second)
-        logger.info("Memory usage before: %s", before_mem)
-        logger.info("Memory usage after: %s", after_mem)
-
-        self._check_tensors_on_meta_device()
-        self._resize_lora_embeddings()
-        del self.model.vllm_tensorized_marker
-        return self.model.eval()
+                **tensorizer_args.deserializer_params) as deserializer:
+        deserializer.load_into_module(model)
+        end = time.perf_counter()
+
+    total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
+    duration = end - start
+    per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
+    after_mem = get_mem_usage()
+    deserializer.close()
+    logger.info("Deserialized %s in %0.2fs, %s/s", total_bytes_str,
+                end - start, per_second)
+    logger.info("Memory usage before: %s", before_mem)
+    logger.info("Memory usage after: %s", after_mem)
+
+    _check_tensors_on_meta_device(model)
+    _resize_lora_embeddings(model)
+    del model.vllm_tensorized_marker
 
 
 def tensorizer_weights_iterator(
diff --git a/vllm/model_executor/model_loader/tensorizer_loader.py b/vllm/model_executor/model_loader/tensorizer_loader.py
index 2afe2b59e2f9..1923e040af38 100644
--- a/vllm/model_executor/model_loader/tensorizer_loader.py
+++ b/vllm/model_executor/model_loader/tensorizer_loader.py
@@ -11,8 +11,8 @@
 from vllm.logger import init_logger
 from vllm.model_executor.model_loader.base_loader import BaseModelLoader
 from vllm.model_executor.model_loader.tensorizer import (
-    TensorizerConfig, is_vllm_tensorized, load_with_tensorizer,
-    serialize_vllm_model, tensorizer_weights_iterator)
+    TensorizerConfig, deserialize_tensorizer_model, init_tensorizer_model,
+    is_vllm_tensorized, serialize_vllm_model, tensorizer_weights_iterator)
 from vllm.model_executor.model_loader.utils import (get_model_architecture,
                                                     initialize_model,
                                                     set_default_torch_dtype)
@@ -61,38 +61,34 @@ def _load_model_serialized_cpu(
             model.load_weights(self._get_weights_iterator())
         return model.eval()
 
-    def _load_model_serialized(
-        self,
-        vllm_config: VllmConfig,
-    ) -> nn.Module:
-        """Load a serialized model with tensorizer.
-
-        Expects a vLLM-tensorized model. See the
-        examples/others/tensorize_vllm_model.py example script
-        for serializing vLLM models."""
-
-        device_config = vllm_config.device_config
-        model_config = vllm_config.model_config
-
-        with set_default_torch_dtype(model_config.dtype):
-            with torch.device(device_config.device):
-                model_class = get_model_architecture(model_config)[0]
-
-                tensorizer_config = copy.copy(self.tensorizer_config)
-                tensorizer_config.model_class = model_class
-                tensorizer_config.hf_config = model_config.hf_config
-                tensorizer_config.dtype = model_config.dtype
-
-                model = load_with_tensorizer(tensorizer_config,
-                                             vllm_config=vllm_config)
-        return model.eval()
-
     def download_model(self, model_config: ModelConfig) -> None:
         self.tensorizer_config.verify_with_model_config(model_config)
 
         with self.tensorizer_config.open_stream():
             pass
 
+    def _patch_tensorizer_config(
+            self, model_config: ModelConfig) -> TensorizerConfig:
+        model_class = get_model_architecture(model_config)[0]
+        tensorizer_config = copy.copy(self.tensorizer_config)
+        tensorizer_config.model_class = model_class
+        tensorizer_config.hf_config = model_config.hf_config
+        tensorizer_config.dtype = model_config.dtype
+        return tensorizer_config
+
+    def load_weights(self, model: nn.Module,
+                     model_config: ModelConfig) -> None:
+        """Load serialized model weights with tensorizer.
+
+        Expects a vLLM-tensorized model. See the
+        examples/others/tensorize_vllm_model.py example script
+        for serializing vLLM models."""
+        if is_vllm_tensorized(self.tensorizer_config):
+            tensorizer_config = self._patch_tensorizer_config(model_config)
+            deserialize_tensorizer_model(model, tensorizer_config)
+        else:
+            model.load_weights(self._get_weights_iterator())
+
     def load_model(self, vllm_config: VllmConfig,
                    model_config: ModelConfig) -> nn.Module:
         parallel_config = vllm_config.parallel_config
@@ -106,7 +102,11 @@ def load_model(self, vllm_config: VllmConfig,
                 get_tensor_model_parallel_rank())
 
         if is_vllm_tensorized(self.tensorizer_config):
-            return self._load_model_serialized(vllm_config=vllm_config)
+            tensorizer_config = self._patch_tensorizer_config(model_config)
+            model = init_tensorizer_model(tensorizer_config=tensorizer_config,
+                                          vllm_config=vllm_config)
+            self.load_weights(model, model_config)
+            return model
         return self._load_model_serialized_cpu(vllm_config=vllm_config)
 
     @staticmethod
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index b6fa68ab0925..94f989eef6d9 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -28,7 +28,7 @@
                                   set_forward_context)
 from vllm.logger import init_logger
 from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
-from vllm.model_executor.model_loader import TensorizerLoader, get_model
+from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
 from vllm.multimodal import MULTIMODAL_REGISTRY
 from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
 from vllm.multimodal.utils import group_mm_inputs_by_modality
@@ -1549,7 +1549,18 @@ def load_model(self) -> None:
         logger.info("Starting to load model %s...", self.model_config.model)
         with DeviceMemoryProfiler() as m:  # noqa: SIM117
             time_before_load = time.perf_counter()
-            self.model = get_model(vllm_config=self.vllm_config)
+            model_loader = get_model_loader(self.load_config)
+            if not hasattr(self, "model"):
+                logger.info("Loading model from scratch...")
+                self.model = model_loader.load_model(
+                    vllm_config=self.vllm_config,
+                    model_config=self.model_config)
+            else:
+                logger.info(
+                    "Model was already initialized. Loading weights inplace..."
+                )
+                model_loader.load_weights(self.model,
+                                          model_config=self.model_config)
             if self.lora_config:
                 self.model = self.load_lora_model(self.model,
                                                   self.model_config,
diff --git a/vllm/v1/worker/tpu_model_runner.py b/vllm/v1/worker/tpu_model_runner.py
index c57ac313884d..5de92351e24b 100644
--- a/vllm/v1/worker/tpu_model_runner.py
+++ b/vllm/v1/worker/tpu_model_runner.py
@@ -21,7 +21,7 @@
 from vllm.forward_context import set_forward_context
 from vllm.logger import init_logger
 from vllm.lora.layers import BaseLayerWithLoRA
-from vllm.model_executor.model_loader import get_model
+from vllm.model_executor.model_loader import get_model_loader
 from vllm.multimodal import MULTIMODAL_REGISTRY
 from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargs,
                                     PlaceholderRange)
@@ -171,7 +171,7 @@ def __init__(
         self.encoder_cache_size = encoder_cache_size
 
         # Lazy initialization
-        # self.model: nn.Module  # Set after load_model
+        self.model: nn.Module  # Set after load_model
         self.kv_caches: list[torch.Tensor] = []
         # req_id -> (input_id -> encoder_output)
         self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}
@@ -419,7 +419,6 @@ def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
         return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0
 
     def get_model(self) -> nn.Module:
-        assert self.model is not None
         return self.model
 
     def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
@@ -936,7 +935,18 @@ def load_model(self) -> None:
                 "vllm.model_executor.layers.vocab_parallel_embedding."
                 "get_tensor_model_parallel_rank",
                 return_value=xm_tp_rank):
-            model = get_model(vllm_config=self.vllm_config)
+            # model = get_model(vllm_config=self.vllm_config)
+            model_loader = get_model_loader(self.load_config)
+            if not hasattr(self, "model"):
+                logger.info("Loading model from scratch...")
+                model = model_loader.load_model(vllm_config=self.vllm_config,
+                                                model_config=self.model_config)
+            else:
+                logger.info(
+                    "Model was already initialized. Loading weights inplace..."
+                )
+                model_loader.load_weights(self.model,
+                                          model_config=self.model_config)
         if self.lora_config is not None:
             model = self.load_lora_model(model, self.model_config,
                                          self.scheduler_config,
@@ -947,7 +957,8 @@ def load_model(self) -> None:
         # loading.
         xm.mark_step()
         xm.wait_device_ops()
-        self.model = model
+        if not hasattr(self, "model"):
+            self.model = model
         self.sampler = TPUSampler()
 
     @torch.no_grad()
```
