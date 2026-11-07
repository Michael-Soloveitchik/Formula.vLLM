# PR #40850: [Kernel][Helion] Optimize Helion config parsing latency

- field: gpu_kernels  | area: vllm_kernels
- author: gmagogsfm
- state: MERGED  created: 2026-04-25T01:36:21Z  merged: 2026-05-08T03:27:35Z
- url: https://github.com/vllm-project/vllm/pull/40850
- labels: ready

# Issue

## Problem

Helion kernel config picking was uncached and used free-form string config keys parsed via RegEx on every lookup. For each CUDA graph capture size (per num_token), the code iterated all stored configs and ran expensive regex matching against each string key. This inflated compilation/dispatch latency because the same string parsing repeated for every shape and every call.

## Example

Benchmark of 80,000 pick_config calls with 300 config keys over 8 input shapes on an H100: old regex path cost 1289 us/call vs 1.8 us/call cached in new path (719x faster).

## Use case

Helion-backed kernels during vLLM startup/CUDA graph capture, where many graph sizes each trigger config selection across a large stored config table.

## Resources

CPU-bound config-selection / kernel compile-dispatch time (not GPU compute). Repeated regex parsing dominated per-call latency.

## Type of users

Users running Helion kernels in vLLM, especially with CUDA graphs enabled and many capture sizes.

## Kind of inputs

Many distinct input shapes (num_tokens / batch / hidden combos) matched against a large config set (hundreds of keys), repeated across capture sizes.

## Mechanism with the issue

pick_config matched a free-form string key like "batch_32_hidden_4096" by RegEx parse on each call, with no caching, so ConfigSet lookup re-parsed string keys O(num_configs) per shape per call. The string-keyed JSON format forced parse-on-every-match.

# Solution

## Fix in words

The diff replaces string config keys with a structured, hashable, immutable CaseKey (dict-backed, sorted-JSON str form, with CaseKey.default() for the empty/default case). The on-disk JSON format changes from {platform: {string_key: config}} to {platform: [{"key": {...}, "config": {...}}]}, so ConfigSet.from_dict/to_dict and get_config now operate on CaseKey objects instead of regex-parsed strings. Keys are deserialized once on load into CaseKey dicts, eliminating per-call regex; combined with caching on pick_config, lookups become hash-based dict access. config_picker now returns None instead of the "default" string sentinel.

## Author's solution

Author made two improvements: (1) replaced regex-based config key parsing with structured dict config keys, eliminating per-call regex overhead; (2) deserialized keys once on load and passed parsed dicts to kernel pick_config functions with caching enabled, so config parsing is not repeated across different shapes. Author notes this changes the config serialization format.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/kernels/helion/helpers.py`
- `tests/kernels/helion/test_autotune.py`
- `tests/kernels/helion/test_case_key.py`
- `tests/kernels/helion/test_config_manager.py`
- `tests/kernels/helion/test_register.py`
- `tests/kernels/helion/test_silu_mul_fp8.py`
- `vllm/kernels/helion/__init__.py`
- `vllm/kernels/helion/case_key.py`
- `vllm/kernels/helion/config_manager.py`
- `vllm/kernels/helion/configs/silu_mul_fp8/nvidia_h100.json`
- `vllm/kernels/helion/configs/silu_mul_fp8/nvidia_h200.json`
- `vllm/kernels/helion/ops/silu_mul_fp8.py`
- `vllm/kernels/helion/register.py`

### Diff
```diff
diff --git a/tests/kernels/helion/helpers.py b/tests/kernels/helion/helpers.py
index dbe553be5589..f25c0a274aaa 100644
--- a/tests/kernels/helion/helpers.py
+++ b/tests/kernels/helion/helpers.py
@@ -6,24 +6,26 @@
 from collections.abc import Callable
 from contextlib import contextmanager
 from pathlib import Path
+from typing import Any
 from unittest.mock import patch
 
 import helion
 
+from vllm.kernels.helion.case_key import CaseKey
 from vllm.kernels.helion.config_manager import ConfigManager
 from vllm.kernels.helion.register import register_kernel
 from vllm.kernels.helion.utils import get_canonical_gpu_name
 
 GPU_PLATFORM = get_canonical_gpu_name()
 
-DEFAULT_CONFIGS: dict[str, helion.Config] = {
-    "default": helion.Config(block_sizes=[32]),
+DEFAULT_CONFIGS: dict[CaseKey, helion.Config] = {
+    CaseKey.default(): helion.Config(block_sizes=[32]),
 }
 
 
 @contextmanager
 def dummy_kernel_registry(
-    configs: dict[str, helion.Config] | None = None,
+    configs: dict[CaseKey, helion.Config] | None = None,
 ):
     """Context manager providing a register function with automatic config setup.
 
@@ -34,7 +36,13 @@ def dummy_kernel_registry(
     """
     if configs is None:
         configs = DEFAULT_CONFIGS
-    config_data = {k: v.__dict__["config"] for k, v in configs.items()}
+
+    def _to_config_entries(cfgs: dict) -> list[dict[str, Any]]:
+        pairs: list[dict[str, Any]] = []
+        for k, v in cfgs.items():
+            config_data = v.__dict__["config"]
+            pairs.append({"key": dict(k), "config": config_data})
+        return pairs
 
     with tempfile.TemporaryDirectory() as tmpdir:
         config_dir = Path(tmpdir)
@@ -55,7 +63,7 @@ def decorator(fn: Callable) -> Callable:
                     kernel_dir = config_dir / name
                     kernel_dir.mkdir(parents=True, exist_ok=True)
                     (kernel_dir / f"{GPU_PLATFORM}.json").write_text(
-                        json.dumps(config_data)
+                        json.dumps(_to_config_entries(configs))
                     )
                     return register_kernel(op_name, **kwargs)(fn)
 
diff --git a/tests/kernels/helion/test_autotune.py b/tests/kernels/helion/test_autotune.py
index 87f06c43581e..8b42e145d484 100644
--- a/tests/kernels/helion/test_autotune.py
+++ b/tests/kernels/helion/test_autotune.py
@@ -63,7 +63,7 @@ def test_autotune_disabled_kernel_produces_valid_config(self):
         with dummy_kernel_registry(configs={}) as register:
             wrapper = register(
                 "autotune_test_kernel",
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
                 fake_impl=lambda *a, **kw: None,
                 input_generator=lambda: {
                     "small": (
diff --git a/tests/kernels/helion/test_case_key.py b/tests/kernels/helion/test_case_key.py
new file mode 100644
index 000000000000..335902fd9ef6
--- /dev/null
+++ b/tests/kernels/helion/test_case_key.py
@@ -0,0 +1,68 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+import pytest
+
+from vllm.utils.import_utils import has_helion
+
+if not has_helion():
+    pytest.skip(
+        "Helion is not installed. Install with: pip install vllm[helion]",
+        allow_module_level=True,
+    )
+
+from vllm.kernels.helion.case_key import CaseKey
+
+
+class TestCaseKey:
+    """Test suite for CaseKey class."""
+
+    def test_construction_with_dict(self):
+        key = CaseKey({"intermediate": 2048, "numtokens": 256})
+        assert key["intermediate"] == 2048
+        assert key["numtokens"] == 256
+
+    def test_empty_construction_raises(self):
+        with pytest.raises(TypeError, match="at least one key-value pair"):
+            CaseKey()
+        with pytest.raises(TypeError, match="at least one key-value pair"):
+            CaseKey({})
+
+    def test_default_construction(self):
+        key = CaseKey.default()
+        assert len(key) == 0
+        assert key.is_default()
+
+    def test_non_default_is_not_default(self):
+        key = CaseKey({"intermediate": 2048})
+        assert not key.is_default()
+
+    def test_hashable_and_equality(self):
+        a = CaseKey({"intermediate": 2048, "numtokens": 256})
+        b = CaseKey({"numtokens": 256, "intermediate": 2048})
+        assert a == b
+        assert hash(a) == hash(b)
+        assert a != CaseKey({"intermediate": 4096})
+        assert CaseKey.default() == CaseKey.default()
+
+        configs = {
+            CaseKey.default(): "default_config",
+            a: "a_config",
+        }
+        assert configs[b] == "a_config"
+        assert configs[CaseKey.default()] == "default_config"
+
+    def test_str_is_sorted_json(self):
+        assert str(CaseKey({"z": 1, "a": 2})) == '{"a":2,"z":1}'
+        assert str(CaseKey.default()) == "{}"
+
+    def test_immutable(self):
+        key = CaseKey({"intermediate": 2048})
+        with pytest.raises(TypeError, match="immutable"):
+            key["intermediate"] = 4096
+        with pytest.raises(TypeError, match="immutable"):
+            del key["intermediate"]
+        with pytest.raises(TypeError, match="immutable"):
+            key.update({"numtokens": 256})
+        with pytest.raises(TypeError, match="immutable"):
+            key.clear()
diff --git a/tests/kernels/helion/test_config_manager.py b/tests/kernels/helion/test_config_manager.py
index 337696ee066b..f8e5eae6f106 100644
--- a/tests/kernels/helion/test_config_manager.py
+++ b/tests/kernels/helion/test_config_manager.py
@@ -23,6 +23,7 @@
 
 import helion
 
+from vllm.kernels.helion.case_key import CaseKey
 from vllm.kernels.helion.config_manager import (
     ConfigManager,
     ConfigSet,
@@ -49,22 +50,25 @@ def test_config_set_creation(self):
 
     def test_config_set_from_dict(self):
         """Test creating ConfigSet from dictionary data."""
-        # Use realistic config data that helion.Config can handle
         config_data = {
             "block_sizes": [32, 16],
             "num_warps": 4,
             "num_stages": 3,
             "pid_type": "persistent_interleaved",
         }
-        data = {"h100": {"batch_32_hidden_4096": config_data}}
+        data = {
+            "h100": [
+                {"key": {"batch": 32, "hidden": 4096}, "config": config_data},
+            ]
+        }
 
         config_set = ConfigSet.from_dict("test_kernel", data)
 
         assert config_set.kernel_name == "test_kernel"
         assert config_set.get_platforms() == ["h100"]
 
-        # Verify the config was created correctly
-        config = config_set.get_config("h100", "batch_32_hidden_4096")
+        internal_key = CaseKey({"batch": 32, "hidden": 4096})
+        config = config_set.get_config("h100", internal_key)
         assert isinstance(config, helion.Config)
         assert config.block_sizes == [32, 16]
         assert config.num_warps == 4
@@ -76,17 +80,19 @@ def test_config_set_get_config_keyerror(self):
         config_set = ConfigSet("test_kernel")
 
         with pytest.raises(KeyError, match="platform 'h100' not found"):
-            config_set.get_config("h100", "batch_32_hidden_4096")
+            config_set.get_config("h100", "nonexistent")
 
-        # Use realistic config data
         config_data = {"num_warps": 8, "num_stages": 4}
-        data = {"h100": {"batch_64_hidden_2048": config_data}}
+        data = {
+            "h100": [
+                {"key": {"batch": 64, "hidden": 2048}, "config": config_data},
+            ]
+        }
         config_set = ConfigSet.from_dict("test_kernel", data)
 
-        with pytest.raises(
-            KeyError, match="config_key 'batch_32_hidden_4096' not found"
-        ):
-            config_set.get_config("h100", "batch_32_hidden_4096")
+        nonexistent_key = CaseKey({"batch": 32, "hidden": 4096})
+        with pytest.raises(KeyError, match="config_key .* not found"):
+            config_set.get_config("h100", nonexistent_key)
 
     def test_config_set_get_platforms(self):
         """Test get_platforms method."""
@@ -95,8 +101,12 @@ def test_config_set_get_platforms(self):
         config2 = {"num_warps": 8, "num_stages": 5}
 
         data = {
-            "h100": {"batch_32_hidden_4096": config1},
-            "a100": {"batch_16_hidden_2048": config2},
+            "h100": [
+                {"key": {"batch": 32, "hidden": 4096}, "config": config1},
+            ],
+            "a100": [
+                {"key": {"batch": 16, "hidden": 2048}, "config": config2},
+            ],
         }
         config_set = ConfigSet.from_dict("test_kernel", data)
 
@@ -105,39 +115,49 @@ def test_config_set_get_platforms(self):
 
     def test_config_set_get_config_keys(self):
         """Test get_config_keys method."""
-        # Use realistic config data
         config1 = {"num_warps": 4, "num_stages": 3}
         config2 = {"num_warps": 8, "num_stages": 5}
 
         data = {
-            "h100": {
-                "batch_32_hidden_4096": config1,
-                "batch_64_hidden_2048": config2,
-            }
+            "h100": [
+                {"key": {"batch": 32, "hidden": 4096}, "config": config1},
+                {"key": {"batch": 64, "hidden": 2048}, "config": config2},
+            ]
         }
         config_set = ConfigSet.from_dict("test_kernel", data)
 
         config_keys = config_set.get_config_keys("h100")
-        assert config_keys == ["batch_32_hidden_4096", "batch_64_hidden_2048"]
+        expected_keys = sorted(
+            [
+                CaseKey({"batch": 32, "hidden": 4096}),
+                CaseKey({"batch": 64, "hidden": 2048}),
+            ],
+            key=lambda k: str(k) if k is not None else "",
+        )
+        assert config_keys == expected_keys
 
         assert config_set.get_config_keys("v100") == []
 
     def test_config_set_to_dict(self):
         """Test converting ConfigSet to dictionary."""
-        # Use realistic config data
         original_config = {
             "block_sizes": [64, 32],
             "num_warps": 16,
             "num_stages": 4,
             "pid_type": "persistent_blocked",
         }
-        original_data = {"h100": {"batch_32_hidden_4096": original_config}}
+        original_data = {
+            "h100": [
+                {"key": {"batch": 32, "hidden": 4096}, "config": original_config},
+            ]
+        }
 
         config_set = ConfigSet.from_dict("test_kernel", original_data)
         result_data = config_set.to_dict()
 
-        # The result should match the original (Config roundtrip should work)
-        assert result_data == original_data
+        internal_key = CaseKey({"batch": 32, "hidden": 4096})
+        assert internal_key in result_data["h100"]
+        assert result_data["h100"][internal_key] == original_config
 
 
 class TestConfigManager:
@@ -202,7 +222,10 @@ def test_load_config_set_valid_file(self):
             kernel_dir.mkdir()
             platform_file = kernel_dir / "h100.json"
             with open(platform_file, "w") as f:
-                json.dump({"batch_32_hidden_4096": kernel_config}, f)
+                json.dump(
+                    [{"key": {"batch": 32, "hidden": 4096}, "config": kernel_config}],
+                    f,
+                )
 
             manager = ConfigManager(base_dir=temp_dir)
             config_set = manager.load_config_set("test_kernel")
@@ -211,7 +234,8 @@ def test_load_config_set_valid_file(self):
             assert config_set.kernel_name == "test_kernel"
             assert config_set.get_platforms() == ["h100"]
 
-            config = config_set.get_config("h100", "batch_32_hidden_4096")
+            internal_key = CaseKey({"batch": 32, "hidden": 4096})
+            config = config_set.get_config("h100", internal_key)
             assert isinstance(config, helion.Config)
             assert config.block_sizes == [128, 64]
             assert config.num_warps == 8
@@ -241,7 +265,11 @@ def test_save_config_set(self):
                 "num_stages": 8,
                 "pid_type": "persistent_blocked",
             }
-            data = {"h100": {"batch_32_hidden_4096": kernel_config}}
+            data = {
+                "h100": [
+                    {"key": {"batch": 32, "hidden": 4096}, "config": kernel_config},
+                ]
+            }
             config_set = ConfigSet.from_dict("test_kernel", data)
 
             manager = ConfigManager(base_dir=temp_dir)
@@ -255,13 +283,21 @@ def test_save_config_set(self):
             assert platform_file.exists()
             with open(platform_file) as f:
                 loaded_data = json.load(f)
-            assert loaded_data == data["h100"]
+            assert isinstance(loaded_data, list)
+            assert len(loaded_data) == 1
+            entry = loaded_data[0]
+            assert entry["key"] == {"batch": 32, "hidden": 4096}
+            assert entry["config"] == kernel_config
 
     def test_save_config_set_creates_directory(self):
         """Test that save_config_set creates parent directories if needed."""
         with tempfile.TemporaryDirectory() as temp_dir:
             nested_dir = Path(temp_dir) / "nested" / "configs"
-            data = {"h100": {"default": {"num_warps": 4}}}
+            data = {
+                "h100": [
+                    {"key": {}, "config": {"num_warps": 4}},
+                ]
+            }
             config_set = ConfigSet.from_dict("test_kernel", data)
 
             manager = ConfigManager(base_dir=nested_dir)
@@ -288,34 +324,41 @@ def test_get_platform_configs(self):
             kernel_dir.mkdir()
             with open(kernel_dir / "h100.json", "w") as f:
                 json.dump(
-                    {
-                        "batch_32_hidden_4096": config_1,
-                        "batch_64_hidden_2048": config_2,
-                        "default": default_config,
-                    },
+                    [
+                        {"key": {"batch": 32, "hidden": 4096}, "config": config_1},
+                        {"key": {"batch": 64, "hidden": 2048}, "config": config_2},
+                        {"key": {}, "config": default_config},
+                    ],
                     f,
                 )
             with open(kernel_dir / "a100.json", "w") as f:
-                json.dump({"batch_16_hidden_1024": config_3}, f)
+                json.dump(
+                    [{"key": {"batch": 16, "hidden": 1024}, "config": config_3}],
+                    f,
+                )
 
             manager = ConfigManager(base_dir=temp_dir)
 
+            key_b32_h4096 = CaseKey({"batch": 32, "hidden": 4096})
+            key_b64_h2048 = CaseKey({"batch": 64, "hidden": 2048})
+            key_b16_h1024 = CaseKey({"batch": 16, "hidden": 1024})
+
             h100_configs = manager.get_platform_configs("test_kernel", "h100")
             assert len(h100_configs) == 3
-            assert "batch_32_hidden_4096" in h100_configs
-            assert "batch_64_hidden_2048" in h100_configs
-            assert "default" in h100_configs
+            assert key_b32_h4096 in h100_configs
+            assert key_b64_h2048 in h100_configs
+            assert CaseKey.default() in h100_configs
             for config in h100_configs.values():
                 assert isinstance(config, helion.Config)
 
-            assert h100_configs["batch_32_hidden_4096"].num_warps == 4
-            assert h100_configs["default"].num_stages == 7
+            assert h100_configs[key_b32_h4096].num_warps == 4
+            assert h100_configs[CaseKey.default()].num_stages == 7
 
             a100_configs = manager.get_platform_configs("test_kernel", "a100")
             assert len(a100_configs) == 1
-            assert "batch_16_hidden_1024" in a100_configs
-            assert isinstance(a100_configs["batch_16_hidden_1024"], helion.Config)
-            assert a100_configs["batch_16_hidden_1024"].num_warps == 2
+            assert key_b16_h1024 in a100_configs
+            assert isinstance(a100_configs[key_b16_h1024], helion.Config)
+            assert a100_configs[key_b16_h1024].num_warps == 2
 
             nonexistent_configs = manager.get_platform_configs("test_kernel", "v100")
             assert len(nonexistent_configs) == 0
diff --git a/tests/kernels/helion/test_register.py b/tests/kernels/helion/test_register.py
index bad3017c5c96..c82c3c8358ed 100644
--- a/tests/kernels/helion/test_register.py
+++ b/tests/kernels/helion/test_register.py
@@ -24,6 +24,7 @@
 import helion.language as hl
 
 from tests.kernels.helion.helpers import dummy_kernel_registry
+from vllm.kernels.helion.case_key import CaseKey
 from vllm.kernels.helion.config_manager import ConfigManager
 from vllm.kernels.helion.register import (
     _HOP_AVAILABLE,
@@ -54,22 +55,22 @@ def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
 def sample_configs():
     """Create real Helion config objects for testing."""
     return {
-        "hiddensize_4096_batchsize_32": helion.Config(
+        CaseKey({"batchsize": 32, "hiddensize": 4096}): helion.Config(
             block_sizes=[128],
             num_warps=4,
             num_stages=3,
         ),
-        "hiddensize_4096_batchsize_64": helion.Config(
+        CaseKey({"batchsize": 64, "hiddensize": 4096}): helion.Config(
             block_sizes=[256],
             num_warps=8,
             num_stages=4,
         ),
-        "hiddensize_4096_batchsize_128": helion.Config(
+        CaseKey({"batchsize": 128, "hiddensize": 4096}): helion.Config(
             block_sizes=[512],
             num_warps=16,
             num_stages=2,
         ),
-        "default": helion.Config(
+        CaseKey.default(): helion.Config(
             block_sizes=[64],
             num_warps=2,
             num_stages=2,
@@ -101,8 +102,7 @@ def configured_kernel(sample_kernel, sample_configs, config_manager_with_test_co
     """Create a ConfiguredHelionKernel for testing."""
 
     def test_config_picker(args, config_keys):
-        """Simple config picker that returns default."""
-        return "default"
+        return None
 
     with (
         patch(
@@ -115,7 +115,6 @@ def test_config_picker(args, config_keys):
         ),
         patch("vllm.kernels.helion.register.helion.kernel") as mock_kernel,
     ):
-        # Mock just the helion.kernel decorator to avoid actual kernel compilation
         mock_decorated = Mock()
         mock_kernel.return_value = Mock(return_value=mock_decorated)
 
@@ -199,7 +198,9 @@ class TestConfiguredHelionKernel:
 
     def test_init_raises_without_picker(self, sample_kernel, sample_configs):
         """Test that __init__ raises when no picker registered."""
-        configs = {"default": sample_configs["default"]}
+        configs: dict[CaseKey, helion.Config] = {
+            CaseKey.default(): sample_configs[CaseKey.default()]
+        }
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(return_value=configs)
 
@@ -227,7 +228,7 @@ def test_config_selector_validates_picker_result(
         """Test that config selector validates picker returns valid key."""
 
         def invalid_picker(args, config_keys):
-            return "invalid_key"
+            return {"invalid": 999}
 
         kernel = create_configured_kernel_with_configs(
             op_name="test_kernel",
@@ -263,7 +264,7 @@ def none_picker(args, config_keys):
         selector = kernel._create_config_selector(key_computer)
 
         result = selector((torch.randn(32, 4096),))
-        assert result is kernel.configs["default"]
+        assert result is kernel.configs[CaseKey.default()]
 
     def test_create_decorated_kernel_passes_helion_settings(
         self, sample_kernel, sample_configs
@@ -271,7 +272,7 @@ def test_create_decorated_kernel_passes_helion_settings(
         """Test that _create_decorated_kernel passes helion_settings."""
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         settings = helion.Settings()
         settings.print_output_code = True
@@ -315,10 +316,10 @@ def tracking_picker(args, config_keys):
             x = args[0]
             batch_size = x.shape[0]
             if batch_size <= 32:
-                return "hiddensize_4096_batchsize_32"
+                return CaseKey({"batchsize": 32, "hiddensize": 4096})
             elif batch_size <= 64:
-                return "hiddensize_4096_batchsize_64"
-            return "hiddensize_4096_batchsize_128"
+                return CaseKey({"batchsize": 64, "hiddensize": 4096})
+            return CaseKey({"batchsize": 128, "hiddensize": 4096})
 
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)
@@ -350,13 +351,13 @@ def tracking_picker(args, config_keys):
 
             tensor = torch.randn(50, 4096)  # batch=50, should select batchsize_64
 
-            # key receives unpacked args, autotuner receives args as tuple
             key_result = key_fn(tensor)
             autotuner = autotuner_fn(None, (tensor,))
             config = autotuner.autotune()
 
-            assert key_result == "hiddensize_4096_batchsize_64"
-            assert config is kernel.configs["hiddensize_4096_batchsize_64"]
+            expected_key = CaseKey({"batchsize": 64, "hiddensize": 4096})
+            assert key_result == str(expected_key)
+            assert config is kernel.configs[expected_key]
 
 
 class TestHelionKernelWrapper:
@@ -369,7 +370,7 @@ def fake_impl(*args, **kwargs):
             return torch.zeros_like(args[0])
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(
@@ -406,7 +407,7 @@ def fake_impl(*args, **kwargs):
             return torch.zeros_like(args[0])
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(return_value={})
@@ -441,7 +442,7 @@ def fake_impl(*args, **kwargs):
             return torch.zeros_like(args[0])
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(return_value={})
@@ -476,7 +477,7 @@ def fake_impl(*args, **kwargs):
             return torch.zeros_like(args[0])
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         expected_inputs = {"key1": (torch.randn(4),)}
         input_gen = Mock(return_value=expected_inputs)
@@ -516,7 +517,7 @@ def fake_impl(*args, **kwargs):
             return torch.zeros_like(args[0])
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(return_value={})
@@ -563,7 +564,7 @@ def fake_impl(*args, **kwargs):
             return torch.zeros_like(args[0])
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)
@@ -601,7 +602,9 @@ def test_init_eagerly_initializes_hop_path(self):
         on the HOP path (no custom op registration needed)."""
         from vllm.kernels.helion.utils import get_canonical_gpu_name
 
-        configs = {"default": helion.Config(block_sizes=[4, 4])}
+        configs: dict[CaseKey, helion.Config] = {
+            CaseKey.default(): helion.Config(block_sizes=[4, 4])
+        }
         with (
             dummy_kernel_registry(configs=configs) as register,
             patch(
@@ -610,7 +613,7 @@ def test_init_eagerly_initializes_hop_path(self):
             ) as mock_gpu,
         ):
             wrapper = register(
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
             )(_add_kernel)
 
             mock_gpu.assert_called_once()
@@ -642,7 +645,7 @@ def test_init_eagerly_initializes(self):
             ) as mock_gpu,
         ):
             wrapper = register(
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
             )(_add_kernel)
 
             # Init must have detected GPU and built the kernel
@@ -660,7 +663,7 @@ def fake_impl(*args, **kwargs):
             return torch.zeros_like(args[0])
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)
@@ -703,7 +706,7 @@ def fake_impl(*args, **kwargs):
             return torch.zeros_like(args[0])
 
         def default_picker(args, config_keys):
-            return "default"
+            return None
 
         mock_config_manager = Mock(spec=ConfigManager)
         mock_config_manager.get_platform_configs = Mock(return_value=sample_configs)
@@ -784,9 +787,9 @@ def test_get_registered_kernels_returns_copy(self):
     def test_get_kernel_by_name_returns_kernel(self):
         """Test get_kernel_by_name returns registered kernel."""
         with dummy_kernel_registry() as register:
-            wrapper = register(
-                "test_kernel", config_picker=lambda args, keys: "default"
-            )(_add_kernel)
+            wrapper = register("test_kernel", config_picker=lambda args, keys: None)(
+                _add_kernel
+            )
 
         from vllm.kernels.helion.register import _REGISTERED_KERNELS
 
@@ -809,7 +812,7 @@ def test_register_kernel_auto_generates_fake_impl(self):
             mock_fake = Mock()
             mock_infer.return_value = mock_fake
             wrapper = register(
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
             )(_add_kernel)
 
         mock_infer.assert_called_once_with(_add_kernel, None)
@@ -818,7 +821,7 @@ def test_register_kernel_auto_generates_fake_impl(self):
     def test_register_kernel_creates_wrapper(self):
         """Test register_kernel creates HelionKernelWrapper."""
         with dummy_kernel_registry() as register:
-            result = register("test_name", config_picker=lambda args, keys: "default")(
+            result = register("test_name", config_picker=lambda args, keys: None)(
                 _add_kernel
             )
 
@@ -829,16 +832,16 @@ def test_register_kernel_creates_wrapper(self):
     def test_register_kernel_auto_detects_name(self):
         """Test register_kernel uses function name when no name provided."""
         with dummy_kernel_registry() as register:
-            wrapper = register(config_picker=lambda args, keys: "default")(_add_kernel)
+            wrapper = register(config_picker=lambda args, keys: None)(_add_kernel)
 
         assert wrapper.op_name == "_add_kernel"
 
     def test_register_kernel_registers_in_global_registry(self):
         """Test register_kernel adds wrapper to global registry."""
         with dummy_kernel_registry() as register:
-            wrapper = register(
-                "test_kernel", config_picker=lambda args, keys: "default"
-            )(_add_kernel)
+            wrapper = register("test_kernel", config_picker=lambda args, keys: None)(
+                _add_kernel
+            )
 
         registered_kernels = get_registered_kernels()
         assert "test_kernel" in registered_kernels
@@ -852,7 +855,7 @@ def test_register_kernel_passes_helion_settings(self):
         with dummy_kernel_registry() as register:
             result = register(
                 "test_name",
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
                 helion_settings=settings,
             )(_add_kernel)
 
@@ -865,7 +868,7 @@ def test_register_kernel_supports_decorator_syntax(self):
         with dummy_kernel_registry() as register:
             result = register(
                 "custom_name",
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
                 fake_impl=mock_fake,
             )(_add_kernel)
 
@@ -875,12 +878,12 @@ def test_register_kernel_supports_decorator_syntax(self):
     def test_register_kernel_raises_on_duplicate_registration(self):
         """Test register_kernel raises error on duplicate names."""
         with dummy_kernel_registry() as register:
-            register("duplicate_name", config_picker=lambda args, keys: "default")(
+            register("duplicate_name", config_picker=lambda args, keys: None)(
                 _add_kernel
             )
 
             with pytest.raises(ValueError, match="already registered"):
-                register("duplicate_name", config_picker=lambda args, keys: "default")(
+                register("duplicate_name", config_picker=lambda args, keys: None)(
                     _add_kernel
                 )
 
@@ -893,7 +896,7 @@ def test_register_kernel_rejects_autotuner_fn_in_settings(self):
 
             @register_kernel(
                 "test",
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
                 helion_settings=mock_settings,
             )
             def test_kernel(x):
@@ -910,7 +913,7 @@ def test_register_kernel_no_warning_with_static_shapes_false(self):
         ):
             register(
                 "test",
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
                 helion_settings=mock_settings,
             )(_add_kernel)
 
@@ -940,7 +943,7 @@ def fake_impl(*args, **kwargs):
 
             wrapper = register_kernel(
                 "disabled_kernel",
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
                 fake_impl=fake_impl,
             )(_add_kernel)
 
@@ -957,12 +960,14 @@ class TestTorchCompileHOP:
     def test_compiled_graph_contains_helion_hop(self):
         """Verify torch.compile on a HelionKernelWrapper emits a
         helion_kernel_wrapper_mutation HOP node in the FX graph."""
-        configs = {"default": helion.Config(block_sizes=[4, 4])}
+        configs: dict[CaseKey, helion.Config] = {
+            CaseKey.default(): helion.Config(block_sizes=[4, 4])
+        }
 
         with dummy_kernel_registry(configs=configs) as register:
             add_helion_kernel = register(
                 op_name="test_torch_compile_add_kernel",
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
             )(_add_kernel)
 
         captured_graph: torch.fx.GraphModule | None = None
@@ -1013,12 +1018,14 @@ def f(x, y):
     def test_inductor_backend_compiles_helion_hop(self):
         """Test torch.compile with inductor backend and Helion fusion enabled."""
 
-        configs = {"default": helion.Config(block_sizes=[4, 4])}
+        configs: dict[CaseKey, helion.Config] = {
+            CaseKey.default(): helion.Config(block_sizes=[4, 4])
+        }
 
         with dummy_kernel_registry(configs=configs) as register:
             add_helion_kernel = register(
                 op_name="test_inductor_add_kernel",
-                config_picker=lambda args, keys: "default",
+                config_picker=lambda args, keys: None,
                 helion_settings=helion.Settings(
                     torch_compile_fusion=True, static_shapes=False
                 ),
diff --git a/tests/kernels/helion/test_silu_mul_fp8.py b/tests/kernels/helion/test_silu_mul_fp8.py
index 887f20b9f563..bd3131e08da2 100644
--- a/tests/kernels/helion/test_silu_mul_fp8.py
+++ b/tests/kernels/helion/test_silu_mul_fp8.py
@@ -13,8 +13,10 @@
         allow_module_level=True,
     )
 
+from vllm.kernels.helion.case_key import CaseKey
 from vllm.kernels.helion.config_manager import ConfigManager
 from vllm.kernels.helion.ops.silu_mul_fp8 import (
+    _pick_cache,
     pick_silu_mul_fp8_config,
     silu_mul_fp8,
     silu_mul_fp8_baseline,
@@ -52,10 +54,13 @@ def reset_config_manager_singleton():
 
 
 class TestSiluMulFp8ConfigPicker:
+    def setup_method(self):
+        _pick_cache.clear()
+
     def test_config_picker_exact_match(self):
         config_keys = [
-            "intermediate_2048_numtokens_256",
-            "intermediate_4096_numtokens_256",
+            CaseKey({"intermediate": 2048, "numtokens": 256}),
+            CaseKey({"intermediate": 4096, "numtokens": 256}),
         ]
 
         input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
@@ -63,33 +68,22 @@ def test_config_picker_exact_match(self):
         args = (input_tensor, scale)
 
         selected_key = pick_silu_mul_fp8_config(args, config_keys)
-        assert selected_key == "intermediate_2048_numtokens_256"
+        assert selected_key == CaseKey({"intermediate": 2048, "numtokens": 256})
 
     def test_config_picker_closest_match(self):
         config_keys = [
-            "intermediate_2048_numtokens_256",
-            "intermediate_4096_numtokens_256",
+            CaseKey({"intermediate": 2048, "numtokens": 256}),
+            CaseKey({"intermediate": 4096, "numtokens": 256}),
         ]
-        # Use 7000 (intermediate_size=3500) which is closer to 4096 than 2048
         input_tensor = torch.randn(32, 7000, dtype=torch.bfloat16, device="cuda")
         scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
         args = (input_tensor, scale)
 
         selected_key = pick_silu_mul_fp8_config(args, config_keys)
-        assert selected_key == "intermediate_4096_numtokens_256"
-
-    def test_config_picker_fallback_to_default(self):
-        config_keys = ["default"]
-
-        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
-        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
-        args = (input_tensor, scale)
-
-        selected_key = pick_silu_mul_fp8_config(args, config_keys)
-        assert selected_key == "default"
+        assert selected_key == CaseKey({"intermediate": 4096, "numtokens": 256})
 
     def test_config_picker_no_configs(self):
-        config_keys: list[str] = []
+        config_keys: list[dict] = []
 
         input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
         scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
@@ -101,9 +95,9 @@ def test_config_picker_no_configs(self):
     @pytest.mark.parametrize("intermediate_size", [2048, 4096, 5120])
     def test_config_picker_different_sizes(self, intermediate_size):
         config_keys = [
-            "intermediate_2048_numtokens_256",
-            "intermediate_4096_numtokens_256",
-            "intermediate_5120_numtokens_256",
+            CaseKey({"intermediate": 2048, "numtokens": 256}),
+            CaseKey({"intermediate": 4096, "numtokens": 256}),
+            CaseKey({"intermediate": 5120, "numtokens": 256}),
         ]
 
         input_tensor = torch.randn(
@@ -113,72 +107,47 @@ def test_config_picker_different_sizes(self, intermediate_size):
         args = (input_tensor, scale)
 
         selected_key = pick_silu_mul_fp8_config(args, config_keys)
-        expected_key = f"intermediate_{intermediate_size}_numtokens_256"
-        assert selected_key == expected_key
+        assert selected_key == {
+            "intermediate": intermediate_size,
+            "numtokens": 256,
+        }
 
     def test_config_picker_numtokens_ceiling(self):
-        """Pick the smallest numtokens >= input num_tokens."""
         config_keys = [
-            "intermediate_4096_numtokens_8",
-            "intermediate_4096_numtokens_32",
-            "intermediate_4096_numtokens_128",
-            "intermediate_4096_numtokens_256",
+            CaseKey({"intermediate": 4096, "numtokens": 8}),
+            CaseKey({"intermediate": 4096, "numtokens": 32}),
+            CaseKey({"intermediate": 4096, "numtokens": 128}),
+            CaseKey({"intermediate": 4096, "numtokens": 256}),
         ]
-        # 20 tokens -> should pick numtokens_32 (smallest >= 20)
         input_tensor = torch.randn(20, 8192, dtype=torch.bfloat16, device="cuda")
         scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
 
         selected_key = pick_silu_mul_fp8_config((input_tensor, scale), config_keys)
-        assert selected_key == "intermediate_4096_numtokens_32"
+        assert selected_key == CaseKey({"intermediate": 4096, "numtokens": 32})
 
     def test_config_picker_numtokens_exact(self):
-        """Exact num_tokens match is preferred over ceiling."""
         config_keys = [
-            "intermediate_4096_numtokens_8",
-            "intermediate_4096_numtokens_32",
-            "intermediate_4096_numtokens_128",
+            CaseKey({"intermediate": 4096, "numtokens": 8}),
+            CaseKey({"intermediate": 4096, "numtokens": 32}),
+            CaseKey({"intermediate": 4096, "numtokens": 128}),
         ]
         input_tensor = torch.randn(32, 8192, dtype=torch.bfloat16, device="cuda")
         scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
 
         selected_key = pick_silu_mul_fp8_config((input_tensor, scale), config_keys)
-        assert selected_key == "intermediate_4096_numtokens_32"
+        assert selected_key == CaseKey({"intermediate": 4096, "numtokens": 32})
 
     def test_config_picker_numtokens_fallback_to_largest(self):
-        """Fall back to the largest numtokens when input exceeds all."""
         config_keys = [
-            "intermediate_4096_numtokens_8",
-            "intermediate_4096_numtokens_32",
-            "intermediate_4096_numtokens_128",
+            CaseKey({"intermediate": 4096, "numtokens": 8}),
+            CaseKey({"intermediate": 4096, "numtokens": 32}),
+            CaseKey({"intermediate": 4096, "numtokens": 128}),
         ]
-        # 512 tokens -> exceeds all available, should pick largest (128)
         input_tensor = torch.randn(512, 8192, dtype=torch.bfloat16, device="cuda")
         scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
 
         selected_key = pick_silu_mul_fp8_config((input_tensor, scale), config_keys)
-        assert selected_key == "intermediate_4096_numtokens_128"
-
-    def test_config_picker_malformed_key_raises(self):
-        """Malformed config keys should raise ValueError."""
-        config_keys = ["intermediate_4096_badformat_256"]
-        input_tensor = torch.randn(32, 8192, dtype=torch.bfloat16, device="cuda")
-        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
-
-        with pytest.raises(ValueError, match="Malformed config key"):
-            pick_silu_mul_fp8_config((input_tensor, scale), config_keys)
-
-    def test_config_picker_default_ignored_when_valid_keys_exist(self):
-        """'default' is skipped in favor of a real match."""
-        config_keys = [
-            "default",
-            "intermediate_4096_numtokens_32",
-            "intermediate_4096_numtokens_128",
-        ]
-        input_tensor = torch.randn(64, 8192, dtype=torch.bfloat16, device="cuda")
-        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
-
-        selected_key = pick_silu_mul_fp8_config((input_tensor, scale), config_keys)
-        assert selected_key == "intermediate_4096_numtokens_128"
+        assert selected_key == CaseKey({"intermediate": 4096, "numtokens": 128})
 
 
 class TestSiluMulFp8Correctness:
diff --git a/vllm/kernels/helion/__init__.py b/vllm/kernels/helion/__init__.py
index 2568baa20dae..8c05c428bb07 100644
--- a/vllm/kernels/helion/__init__.py
+++ b/vllm/kernels/helion/__init__.py
@@ -3,11 +3,13 @@
 """Helion integration for vLLM."""
 
 import vllm.kernels.helion.ops  # noqa: F401  Auto-register all Helion ops
+from vllm.kernels.helion.case_key import CaseKey
 from vllm.kernels.helion.config_manager import (
     ConfigManager,
     ConfigSet,
 )
 from vllm.kernels.helion.register import (
+    ConfigPicker,
     ConfiguredHelionKernel,
     HelionKernelWrapper,
     get_kernel_by_name,
@@ -19,9 +21,11 @@
 
 __all__ = [
     # Config management
+    "CaseKey",
     "ConfigManager",
     "ConfigSet",
     # Kernel registration
+    "ConfigPicker",
     "ConfiguredHelionKernel",
     "HelionKernelWrapper",
     "get_kernel_by_name",
diff --git a/vllm/kernels/helion/case_key.py b/vllm/kernels/helion/case_key.py
new file mode 100644
index 000000000000..32b544de39cc
--- /dev/null
+++ b/vllm/kernels/helion/case_key.py
@@ -0,0 +1,72 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Structured key for identifying kernel config/autotune/benchmark cases.
+"""
+
+from __future__ import annotations
+
+import json
+from typing import Any
+
+
+class CaseKey(dict[str, Any]):
+    """Immutable, hashable dict for identifying kernel cases.
+
+    Used as the key for config lookup, autotuning, benchmarking, and
+    input generation.  Behaves like a read-only dict and can be used
+    as a dict key or in sets.
+
+    The canonical string form (``__str__``) is stable JSON with sorted
+    keys.  Use ``CaseKey.default()`` for the default/fallback key.
+    The regular constructor requires at least one key-value pair::
+
+        CaseKey({"intermediate": 2048, "numtokens": 256})
+        CaseKey.default()  # default/fallback
+    """
+
+    def __init__(self, *args: Any, _allow_empty: bool = False, **kwargs: Any):
+        super().__init__(*args, **kwargs)
+        if not self and not _allow_empty:
+            raise TypeError(
+                "CaseKey requires at least one key-value pair. "
+                "Use CaseKey.default() for the default config key."
+            )
+        self._str: str | None = None
+        self._hash: int | None = None
+
+    @classmethod
+    def default(cls) -> CaseKey:
+        """Create a default case key (empty)."""
+        return cls(_allow_empty=True)
+
+    def __hash__(self) -> int:  # type: ignore[override]
+        if self._hash is None:
+            self._hash = hash(str(self))
+        return self._hash
+
+    def __str__(self) -> str:
+        if self._str is None:
+            self._str = json.dumps(dict(self), sort_keys=True, separators=(",", ":"))
+        return self._str
+
+    def __repr__(self) -> str:
+        if not self:
+            return "CaseKey.default()"
+        return f"CaseKey({dict(self)})"
+
+    def is_default(self) -> bool:
+        """Return True if this is the default case key (empty)."""
+        return not self
+
+    def _readonly(self, *args: Any, **kwargs: Any) -> Any:
+        raise TypeError("CaseKey is immutable")
+
+    __setitem__ = _readonly  # type: ignore[assignment]
+    __delitem__ = _readonly  # type: ignore[assignment]
+    __ior__ = _readonly  # type: ignore[assignment]
+    update = _readonly  # type: ignore[assignment]
+    pop = _readonly  # type: ignore[assignment]
+    popitem = _readonly  # type: ignore[assignment]
+    setdefault = _readonly  # type: ignore[assignment]
+    clear = _readonly  # type: ignore[assignment]
diff --git a/vllm/kernels/helion/config_manager.py b/vllm/kernels/helion/config_manager.py
index f34d936041f4..ca37a68e8101 100644
--- a/vllm/kernels/helion/config_manager.py
+++ b/vllm/kernels/helion/config_manager.py
@@ -11,25 +11,25 @@
 Each kernel has a directory: {kernel_name}/
 Inside, each GPU platform has its own JSON file: {kernel_name}/{platform}.json
 
-For example:
-    silu_mul_fp8/
-        nvidia_h100.json    # { "default": {...}, "batch_32_hidden_4096": {...} }
-        nvidia_h200.json    # { "batch_16_hidden_2048": {...} }
-
-Each platform file maps config keys to Helion config objects.
-Config keys should be structured strings that encode the relevant
-parameters (e.g., "batch_32_hidden_4096", "seq_512_heads_16", "fp8_batch_64", etc.).
-
-Classes
--------
-- ConfigSet: In-memory collection of configs for a kernel with lookup/query APIs.
-- ConfigManager: File-level operations for config persistence.
+Platform files store config entries as a JSON array::
+
+    [
+        {"key": {}, "config": {...}},
+        {"key": {"intermediate": 2048, "numtokens": 256}, "config": {...}},
+        ...,
+    ]
+
+Config keys are ``CaseKey`` instances mapping parameter names to
+values.  The default config uses ``CaseKey.default()``.
 """
 
+from __future__ import annotations
+
 import json
 from pathlib import Path
 from typing import Any
 
+from vllm.kernels.helion.case_key import CaseKey
 from vllm.logger import init_logger
 from vllm.utils.import_utils import has_helion
 
@@ -45,11 +45,13 @@
 
 
 class ConfigSet:
-    """In-memory collection of Helion configs with lookup/query capabilities."""
+    """In-memory collection of Helion configs with lookup/query capabilities.
 
-    # Type alias for nested config structure:
-    # platform -> config_key -> helion.Config
-    _ConfigDict = dict[str, dict[str, "helion.Config"]]
+    Configs are stored keyed by ``CaseKey``.  The default config
+    uses ``CaseKey.default()`` as its key.
+    """
+
+    _ConfigDict = dict[str, dict[CaseKey, "helion.Config"]]
 
     def __init__(self, kernel_name: str):
         self._kernel_name = kernel_name
@@ -59,7 +61,7 @@ def __init__(self, kernel_name: str):
     def kernel_name(self) -> str:
         return self._kernel_name
 
-    def get_config(self, platform: str, config_key: str) -> helion.Config:
+    def get_config(self, platform: str, config_key: CaseKey) -> helion.Config:
         platform_dict = self._configs.get(platform)
         if platform_dict is None:
             avail_platforms = self.get_platforms()
@@ -82,7 +84,8 @@ def get_config(self, platform: str, config_key: str) -> helion.Config:
             avail_keys = self.get_config_keys(platform)
             raise KeyError(
                 f"Config not found for kernel '{self._kernel_name}': "
-                f"config_key '{config_key}' not found for platform '{platform}'. "
+                f"config_key '{config_key}' not found for "
+                f"platform '{platform}'. "
                 f"Available config_keys: {avail_keys or '(none)'}"
             )
 
@@ -91,25 +94,34 @@ def get_config(self, platform: str, config_key: str) -> helion.Config:
     def get_platforms(self) -> list[str]:
         return sorted(self._configs.keys())
 
-    def get_config_keys(self, platform: str) -> list[str]:
+    def get_config_keys(self, platform: str) -> list[CaseKey]:
         platform_dict = self._configs.get(platform.lower())
         if platform_dict is None:
             return []
-        return sorted(platform_dict.keys())
-
-    def to_dict(self) -> dict[str, Any]:
-        result: dict[str, Any] = {}
-
-        for platform, config_keys_dict in self._configs.items():
-            result[platform] = {}
-
-            for config_key, config in config_keys_dict.items():
-                result[platform][config_key] = json.loads(config.to_json())
+        return sorted(platform_dict.keys(), key=str)
+
+    def to_config_entries(self) -> dict[str, list[dict[str, Any]]]:
+        """Serialize to config entries format for JSON output."""
+        result: dict[str, list[dict[str, Any]]] = {}
+        for platform, config_dict in self._configs.items():
+            pairs: list[dict[str, Any]] = []
+            for config_key, config in config_dict.items():
+                config_data = json.loads(config.to_json())
+                pairs.append({"key": dict(config_key), "config": config_data})
+            result[platform] = pairs
+        return result
 
+    def to_dict(self) -> dict[str, dict[CaseKey, Any]]:
+        """Return configs as a nested dict (platform -> key -> config)."""
+        result: dict[str, dict[CaseKey, Any]] = {}
+        for platform, config_dict in self._configs.items():
+            result[platform] = {
+                k: json.loads(v.to_json()) for k, v in config_dict.items()
+            }
         return result
 
     @classmethod
-    def from_dict(cls, kernel_name: str, data: dict[str, Any]) -> "ConfigSet":
+    def from_dict(cls, kernel_name: str, data: dict[str, Any]) -> ConfigSet:
         config_set = cls(kernel_name)
         count = 0
 
@@ -117,9 +129,11 @@ def from_dict(cls, kernel_name: str, data: dict[str, Any]) -> "ConfigSet":
             if platform not in config_set._configs:
                 config_set._configs[platform] = {}
 
-            for config_key, config_data in platform_data.items():
-                config = helion.Config(**config_data)
-                config_set._configs[platform][config_key] = config
+            for entry in platform_data:
+                raw_key = entry["key"]
+                key = CaseKey.default() if not raw_key else CaseKey(raw_key)
+                config = helion.Config(**entry["config"])
+                config_set._configs[platform][key] = config
                 count += 1
 
         if count > 0:
@@ -132,7 +146,10 @@ def from_dict(cls, kernel_name: str, data: dict[str, Any]) -> "ConfigSet":
         return config_set
 
     def set_config(
-        self, platform: str, config_key: str, config: "helion.Config"
+        self,
+        platform: str,
+        config_key: CaseKey,
+        config: helion.Config,
     ) -> None:
         platform = platform.lower()
         if platform not in self._configs:
@@ -145,7 +162,7 @@ def set_config(
             config_key,
         )
 
-    def has_config(self, platform: str, config_key: str) -> bool:
+    def has_config(self, platform: str, config_key: CaseKey) -> bool:
         platform = platform.lower()
         platform_dict = self._configs.get(platform)
         if platform_dict is None:
@@ -156,18 +173,18 @@ def has_config(self, platform: str, config_key: str) -> bool:
 class ConfigManager:
     """File-level configuration management for Helion kernels (global singleton)."""
 
-    _instance: "ConfigManager | None" = None
+    _instance: ConfigManager | None = None
     _instance_base_dir: Path | None = None
 
-    def __new__(cls, base_dir: str | Path | None = None) -> "ConfigManager":
+    def __new__(cls, base_dir: str | Path | None = None) -> ConfigManager:
         resolved_base_dir = cls._resolve_base_dir(base_dir)
 
         if cls._instance is not None:
             if cls._instance_base_dir != resolved_base_dir:
                 raise ValueError(
                     f"ConfigManager singleton already exists with base_dir "
-                    f"'{cls._instance_base_dir}', cannot create with different "
-                    f"base_dir '{resolved_base_dir}'"
+                    f"'{cls._instance_base_dir}', cannot create with "
+                    f"different base_dir '{resolved_base_dir}'"
                 )
             return cls._instance
 
@@ -190,7 +207,7 @@ def _resolve_base_dir(base_dir: str | Path | None) -> Path:
         return (Path(__file__).parent / "configs").resolve()
 
     @classmethod
-    def get_instance(cls) -> "ConfigManager":
+    def get_instance(cls) -> ConfigManager:
         if cls._instance is None:
             raise RuntimeError(
                 "ConfigManager instance has not been created. "
@@ -229,16 +246,16 @@ def ensure_base_dir_writable(self) -> None:
                 f"Config directory '{self._base_dir}' is not writable: {e}"
             ) from e
 
-    def _load_platform_file(self, kernel_name: str, platform: str) -> dict[str, Any]:
+    def _load_platform_file(self, kernel_name: str, platform: str) -> Any:
         config_path = self.get_config_file_path(kernel_name, platform)
         if not config_path.exists():
-            return {}
+            return []
         try:
             with open(config_path) as f:
                 return json.load(f)
         except (json.JSONDecodeError, OSError) as e:
             logger.error("Failed to load config file %s: %s", config_path, e)
-            return {}
+            return []
 
     def load_config_set(self, kernel_name: str) -> ConfigSet:
         kernel_dir = self.get_kernel_dir(kernel_name)
@@ -253,32 +270,36 @@ def load_config_set(self, kernel_name: str) -> ConfigSet:
                     platform_data = json.load(f)
                 data[platform] = platform_data
             except (json.JSONDecodeError, OSError) as e:
-                logger.error("Failed to load config file %s: %s", platform_file, e)
+                logger.error(
+                    "Failed to load config file %s: %s",
+                    platform_file,
+                    e,
+                )
 
         return ConfigSet.from_dict(kernel_name, data)
 
     def get_platform_configs(
         self, kernel_name: str, platform: str
-    ) -> dict[str, helion.Config]:
+    ) -> dict[CaseKey, helion.Config]:
         platform_data = self._load_platform_file(kernel_name, platform)
         if not platform_data:
             return {}
         config_set = ConfigSet.from_dict(kernel_name, {platform: platform_data})
-        config_keys = config_set.get_config_keys(platform)
         return {
-            config_key: config_set.get_config(platform, config_key)
-            for config_key in config_keys
+            k: config_set.get_config(platform, k)
+            for k in config_set.get_config_keys(platform)
         }
 
     def save_config_set(self, config_set: ConfigSet) -> Path:
         kernel_dir = self.get_kernel_dir(config_set.kernel_name)
         kernel_dir.mkdir(parents=True, exist_ok=True)
 
-        full_data = config_set.to_dict()
-        for platform, platform_data in full_data.items():
+        full_data = config_set.to_config_entries()
+        for platform, pairs in full_data.items():
             platform_path = kernel_dir / f"{platform}.json"
             with open(platform_path, "w") as f:
-                json.dump(platform_data, f, indent=2)
+                json.dump(pairs, f, indent=2)
+                f.write("\n")
             logger.info("Saved config to: %s", platform_path)
 
         return kernel_dir
@@ -287,21 +308,34 @@ def save_configs(
         self,
         kernel_name: str,
         platform: str,
-        configs: dict[str, "helion.Config"],
+        configs: dict[CaseKey, helion.Config],
     ) -> Path:
         """Save configs for a kernel/platform, merging with existing."""
-        platform_data = self._load_platform_file(kernel_name, platform)
-        for config_key, config in configs.items():
-            platform_data[config_key] = json.loads(config.to_json())
+        config_set = ConfigSet.from_dict(
+            kernel_name,
+            {platform: self._load_platform_file(kernel_name, platform)},
+        )
+        for key, config in configs.items():
+            config_set.set_config(platform, key, config)
 
+        pairs = config_set.to_config_entries().get(platform, [])
         platform_path = self.get_config_file_path(kernel_name, platform)
         platform_path.parent.mkdir(parents=True, exist_ok=True)
         with open(platform_path, "w") as f:
-            json.dump(platform_data, f, indent=2)
+            json.dump(pairs, f, indent=2)
+            f.write("\n")
 
         logger.info("Saved config to: %s", platform_path)
         return platform_path
 
-    def config_exists(self, kernel_name: str, platform: str, config_key: str) -> bool:
+    def config_exists(
+        self,
+        kernel_name: str,
+        platform: str,
+        config_key: CaseKey,
+    ) -> bool:
         platform_data = self._load_platform_file(kernel_name, platform)
-        return config_key in platform_data
+        if not platform_data:
+            return False
+        target = dict(config_key)
+        return any(entry["key"] == target for entry in platform_data)
diff --git a/vllm/kernels/helion/configs/silu_mul_fp8/nvidia_h100.json b/vllm/kernels/helion/configs/silu_mul_fp8/nvidia_h100.json
index c314eb2dab86..4dc5c2cab308 100644
--- a/vllm/kernels/helion/configs/silu_mul_fp8/nvidia_h100.json
+++ b/vllm/kernels/helion/configs/silu_mul_fp8/nvidia_h100.json
@@ -1,13866 +1,15711 @@
-{
-  "intermediate_2048_numtokens_256": {
-    "block_sizes": [
-      64,
-      32
-    ],
-    "loop_orders": [
-      [
-        0,
-        1
-      ]
-    ],
-    "flatten_loops": [
-      true
-    ],
-    "l2_groupings": [
-      1
-    ],
-    "range_unroll_factors": [
-      0
-    ],
-    "range_warp_specializes": [],
-    "range_num_stages": [
-      0
-    ],
-    "range_multi_buffers": [
-      null
-    ],
-    "range_flattens": [
-      null
-    ],
-    "load_eviction_policies": [
-      "",
-      "",
-      ""
-    ],
-    "num_warps": 8,
-    "num_stages": 1,
-    "indexing": [
-      "pointer",
-      "pointer",
-      "pointer",
-      "pointer"
-    ],
-    "pid_type": "flat"
-  },
-  "intermediate_4096_numtokens_256": {
-    "block_sizes": [
-      256,
-      32
-    ],
-    "loop_orders": [
-      [
-        0,
-        1
-      ]
-    ],
-    "flatten_loops": [
-      true
-    ],
-    "l2_groupings": [
-      1
-    ],
-    "range_unroll_factors": [
-      0
-    ],
-    "range_warp_specializes": [],
-    "range_num_stages": [
-      0
-    ],
-    "range_multi_buffers": [
-      null
-    ],
-    "range_flattens": [
-      null
-    ],
-    "load_eviction_policies": [
-      "",
-      "",
-      ""
-    ],
-    "num_warps": 8,
-    "num_stages": 1,
-    "indexing": [
-      "pointer",
-      "pointer",
-      "pointer",
-      "pointer"
-    ],
-    "pid_type": "flat"
-  },
-  "default": {
-    "block_sizes": [
-      1,
-      512
-    ],
-    "loop_orders": [
-      [
+[
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 256
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 256
+    },
+    "config": {
+      "block_sizes": [
+        256,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {},
+    "config": {
+      "block_sizes": [
+        1,
+        512
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "first",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 2,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 256
+    },
+    "config": {
+      "block_sizes": [
+        256,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 256
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        4096
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 256
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 7688,
+      "numtokens": 256
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 256
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 1
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 1
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 1
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 1
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 1
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 1
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 2
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 2
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 2
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 2
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 2
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        16384
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "first",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 3,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "xyz"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 2
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 4
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 4
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "first"
+      ],
+      "num_warps": 8,
+      "num_stages": 7,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "xyz"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 4
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 4
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 8,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 4
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        2048
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        "last"
+      ],
+      "num_warps": 8,
+      "num_stages": 6,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "xyz"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 4
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 2,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "xyz"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 8
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 8
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "last",
+        ""
+      ],
+      "num_warps": 16,
+      "num_stages": 5,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "xyz"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 8
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 8
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        ""
+      ],
+      "num_warps": 1,
+      "num_stages": 8,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 8
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        1024
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "",
+        "first"
+      ],
+      "num_warps": 2,
+      "num_stages": 5,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 8
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 16
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "last",
+        "first"
+      ],
+      "num_warps": 16,
+      "num_stages": 2,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "xyz"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 16
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 16
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 16
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 16
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 16
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        256
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "last",
+        "last"
+      ],
+      "num_warps": 16,
+      "num_stages": 7,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 24
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        "first"
+      ],
+      "num_warps": 4,
+      "num_stages": 8,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 24
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        "first"
+      ],
+      "num_warps": 16,
+      "num_stages": 3,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 24
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 24
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 5,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 24
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        1024
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 2,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 24
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "first",
+        "first"
+      ],
+      "num_warps": 32,
+      "num_stages": 3,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 32
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        16
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 32
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 32
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        4096
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        ""
+      ],
+      "num_warps": 32,
+      "num_stages": 5,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 32
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 3,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 32
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 32
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 3,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 40
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 40
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        4096
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "last"
+      ],
+      "num_warps": 8,
+      "num_stages": 4,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 40
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 40
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "last"
+      ],
+      "num_warps": 2,
+      "num_stages": 2,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 40
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 40
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        1
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        "last"
+      ],
+      "num_warps": 16,
+      "num_stages": 5,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "persistent_interleaved",
+      "num_sm_multiplier": 32,
+      "maxnreg": 32
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 48
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 48
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 48
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 48
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        4096
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        "last"
+      ],
+      "num_warps": 4,
+      "num_stages": 2,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 48
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 48
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 56
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 56
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 56
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 56
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 56
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        8192
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 56
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        4096
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "first",
+        ""
+      ],
+      "num_warps": 2,
+      "num_stages": 4,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 64
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 64
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 64
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 64
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 64
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        8192
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 64
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "first",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 72
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 72
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 72
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        "first"
+      ],
+      "num_warps": 16,
+      "num_stages": 2,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 72
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 72
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 5,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 72
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "",
+        "last"
+      ],
+      "num_warps": 8,
+      "num_stages": 3,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 80
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 80
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 80
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 80
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "last",
+        "last"
+      ],
+      "num_warps": 16,
+      "num_stages": 2,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 80
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        4096
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "last",
+        ""
+      ],
+      "num_warps": 16,
+      "num_stages": 2,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 80
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "last",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 6,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 88
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 88
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 88
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 88
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 88
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "first",
+        ""
+      ],
+      "num_warps": 32,
+      "num_stages": 2,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 88
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 32,
+      "num_stages": 5,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 96
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 96
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 96
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        4096
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        "last"
+      ],
+      "num_warps": 4,
+      "num_stages": 6,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 96
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 96
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        2048
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        ""
+      ],
+      "num_warps": 32,
+      "num_stages": 2,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 96
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        4096
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "first",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 1,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 104
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 104
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 104
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 104
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 104
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        8192
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "first"
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 104
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "last",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 112
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 112
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 112
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 112
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        "last"
+      ],
+      "num_warps": 16,
+      "num_stages": 3,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 112
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 3,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 112
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 120
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        ""
+      ],
+      "num_warps": 4,
+      "num_stages": 6,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 120
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 120
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 120
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 120
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        "last"
+      ],
+      "num_warps": 1,
+      "num_stages": 6,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 120
+    },
+    "config": {
+      "block_sizes": [
+        32,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "first",
+        "last"
+      ],
+      "num_warps": 16,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 128
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        16
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 128
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 128
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 128
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 128
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        "last"
+      ],
+      "num_warps": 2,
+      "num_stages": 1,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 128
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        4096
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 136
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 136
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 136
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 136
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        ""
+      ],
+      "num_warps": 32,
+      "num_stages": 7,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 136
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        8192
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        8
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "last"
+      ],
+      "num_warps": 8,
+      "num_stages": 4,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 136
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        16384
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        8
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 6,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 144
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "last",
+        "first"
+      ],
+      "num_warps": 16,
+      "num_stages": 7,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 144
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 144
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 144
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        2048
+      ],
+      "loop_orders": [
+        [
+          1,
+          0
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "first"
+      ],
+      "num_warps": 1,
+      "num_stages": 4,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 144
+    },
+    "config": {
+      "block_sizes": [
+        256,
+        16
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "first",
+        "first"
+      ],
+      "num_warps": 16,
+      "num_stages": 2,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 144
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "",
+        "last"
+      ],
+      "num_warps": 16,
+      "num_stages": 8,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 152
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        8
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        "first"
+      ],
+      "num_warps": 8,
+      "num_stages": 7,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 152
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 152
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 152
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        16
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        "first"
+      ],
+      "num_warps": 1,
+      "num_stages": 2,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 152
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        4096
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        "first"
+      ],
+      "num_warps": 4,
+      "num_stages": 6,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 152
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        16384
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        ""
+      ],
+      "num_warps": 16,
+      "num_stages": 6,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 160
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "last",
+        "last"
+      ],
+      "num_warps": 1,
+      "num_stages": 3,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 160
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 160
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 160
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 32,
+      "num_stages": 4,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 160
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        4
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "first",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 8,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 160
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        1024
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        8
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        "first"
+      ],
+      "num_warps": 8,
+      "num_stages": 8,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 168
+    },
+    "config": {
+      "block_sizes": [
+        4,
+        2048
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "last",
+        "",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 8,
+      "indexing": [
+        "pointer",
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 168
+    },
+    "config": {
+      "block_sizes": [
+        8,
+        512
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 168
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 168
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        128
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 168
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 168
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        64
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        "first"
+      ],
+      "num_warps": 2,
+      "num_stages": 6,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 176
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        32
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 176
+    },
+    "config": {
+      "block_sizes": [
+        16,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 176
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        4
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        "first"
+      ],
+      "num_warps": 4,
+      "num_stages": 6,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 176
+    },
+    "config": {
+      "block_sizes": [
+        1,
+        8192
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        false
+      ],
+      "l2_groupings": [
+        16
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "",
+        "first"
+      ],
+      "num_warps": 16,
+      "num_stages": 5,
+      "indexing": [
+        "tensor_descriptor",
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 176
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 14336,
+      "numtokens": 176
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2048,
+      "numtokens": 184
+    },
+    "config": {
+      "block_sizes": [
+        2,
+        256
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        2
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 16,
+      "num_stages": 6,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 2880,
+      "numtokens": 184
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 4096,
+      "numtokens": 184
+    },
+    "config": {
+      "block_sizes": [
+        128,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        1
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "",
+        "",
+        ""
+      ],
+      "num_warps": 8,
+      "num_stages": 1,
+      "indexing": [
+        "pointer",
+        "pointer",
+        "pointer",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 8192,
+      "numtokens": 184
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        64
+      ],
+      "loop_orders": [
+        [
+          0,
+          1
+        ]
+      ],
+      "flatten_loops": [
+        true
+      ],
+      "l2_groupings": [
+        32
+      ],
+      "range_unroll_factors": [
+        0
+      ],
+      "range_warp_specializes": [],
+      "range_num_stages": [
+        0
+      ],
+      "range_multi_buffers": [
+        null
+      ],
+      "range_flattens": [
+        null
+      ],
+      "load_eviction_policies": [
+        "first",
+        "last",
+        "last"
+      ],
+      "num_warps": 32,
+      "num_stages": 8,
+      "indexing": [
+        "tensor_descriptor",
+        "pointer",
+        "tensor_descriptor",
+        "pointer"
+      ],
+      "pid_type": "flat"
+    }
+  },
+  {
+    "key": {
+      "intermediate": 11008,
+      "numtokens": 184
+    },
+    "config": {
+      "block_sizes": [
+        64,
+        256
+      ],
+      "loop_orders":

*(diff truncated)*
```
