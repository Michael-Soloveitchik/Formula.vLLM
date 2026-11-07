# PR #4690: [Core] Implement sharded state loader

- field: scheduling  | area: examples
- author: aurickq
- state: MERGED  created: 2024-05-08T19:24:47Z  merged: 2024-05-16T05:11:55Z
- url: https://github.com/vllm-project/vllm/pull/4690
- labels: 

# Issue

## Problem

When loading a tensor-parallel model with DistributedGPUExecutor, every worker reads the entire model checkpoint from disk even though each worker only keeps a small shard after TP splitting. This duplicates full-checkpoint I/O across all workers, so disk read volume scales with the number of workers while usable data per worker stays tiny. There is no load path that lets a worker read only its own shard.

## Example

Loading Arctic with 8x tensor parallelism from NVMe: each of the 8 workers reads the full checkpoint, giving ~10x slower load than reading only per-worker shards.

## Use case

Starting up / cold-loading large multi-GPU LLMs for serving, especially models too big to fit on one GPU that must be sharded across many workers.

## Resources

Disk/NVMe read bandwidth and model-load latency; also host memory and CPU time spent parsing the full checkpoint on every worker.

## Type of users

Users serving large models with multi-GPU tensor parallelism (high TP degree), and users of post-load quantization like DeepSpeed FP_Quantize who want a pre-quantized checkpoint.

## Kind of inputs

Large model checkpoints (many GB) loaded under high tensor-parallel-size, stored on local disk/NVMe, optionally quantized after load.

## Mechanism with the issue

Default loaders in model_loader/loader.py read the whole checkpoint per worker then discard non-local shards. No LoadFormat exists for per-rank shard files, so DistributedGPUExecutor has no way to dispatch a per-worker save or a shard-only load. Full-checkpoint read repeated N times = N x disk I/O.

# Solution

## Fix in words

Adds a new `SHARDED_STATE` LoadFormat and a `ShardedStateLoader` in loader.py. `save_model` writes each TP worker's filtered state_dict to per-rank safetensors files (`model-rank-{rank}-part-{part}.safetensors`), splitting into parts by `max_size`; `_filter_subtensors` drops tensors that alias/subset another tensor's storage so shared memory isn't saved twice. `load_model` globs only the current rank's shard files via `get_tensor_model_parallel_rank()` and copies tensors straight into model params (narrowing param views to handle LoRA padding), raising if keys are missing or unsharded. `DistributedGPUExecutor.save_sharded_state` fans the save out to all workers via `_run_workers`, and `examples/save_sharded_state.py` drives save + copies metadata files. Net effect: each worker reads only its own shard, giving ~10x faster load at 8x TP.

## Author's solution

New model loader that directly loads each worker's sharded state when using DistributedGPUExecutor, so under tensor parallelism a worker reads only its own shard instead of the full checkpoint — ~10x load speedup from NVMe at 8x TP in Arctic tests. Also enables saving a pre-quantized checkpoint (e.g. DeepSpeed FP_Quantize) loaded directly into each worker. Split out from PR #4652.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `examples/save_sharded_state.py`
- `tests/test_sharded_state_loader.py`
- `vllm/config.py`
- `vllm/executor/distributed_gpu_executor.py`
- `vllm/model_executor/model_loader/loader.py`
- `vllm/worker/model_runner.py`
- `vllm/worker/worker.py`

### Diff
```diff
diff --git a/examples/save_sharded_state.py b/examples/save_sharded_state.py
new file mode 100644
index 000000000000..c595d98ba275
--- /dev/null
+++ b/examples/save_sharded_state.py
@@ -0,0 +1,75 @@
+"""
+Saves each worker's model state dict directly to a checkpoint, which enables a
+fast load path for large tensor-parallel models where each worker only needs to
+read its own shard rather than the entire checkpoint.
+
+Example usage:
+
+python save_sharded_state.py \
+    --model /path/to/load \
+    --quantization deepspeedfp \
+    --tensor-parallel-size 8 \
+    --output /path/to/save
+
+Then, the model can be loaded with
+
+llm = LLM(
+    model="/path/to/save",
+    load_format="sharded_state",
+    quantization="deepspeedfp",
+    tensor_parallel_size=8,
+)
+"""
+import argparse
+import dataclasses
+import os
+import shutil
+from pathlib import Path
+
+from vllm import LLM, EngineArgs
+
+parser = argparse.ArgumentParser()
+EngineArgs.add_cli_args(parser)
+parser.add_argument("--output",
+                    "-o",
+                    required=True,
+                    type=str,
+                    help="path to output checkpoint")
+parser.add_argument("--file-pattern",
+                    type=str,
+                    help="string pattern of saved filenames")
+parser.add_argument("--max-file-size",
+                    type=str,
+                    default=5 * 1024**3,
+                    help="max size (in bytes) of each safetensors file")
+
+
+def main(args):
+    engine_args = EngineArgs.from_cli_args(args)
+    if engine_args.enable_lora:
+        raise ValueError("Saving with enable_lora=True is not supported!")
+    model_path = engine_args.model
+    if not Path(model_path).is_dir():
+        raise ValueError("model path must be a local directory")
+    # Create LLM instance from arguments
+    llm = LLM(**dataclasses.asdict(engine_args))
+    # Prepare output directory
+    Path(args.output).mkdir(exist_ok=True)
+    # Dump worker states to output directory
+    model_executor = llm.llm_engine.model_executor
+    model_executor.save_sharded_state(path=args.output,
+                                      pattern=args.file_pattern,
+                                      max_size=args.max_file_size)
+    # Copy metadata files to output directory
+    for file in os.listdir(model_path):
+        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
+            if os.path.isdir(os.path.join(model_path, file)):
+                shutil.copytree(os.path.join(model_path, file),
+                                os.path.join(args.output, file))
+            else:
+                shutil.copy(os.path.join(model_path, file), args.output)
+
+
+if __name__ == "__main__":
+    args = parser.parse_args()
+    main(args)
diff --git a/tests/test_sharded_state_loader.py b/tests/test_sharded_state_loader.py
new file mode 100644
index 000000000000..8540e98da366
--- /dev/null
+++ b/tests/test_sharded_state_loader.py
@@ -0,0 +1,90 @@
+import os
+import shutil
+from tempfile import TemporaryDirectory
+
+import pytest
+import torch
+from huggingface_hub import snapshot_download
+
+from vllm import LLM, SamplingParams
+from vllm.model_executor.model_loader.loader import ShardedStateLoader
+
+prompts = [
+    "Hello, my name is",
+    "The president of the United States is",
+    "The capital of France is",
+    "The future of AI is",
+]
+
+# Create a sampling params object.
+sampling_params = SamplingParams(
+    temperature=0.8,
+    top_p=0.95,
+    seed=0,
+    max_tokens=256,
+    ignore_eos=True,
+)
+
+
+def test_filter_subtensors():
+    state_dict = {
+        "a": torch.empty(2),
+        "b": torch.empty((2, 4)),
+        "c": torch.empty((2, 4, 8)),
+    }
+    state_dict.update({
+        "x": state_dict["b"],
+        "y": state_dict["c"][1, 2, :],
+        "z": state_dict["c"][1, :, 4],
+    })
+    filtered_state_dict = ShardedStateLoader._filter_subtensors(state_dict)
+    assert tuple(filtered_state_dict.keys()) == ("a", "b", "c")
+    for key, tensor in filtered_state_dict.items():
+        assert tensor.equal(state_dict[key])
+
+
+@pytest.mark.parametrize("enable_lora", [False, True])
+def test_sharded_state_loader(enable_lora):
+    weights_patterns = ("*.bin", "*.pt", "*.safetensors")
+
+    with TemporaryDirectory() as cache_dir, TemporaryDirectory() as output_dir:
+        input_dir = snapshot_download("meta-llama/Llama-2-7b-hf",
+                                      cache_dir=cache_dir)
+
+        llm = LLM(
+            model=input_dir,
+            worker_use_ray=True,
+            gpu_memory_utilization=0.3,
+        )
+
+        # Dump worker states to output directory
+        model_executor = llm.llm_engine.model_executor
+        model_executor.save_sharded_state(path=output_dir)
+        # Copy metadata files to output directory
+        for file in os.listdir(input_dir):
+            if not any(file.endswith(ext) for ext in weights_patterns):
+                shutil.copy(f"{input_dir}/{file}", output_dir)
+        del llm.llm_engine.model_executor
+
+        llm_before = LLM(
+            model=input_dir,
+            worker_use_ray=True,
+            enable_lora=enable_lora,
+            gpu_memory_utilization=0.3,
+        )
+        gen_before = llm_before.generate(prompts, sampling_params)
+        out_before = [gen.outputs[0].__dict__ for gen in gen_before]
+        del llm_before.llm_engine.model_executor
+
+        llm_after = LLM(
+            model=output_dir,
+            worker_use_ray=True,
+            enable_lora=enable_lora,
+            gpu_memory_utilization=0.3,
+            load_format="sharded_state",
+        )
+        gen_after = llm_after.generate(prompts, sampling_params)
+        out_after = [gen.outputs[0].__dict__ for gen in gen_after]
+        del llm_after.llm_engine.model_executor
+
+        assert out_before == out_after
diff --git a/vllm/config.py b/vllm/config.py
index 435f47dc9459..620b1ca4296e 100644
--- a/vllm/config.py
+++ b/vllm/config.py
@@ -463,6 +463,7 @@ class LoadFormat(str, enum.Enum):
     NPCACHE = "npcache"
     DUMMY = "dummy"
     TENSORIZER = "tensorizer"
+    SHARDED_STATE = "sharded_state"
 
 
 @dataclass
diff --git a/vllm/executor/distributed_gpu_executor.py b/vllm/executor/distributed_gpu_executor.py
index 4c922ef63ee0..c5b1e61112af 100644
--- a/vllm/executor/distributed_gpu_executor.py
+++ b/vllm/executor/distributed_gpu_executor.py
@@ -77,6 +77,17 @@ def remove_lora(self, lora_id: int) -> bool:
     def list_loras(self) -> Set[int]:
         return self._run_workers("list_loras")
 
+    def save_sharded_state(
+        self,
+        path: str,
+        pattern: Optional[str] = None,
+        max_size: Optional[int] = None,
+    ) -> None:
+        self._run_workers("save_sharded_state",
+                          path=path,
+                          pattern=pattern,
+                          max_size=max_size)
+
     @abstractmethod
     def _run_workers(
         self,
diff --git a/vllm/model_executor/model_loader/loader.py b/vllm/model_executor/model_loader/loader.py
index b14824a359b6..dc568928b285 100644
--- a/vllm/model_executor/model_loader/loader.py
+++ b/vllm/model_executor/model_loader/loader.py
@@ -1,4 +1,5 @@
 # ruff: noqa: SIM117
+import collections
 import copy
 import glob
 import os
@@ -366,6 +367,150 @@ def load_model(self, *, model_config: ModelConfig,
                                                cache_config)
 
 
+class ShardedStateLoader(BaseModelLoader):
+    """
+    Model loader that directly loads each worker's model state dict, which
+    enables a fast load path for large tensor-parallel models where each worker
+    only needs to read its own shard rather than the entire checkpoint. See
+    `examples/save_sharded_states.py` for creating a sharded checkpoint.
+    """
+
+    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
+
+    def __init__(self, load_config: LoadConfig):
+        super().__init__(load_config)
+        extra_config = ({} if load_config.model_loader_extra_config is None
+                        else load_config.model_loader_extra_config.copy())
+        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
+        if extra_config:
+            raise ValueError(f"Unexpected extra config keys for load format "
+                             f"{load_config.load_format}: "
+                             f"{load_config.model_loader_extra_config.keys()}")
+
+    @staticmethod
+    def _filter_subtensors(
+            tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
+        """
+        Filter out all tensors that share the same memory or a subset of the
+        memory of another tensor.
+        """
+        same_storage_groups = collections.defaultdict(list)
+        for key, tensor in tensors.items():
+            if tensor.numel():
+                ptr = tensor.untyped_storage().data_ptr()
+                same_storage_groups[tensor.device, ptr].append((key, tensor))
+
+        def get_end_ptr(tensor: torch.Tensor) -> int:
+            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()
+
+        result = {}
+        for group in same_storage_groups.values():
+            for k, t in group:
+                a, b = t.data_ptr(), get_end_ptr(t)
+                for k2, t2 in group:
+                    if not t2.is_contiguous():
+                        continue
+                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
+                    if a < a2 or b2 < b:
+                        continue
+                    if a2 < a or b < b2 or not t.is_contiguous():
+                        break  # t2 covers strictly more memory than t.
+                    if k2 < k:
+                        # Same tensors, keep the one with the smaller key.
+                        break
+                else:
+                    result[k] = t
+        return result
+
+    def load_model(self, *, model_config: ModelConfig,
+                   device_config: DeviceConfig,
+                   lora_config: Optional[LoRAConfig],
+                   vision_language_config: Optional[VisionLanguageConfig],
+                   parallel_config: ParallelConfig,
+                   scheduler_config: SchedulerConfig,
+                   cache_config: CacheConfig) -> nn.Module:
+        from safetensors.torch import safe_open
+
+        from vllm.distributed import get_tensor_model_parallel_rank
+        with set_default_torch_dtype(model_config.dtype):
+            with torch.device(device_config.device):
+                model = _initialize_model(model_config, self.load_config,
+                                          lora_config, vision_language_config,
+                                          cache_config)
+            rank = get_tensor_model_parallel_rank()
+            pattern = os.path.join(
+                model_config.model,
+                self.pattern.format(rank=rank, part="*"),
+            )
+            filepaths = glob.glob(pattern)
+            if not filepaths:
+                # TODO: support un-sharded checkpoints too
+                raise ValueError(
+                    f"Could not find checkpoint files '{pattern}', only "
+                    f"pre-sharded checkpoints are currently supported!")
+            state_dict = self._filter_subtensors(model.state_dict())
+            for path in filepaths:
+                with safe_open(path, framework="pt") as f:
+                    for key in f.keys():  # noqa: SIM118
+                        tensor = f.get_tensor(key)
+                        # If loading with LoRA enabled, additional padding may
+                        # be added to certain parameters. We only load into a
+                        # narrowed view of the parameter data.
+                        param_data = state_dict[key].data
+                        param_shape = state_dict[key].shape
+                        for dim, size in enumerate(tensor.shape):
+                            if size < param_shape[dim]:
+                                param_data = param_data.narrow(dim, 0, size)
+                        if tensor.shape != param_shape:
+                            logger.warning(
+                                "loading tensor of shape %s into "
+                                "parameter '%s' of shape %s", tensor.shape,
+                                key, param_shape)
+                        param_data.copy_(tensor)
+                        state_dict.pop(key)
+            if state_dict:
+                raise ValueError(
+                    f"Missing keys {tuple(state_dict)} in loaded state!")
+        return model.eval()
+
+    @staticmethod
+    def save_model(
+        model: torch.nn.Module,
+        path: str,
+        pattern: Optional[str] = None,
+        max_size: Optional[int] = None,
+    ) -> None:
+        from safetensors.torch import save_file
+
+        from vllm.distributed import get_tensor_model_parallel_rank
+        if pattern is None:
+            pattern = ShardedStateLoader.DEFAULT_PATTERN
+        rank = get_tensor_model_parallel_rank()
+        part_idx = 0
+        total_size = 0
+        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
+        state_dict_part: Dict[str, torch.Tensor] = {}
+        for key, tensor in state_dict.items():
+            param_size = tensor.nelement() * tensor.element_size()
+            if max_size is not None and total_size + param_size > max_size:
+                filename = pattern.format(rank=rank, part=part_idx)
+                save_file(
+                    state_dict_part,
+                    os.path.join(path, filename),
+                )
+                part_idx += 1
+                total_size = 0
+                state_dict_part = {}
+            state_dict_part[key] = tensor
+            total_size += param_size
+        if len(state_dict_part) > 0:
+            filename = pattern.format(rank=rank, part=part_idx)
+            save_file(
+                state_dict_part,
+                os.path.join(path, filename),
+            )
+
+
 def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
     """Get a model loader based on the load format."""
 
@@ -378,4 +523,7 @@ def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
     if load_config.load_format == LoadFormat.TENSORIZER:
         return TensorizerLoader(load_config)
 
+    if load_config.load_format == LoadFormat.SHARDED_STATE:
+        return ShardedStateLoader(load_config)
+
     return DefaultModelLoader(load_config)
diff --git a/vllm/worker/model_runner.py b/vllm/worker/model_runner.py
index 3f7e87c1de48..f4fcdc76a417 100644
--- a/vllm/worker/model_runner.py
+++ b/vllm/worker/model_runner.py
@@ -212,6 +212,20 @@ def load_model(self) -> None:
                            "but the KV cache data type is not FP8. "
                            "KV cache scaling factors will not be used.")
 
+    def save_sharded_state(
+        self,
+        path: str,
+        pattern: Optional[str] = None,
+        max_size: Optional[int] = None,
+    ) -> None:
+        from vllm.model_executor.model_loader.loader import ShardedStateLoader
+        ShardedStateLoader.save_model(
+            self.model,
+            path,
+            pattern=pattern,
+            max_size=max_size,
+        )
+
     def get_max_block_per_batch(self) -> int:
         block_size = self.block_size
         return (self.max_seq_len_to_capture + block_size - 1) // block_size
diff --git a/vllm/worker/worker.py b/vllm/worker/worker.py
index 82cf58101a95..faea50fbfbf5 100644
--- a/vllm/worker/worker.py
+++ b/vllm/worker/worker.py
@@ -119,6 +119,18 @@ def init_device(self) -> None:
     def load_model(self):
         self.model_runner.load_model()
 
+    def save_sharded_state(
+        self,
+        path: str,
+        pattern: Optional[str] = None,
+        max_size: Optional[int] = None,
+    ) -> None:
+        self.model_runner.save_sharded_state(
+            path,
+            pattern=pattern,
+            max_size=max_size,
+        )
+
     @torch.inference_mode()
     def determine_num_available_blocks(self) -> Tuple[int, int]:
         """Profiles the peak memory usage of the model to determine how many

```
