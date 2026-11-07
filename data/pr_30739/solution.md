# PR #30739: [BugFix] Support online dense model DP without overhead

- field: scheduling  | area: v1_engine
- author: njhill
- state: MERGED  created: 2025-12-16T04:08:06Z  merged: 2026-01-02T15:36:38Z
- url: https://github.com/vllm-project/vllm/pull/30739
- labels: ready, v1, kv-connector

# Issue

## Problem

When running non-MoE (dense) models under data parallelism in the vLLM v1 engine, the DP ranks are needlessly kept in lockstep. The engine performs redundant all-reduce ops to synchronize steps across ranks and coordinates "idle" ranks to run dummy forward passes so all ranks advance together. Dense models have no cross-rank expert communication, so this synchronization is pure overhead.

## Example

Serving Qwen/Qwen3-8B with --data-parallel-size 4 on 4xH100: the step/wave synchronization added ~43ms to mean TTFT (131.78ms vs 88.94ms after) and dropped request throughput from 40.31 to 38.31 req/s.

## Use case

Online/AsyncLLM serving of a standard dense transformer replicated across multiple GPUs for higher throughput via data parallel replicas behind a load balancer.

## Resources

Wastes GPU compute on dummy forward passes and burns inter-GPU bandwidth (NVLink/all-reduce) plus latency on cross-rank step synchronization that dense models do not require.

## Type of users

Users serving dense (non-MoE) models with --data-parallel-size > 1 in the online AsyncLLM path, typically multi-GPU single-node replica deployments.

## Kind of inputs

Any inference request load; impact grows with concurrency since every step incurs the sync. Benchmark used 4000 random prompts, 128 in / 512 out, max-concurrency 200.

## Mechanism with the issue

The DP coordinator and wave/step synchronization logic ran for all DP>1 configs regardless of model type. ParallelConfig propagated data_parallel_rank into process groups and compilation splitting ops as if MoE-style all-to-all coordination were needed, forcing lockstep stepping and dummy passes on dense ranks.

# Solution

## Fix in words

Adds ModelConfig.is_moe (now a property; also makes is_quantized a property) and threads it into ParallelConfig.is_moe_model, set during VllmConfig.__post_init__. Introduces VllmConfig.needs_dp_coordinator which returns True only for MoE models with DP>1 (wave coordination, even in external LB) or dense models in internal/hybrid LB mode (stats collection only) — so dense external-LB ranks skip the coordinator entirely. Compilation splitting ops now use an effective_dp_size of 1 for dense models, making the worker-level parallel config equivalent to DP=1. A new data_parallel_index field decouples the "identity" rank (used for cache dirs, KV side-channel ports, debug dump paths) from the DP rank used for torch process groups, which is no longer overridden for dense models. Offline DP with DP>1 now raises ValueError for dense models at startup.

## Author's solution

Change the worker-level parallel config to be equivalent to DP=1 for non-MoE models so each rank runs independently, eliminating the redundant all-reduce step synchronization and dummy forward passes. When internal load-balancing is used, keep the DP coordinator running to propagate engine stats for load balancing, but disable the step/wave synchronization. Supported only in the online/AsyncLLM case; offline DP now fails at startup for dense models.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/test_config.py`
- `tests/v1/engine/test_engine_core_client.py`
- `vllm/compilation/backends.py`
- `vllm/compilation/decorators.py`
- `vllm/config/model.py`
- `vllm/config/parallel.py`
- `vllm/config/vllm.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`
- `vllm/distributed/parallel_state.py`
- `vllm/engine/arg_utils.py`
- `vllm/forward_context.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/engine/coordinator.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/llm_engine.py`
- `vllm/v1/engine/utils.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_worker.py`

### Diff
```diff
diff --git a/tests/test_config.py b/tests/test_config.py
index ee706ab3d9c8..8d85d6827695 100644
--- a/tests/test_config.py
+++ b/tests/test_config.py
@@ -205,8 +205,8 @@ def test_default_pooling_type(model_id, default_pooling_type, pooling_type):
 )
 def test_moe_model_detection(model_id, expected_is_moe_model):
     model_config = ModelConfig(model_id)
-    # Just check that is_moe_model field exists and is a boolean
-    assert model_config.is_model_moe() == expected_is_moe_model
+    # Just check that is_moe field exists and is a boolean
+    assert model_config.is_moe == expected_is_moe_model
 
 
 @pytest.mark.parametrize(
@@ -224,7 +224,7 @@ def test_moe_model_detection(model_id, expected_is_moe_model):
 def test_is_quantized(model_id, quantized):
     model_config = ModelConfig(model_id)
     # Just check that quantized field exists and is a boolean
-    assert model_config.is_quantized() == quantized
+    assert model_config.is_quantized == quantized
 
 
 @pytest.mark.skipif(
@@ -925,7 +925,7 @@ def test_vllm_config_callable_defaults():
         model_config=quantized_model, optimization_level=OptimizationLevel.O2
     )
     enable_if_quantized = lambda cfg: (
-        cfg.model_config is not None and cfg.model_config.is_quantized()
+        cfg.model_config is not None and cfg.model_config.is_quantized
     )
     assert enable_if_quantized(config_quantized) is True
     assert enable_if_quantized(config_no_model) is False
@@ -936,7 +936,7 @@ def test_vllm_config_callable_defaults():
         model_config=moe_model, optimization_level=OptimizationLevel.O2
     )
     enable_if_sequential = lambda cfg: (
-        cfg.model_config is not None and not cfg.model_config.is_model_moe()
+        cfg.model_config is not None and not cfg.model_config.is_moe
     )
     assert enable_if_sequential(config_moe) is False
     assert enable_if_sequential(config_quantized) is True
@@ -1050,3 +1050,46 @@ def test_scheduler_config_init():
     with pytest.raises(AttributeError):
         # InitVar does not become an attribute
         print(SchedulerConfig.default_factory().max_model_len)
+
+
+@pytest.mark.parametrize(
+    (
+        "model_id",
+        "data_parallel_size",
+        "external_lb",
+        "expected_needs_coordinator",
+    ),
+    [
+        # Non-MoE model with DP=1 should not need coordinator
+        ("facebook/opt-125m", 1, False, False),
+        # Non-MoE model with DP>1 internal LB should need coordinator
+        ("facebook/opt-125m", 2, False, True),
+        # Non-MoE model with DP>1 external LB should not need coordinator
+        ("facebook/opt-125m", 2, True, False),
+        # MoE model with DP=1 should not need coordinator
+        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 1, False, False),
+        # MoE model with DP>1 internal LB should need both coordinator
+        # and wave coordination
+        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 2, False, True),
+        # MoE model with DP>1 external LB needs coordinator for wave coordination
+        # (wave coordination runs in coordinator process)
+        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 2, True, True),
+    ],
+)
+def test_needs_dp_coordination(
+    model_id,
+    data_parallel_size,
+    external_lb,
+    expected_needs_coordinator,
+):
+    """Test that DP coordinator and wave coordination are configured correctly."""
+    from vllm.config import ParallelConfig
+
+    model_config = ModelConfig(model_id)
+    parallel_config = ParallelConfig(
+        data_parallel_size=data_parallel_size,
+        data_parallel_external_lb=external_lb,
+    )
+    vllm_config = VllmConfig(model_config=model_config, parallel_config=parallel_config)
+
+    assert vllm_config.needs_dp_coordinator == expected_needs_coordinator
diff --git a/tests/v1/engine/test_engine_core_client.py b/tests/v1/engine/test_engine_core_client.py
index c8d25f9700bf..05f92b6aa785 100644
--- a/tests/v1/engine/test_engine_core_client.py
+++ b/tests/v1/engine/test_engine_core_client.py
@@ -133,6 +133,7 @@ def setsockopt(self, *_args, **_kwargs):
     parallel_config = SimpleNamespace(
         data_parallel_size=1,
         data_parallel_rank=0,
+        data_parallel_index=0,
         data_parallel_size_local=1,
         data_parallel_rank_local=None,
         data_parallel_hybrid_lb=False,
diff --git a/vllm/compilation/backends.py b/vllm/compilation/backends.py
index 2fb6265560b1..4f855fc1d7c2 100644
--- a/vllm/compilation/backends.py
+++ b/vllm/compilation/backends.py
@@ -630,7 +630,7 @@ def __call__(
         os.makedirs(cache_dir, exist_ok=True)
         self.compilation_config.cache_dir = cache_dir
         rank = vllm_config.parallel_config.rank
-        dp_rank = vllm_config.parallel_config.data_parallel_rank
+        dp_rank = vllm_config.parallel_config.data_parallel_index
         local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", self.prefix)
         os.makedirs(local_cache_dir, exist_ok=True)
         self.compilation_config.local_cache_dir = local_cache_dir
diff --git a/vllm/compilation/decorators.py b/vllm/compilation/decorators.py
index 40bde97ac61d..5334a57040eb 100644
--- a/vllm/compilation/decorators.py
+++ b/vllm/compilation/decorators.py
@@ -403,7 +403,7 @@ def __call__(self, *args, **kwargs):
             )
 
             rank = self.vllm_config.parallel_config.rank
-            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
+            dp_rank = self.vllm_config.parallel_config.data_parallel_index
             cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}")
             aot_compilation_path = os.path.join(cache_dir, "model")
             try:
diff --git a/vllm/config/model.py b/vllm/config/model.py
index a730aa8ad1b9..b061c8cfe70d 100644
--- a/vllm/config/model.py
+++ b/vllm/config/model.py
@@ -642,7 +642,7 @@ def _get_transformers_backend_cls(self) -> str:
         cls = "Transformers"
         # If 'hf_config != hf_text_config' it's a nested config, i.e. multimodal
         cls += "MultiModal" if self.hf_config != self.hf_text_config else ""
-        cls += "MoE" if self.get_num_experts() > 1 else ""
+        cls += "MoE" if self.is_moe else ""
         # Check if the architecture we're wrapping has defaults
         runner = None
         task = None
@@ -1001,8 +1001,7 @@ def _verify_bnb_config(self) -> None:
             self.enforce_eager = True
 
     def _verify_with_expert_parallelism(self) -> None:
-        num_experts = self.get_num_experts()
-        if num_experts < 1:
+        if not self.is_moe:
             raise ValueError(
                 "Number of experts in the model must be greater than 0 "
                 "when expert parallelism is enabled."
@@ -1797,11 +1796,11 @@ def is_prefix_caching_supported(self) -> bool:
                 logger.debug("Generative models support prefix caching.")
                 return True
 
-    def is_model_moe(
-        self,
-    ) -> bool:
-        return self.get_num_experts() > 1
+    @property
+    def is_moe(self) -> bool:
+        return self.get_num_experts() > 0
 
+    @property
     def is_quantized(self) -> bool:
         return getattr(self.hf_config, "quantization_config", None) is not None
 
diff --git a/vllm/config/parallel.py b/vllm/config/parallel.py
index 11504fb08355..9273ca66e830 100644
--- a/vllm/config/parallel.py
+++ b/vllm/config/parallel.py
@@ -119,6 +119,8 @@ class ParallelConfig:
     between local data parallel ranks, but an external LB balances
     between vLLM nodes/replicas. Set explicitly in conjunction with
     --data-parallel-start-rank."""
+    is_moe_model: bool | None = None
+    """Whether the deployed model is MoE (if known)."""
     enable_expert_parallel: bool = False
     """Use expert parallelism instead of tensor parallelism for MoE layers."""
     enable_eplb: bool = False
@@ -255,6 +257,10 @@ class is dynamically inherited by the worker class. This is used to inject
     Block_size should be divisible by cp_kv_cache_interleave_size.
     """
 
+    data_parallel_index: int = Field(init=False)
+    """Equal to the data parallel rank but not used for torch process groups
+    and not overridden for dense models."""
+
     _api_process_count: int = Field(default=1, gt=0)
     """
     The number of API processes initialized.
@@ -466,6 +472,7 @@ def compute_hash(self):
             "data_parallel_rank",
             "data_parallel_rank_local",
             "data_parallel_size_local",
+            "data_parallel_index",
             "data_parallel_backend",
             "data_parallel_external_lb",
             "data_parallel_hybrid_lb",
@@ -546,6 +553,14 @@ def __post_init__(self) -> None:
             self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
             self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT
 
+            if self.data_parallel_size > 1 and self.is_moe_model is False:
+                raise ValueError(
+                    "Offline data parallel mode is not supported/useful"
+                    " for dense models."
+                )
+
+        self.data_parallel_index = self.data_parallel_rank
+
         if self.distributed_executor_backend == "external_launcher":
             os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
             logger.info("Disabling V1 multiprocessing for external launcher.")
diff --git a/vllm/config/vllm.py b/vllm/config/vllm.py
index 70319f98fabb..0598ebf7cd3d 100644
--- a/vllm/config/vllm.py
+++ b/vllm/config/vllm.py
@@ -343,6 +343,29 @@ def pad_for_cudagraph(self, batch_size: int) -> int:
         # i.e., batch_size <= self.compilation_config.max_cudagraph_capture_size
         return self.compilation_config.bs_to_padded_graph_size[batch_size]
 
+    @property
+    def needs_dp_coordinator(self) -> bool:
+        """
+        Determine if the DPCoordinator process is needed.
+
+        The DPCoordinator is needed in two cases:
+        1. For MoE models with DP > 1: to handle wave coordination
+           (even in external LB mode, since wave coordination runs in the coordinator)
+        2. For non-MoE models in internal/hybrid LB mode: to collect and publish
+           queue stats for load balancing across DP ranks
+
+        Returns:
+            True if DPCoordinator process is needed, False otherwise.
+        """
+
+        # For non-MoE models, only need coordinator in internal/hybrid LB mode
+        # (for stats collection).
+        return self.parallel_config.data_parallel_size > 1 and (
+            self.model_config is None
+            or self.model_config.is_moe
+            or not self.parallel_config.data_parallel_external_lb
+        )
+
     def enable_trace_function_call_for_thread(self) -> None:
         """
         Set up function tracing for the current thread,
@@ -522,6 +545,8 @@ def __post_init__(self):
             self.model_config.verify_with_parallel_config(self.parallel_config)
             self.model_config.verify_dual_chunk_attention_config(self.load_config)
 
+            self.parallel_config.is_moe_model = self.model_config.is_moe
+
         self.cache_config.verify_with_parallel_config(self.parallel_config)
 
         if self.lora_config is not None:
@@ -811,9 +836,14 @@ def has_blocked_weights():
             )
 
         # Do this after all the updates to compilation_config.mode
+        effective_dp_size = (
+            self.parallel_config.data_parallel_size
+            if self.model_config is None or self.model_config.is_moe
+            else 1
+        )
         self.compilation_config.set_splitting_ops_for_v1(
             all2all_backend=self.parallel_config.all2all_backend,
-            data_parallel_size=self.parallel_config.data_parallel_size,
+            data_parallel_size=effective_dp_size,
         )
 
         if self.compilation_config.pass_config.enable_sp:
@@ -1281,13 +1311,8 @@ def compile_debug_dump_path(self) -> Path | None:
         if self.compilation_config.debug_dump_path is None:
             return None
         tp_rank = self.parallel_config.rank
-        dp_rank = self.parallel_config.data_parallel_rank
-        data_parallel_size = self.parallel_config.data_parallel_size
-        append_path = (
-            f"rank_{tp_rank}"
-            if data_parallel_size == 1
-            else f"rank_{tp_rank}_dp_{dp_rank}"
-        )
+        dp_rank = self.parallel_config.data_parallel_index
+        append_path = f"rank_{tp_rank}_dp_{dp_rank}"
         path = self.compilation_config.debug_dump_path / append_path
         return path
 
diff --git a/vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py b/vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py
index 9a15d3fa6ed0..af7792286f6a 100644
--- a/vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py
+++ b/vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py
@@ -915,6 +915,6 @@ def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
     # This logic is now centralized
     return (
         envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
-        + vllm_config.parallel_config.data_parallel_rank
+        + vllm_config.parallel_config.data_parallel_index
         * vllm_config.parallel_config.tensor_parallel_size
     )
diff --git a/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py b/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
index 757ca41e9844..5160b4ef3163 100644
--- a/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
+++ b/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
@@ -464,7 +464,7 @@ def __init__(self, vllm_config: VllmConfig, engine_id: str):
         self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
         self.side_channel_port = (
             envs.VLLM_NIXL_SIDE_CHANNEL_PORT
-            + vllm_config.parallel_config.data_parallel_rank
+            + vllm_config.parallel_config.data_parallel_index
         )
         assert vllm_config.kv_transfer_config is not None
         if current_platform.device_type == "cpu":
diff --git a/vllm/distributed/parallel_state.py b/vllm/distributed/parallel_state.py
index f5ada5a009ec..4611d42a5874 100644
--- a/vllm/distributed/parallel_state.py
+++ b/vllm/distributed/parallel_state.py
@@ -1115,7 +1115,11 @@ def get_dp_group() -> GroupCoordinator:
 
 
 def get_ep_group() -> GroupCoordinator:
-    assert _EP is not None, "expert parallel group is not initialized"
+    assert _EP is not None, (
+        "expert parallel group is not initialized. "
+        "EP group is only created for MoE models with num_experts > 0. "
+        "This function should only be called for MoE models."
+    )
     return _EP
 
 
@@ -1400,20 +1404,23 @@ def initialize_model_parallel(
 
     global _EP
     assert _EP is None, "expert parallel group is already initialized"
-    group_ranks = (
-        all_ranks.transpose(1, 2)
-        .reshape(
-            -1,
-            data_parallel_size
-            * prefill_context_model_parallel_size
-            * tensor_model_parallel_size,
+    # Don't create EP group for dense models.
+    if config is None or config.model_config is None or config.model_config.is_moe:
+        group_ranks = (
+            all_ranks.transpose(1, 2)
+            .reshape(
+                -1,
+                data_parallel_size
+                * prefill_context_model_parallel_size
+                * tensor_model_parallel_size,
+            )
+            .unbind(0)
         )
-        .unbind(0)
-    )
-    group_ranks = [x.tolist() for x in group_ranks]
-    _EP = init_model_parallel_group(
-        group_ranks, get_world_group().local_rank, backend, group_name="ep"
-    )
+        group_ranks = [x.tolist() for x in group_ranks]
+        _EP = init_model_parallel_group(
+            group_ranks, get_world_group().local_rank, backend, group_name="ep"
+        )
+    # If no EP group needed, _EP remains None
 
     logger.info_once(
         "rank %s in world size %s is assigned as "
@@ -1425,7 +1432,7 @@ def initialize_model_parallel(
         _PP.rank_in_group,
         _PCP.rank_in_group,
         _TP.rank_in_group,
-        _EP.rank_in_group,
+        _EP.rank_in_group if _EP is not None else "N/A",
     )
 
 
diff --git a/vllm/engine/arg_utils.py b/vllm/engine/arg_utils.py
index dd12b032c78c..2627a3b6531f 100644
--- a/vllm/engine/arg_utils.py
+++ b/vllm/engine/arg_utils.py
@@ -1574,6 +1574,7 @@ def create_engine_config(
             data_parallel_rpc_port=data_parallel_rpc_port,
             data_parallel_backend=self.data_parallel_backend,
             data_parallel_hybrid_lb=self.data_parallel_hybrid_lb,
+            is_moe_model=model_config.is_moe,
             enable_expert_parallel=self.enable_expert_parallel,
             all2all_backend=self.all2all_backend,
             enable_dbo=self.enable_dbo,
diff --git a/vllm/forward_context.py b/vllm/forward_context.py
index 7a569ec32eac..1856d7f70c68 100644
--- a/vllm/forward_context.py
+++ b/vllm/forward_context.py
@@ -102,6 +102,7 @@ def make(
     ) -> "DPMetadata":
         assert num_tokens_across_dp_cpu is not None
         assert parallel_config.data_parallel_size > 1
+        assert parallel_config.is_moe_model is not False
         dp_rank = parallel_config.data_parallel_rank
         batchsize = num_tokens
 
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index 3e6176289746..0111fd6e7198 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -127,7 +127,7 @@ def __init__(
 
         self.kv_event_publisher = EventPublisherFactory.create(
             self.kv_events_config,
-            self.parallel_config.data_parallel_rank,
+            self.parallel_config.data_parallel_index,
         )
         self.ec_connector = None
         if self.vllm_config.ec_transfer_config is not None:
diff --git a/vllm/v1/engine/coordinator.py b/vllm/v1/engine/coordinator.py
index 953342cdd5d0..c2a9fe7c046a 100644
--- a/vllm/v1/engine/coordinator.py
+++ b/vllm/v1/engine/coordinator.py
@@ -55,7 +55,9 @@ class DPCoordinator:
     request wave / running state changes.
     """
 
-    def __init__(self, parallel_config: ParallelConfig):
+    def __init__(
+        self, parallel_config: ParallelConfig, enable_wave_coordination: bool = True
+    ):
         dp_size = parallel_config.data_parallel_size
         assert dp_size > 1, "Coordinator only used for data parallel"
 
@@ -83,6 +85,7 @@ def __init__(self, parallel_config: ParallelConfig):
                 "front_publish_address": front_publish_address,
                 "back_output_address": back_output_address,
                 "back_publish_address": back_publish_address,
+                "enable_wave_coordination": enable_wave_coordination,
             },
             daemon=True,
         )
@@ -110,13 +113,19 @@ def __init__(self):
 
 
 class DPCoordinatorProc:
-    def __init__(self, engine_count: int, min_stats_update_interval_ms: int = 100):
+    def __init__(
+        self,
+        engine_count: int,
+        min_stats_update_interval_ms: int = 100,
+        enable_wave_coordination: bool = True,
+    ):
         set_process_title("DPCoordinator")
         self.ctx = zmq.Context()
 
         self.engines = [EngineState() for _ in range(engine_count)]
 
         self.stats_update_interval_ms = min_stats_update_interval_ms
+        self.enable_wave_coordination = enable_wave_coordination
 
     @staticmethod
     def run_coordinator(
@@ -125,10 +134,12 @@ def run_coordinator(
         back_output_address: str,
         back_publish_address: str,
         min_stats_update_interval_ms: int = 100,
+        enable_wave_coordination: bool = True,
     ):
         coordinator = DPCoordinatorProc(
             engine_count=engine_count,
             min_stats_update_interval_ms=min_stats_update_interval_ms,
+            enable_wave_coordination=enable_wave_coordination,
         )
         try:
             coordinator.process_input_socket(
@@ -265,22 +276,25 @@ def process_input_socket(
                             )
                         continue  # Skip normal engine notification processing
 
-                    # We received a message on the front-end XPUB socket,
-                    # from an API server sending a new request while the
-                    # engines are paused, so that we can wake the other
-                    # engines.
-                    engine_to_exclude, wave = decoded
-                    if not engines_running:
-                        if wave < current_wave:
-                            # If the wave number is stale, ensure the message
-                            # is handled by all the engines.
-                            engine_to_exclude = None
-
-                        engines_running = True
-                        wave_state_changed = True
-                        self._send_start_wave(
-                            publish_back, current_wave, engine_to_exclude
-                        )
+                    # Wave coordination: handle new-request messages from front-end.
+                    # Only process these when wave coordination is enabled
+                    if self.enable_wave_coordination:
+                        # We received a message on the front-end XPUB socket,
+                        # from an API server sending a new request while the
+                        # engines are paused, so that we can wake the other
+                        # engines.
+                        engine_to_exclude, wave = decoded
+                        if not engines_running:
+                            if wave < current_wave:
+                                # If the wave number is stale, ensure the message
+                                # is handled by all the engines.
+                                engine_to_exclude = None
+
+                            engines_running = True
+                            wave_state_changed = True
+                            self._send_start_wave(
+                                publish_back, current_wave, engine_to_exclude
+                            )
 
                 if output_back in events:
                     # We received a message from one of the engines.
@@ -325,34 +339,39 @@ def process_input_socket(
                         stats[1] = scheduler_stats.num_running_reqs
                         stats_changed = True
 
-                    if (wave := outputs.wave_complete) is not None:
-                        # 2. Notification from rank 0 engine that we've
-                        # moved into the global paused state
-                        # (engines_running==False).
-                        if current_wave <= wave:
-                            new_wave = wave + 1
+                    # Wave coordination: handle wave completion and start notifications
+                    # Only process these when wave coordination is enabled
+                    if self.enable_wave_coordination:
+                        if (wave := outputs.wave_complete) is not None:
+                            # 2. Notification from rank 0 engine that we've
+                            # moved into the global paused state
+                            # (engines_running==False).
+                            if current_wave <= wave:
+                                new_wave = wave + 1
+                                logger.debug(
+                                    "Moving DP wave from %d to %d.",
+                                    current_wave,
+                                    new_wave,
+                                )
+                                current_wave = new_wave
+                                engines_running = False
+                                wave_state_changed = True
+                        elif (wave := outputs.start_wave) is not None and (
+                            wave > current_wave
+                            or (wave == current_wave and not engines_running)
+                        ):
+                            # 3. The engine received request for a non-current wave
+                            # so we must ensure that other engines progress to the
+                            # next wave (race condition handling).
                             logger.debug(
-                                "Moving DP wave from %d to %d.", current_wave, new_wave
+                                "Starting wave %d after notification of "
+                                "stale wave request from engine.",
+                                wave,
                             )
-                            current_wave = new_wave
-                            engines_running = False
+                            current_wave = wave
+                            engines_running = True
                             wave_state_changed = True
-                    elif (wave := outputs.start_wave) is not None and (
-                        wave > current_wave
-                        or (wave == current_wave and not engines_running)
-                    ):
-                        # 3. The engine received request for a non-current wave
-                        # so we must ensure that other engines progress to the
-                        # next wave (race condition handling).
-                        logger.debug(
-                            "Starting wave %d after notification of "
-                            "stale wave request from engine.",
-                            wave,
-                        )
-                        current_wave = wave
-                        engines_running = True
-                        wave_state_changed = True
-                        self._send_start_wave(publish_back, wave, eng_index)
+                            self._send_start_wave(publish_back, wave, eng_index)
 
                 if wave_state_changed:
                     message = (None, current_wave, engines_running)
diff --git a/vllm/v1/engine/core.py b/vllm/v1/engine/core.py
index 5f8883c164b3..b5df52466031 100644
--- a/vllm/v1/engine/core.py
+++ b/vllm/v1/engine/core.py
@@ -84,6 +84,7 @@ def __init__(
         executor_class: type[Executor],
         log_stats: bool,
         executor_fail_callback: Callable | None = None,
+        include_finished_set: bool = False,
     ):
         # plugins need to be loaded at the engine/scheduler level too
         from vllm.plugins import load_general_plugins
@@ -91,7 +92,7 @@ def __init__(
         load_general_plugins()
 
         self.vllm_config = vllm_config
-        if vllm_config.parallel_config.data_parallel_rank == 0:
+        if not vllm_config.parallel_config.data_parallel_rank_local:
             logger.info(
                 "Initializing a V1 LLM engine (v%s) with config: %s",
                 VLLM_VERSION,
@@ -138,7 +139,7 @@ def __init__(
             vllm_config=vllm_config,
             kv_cache_config=kv_cache_config,
             structured_output_manager=self.structured_output_manager,
-            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
+            include_finished_set=include_finished_set,
             log_stats=self.log_stats,
             block_size=scheduler_block_size,
         )
@@ -607,6 +608,7 @@ def __init__(
         executor_class: type[Executor],
         log_stats: bool,
         client_handshake_address: str | None = None,
+        *,
         engine_index: int = 0,
     ):
         self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
@@ -638,17 +640,22 @@ def __init__(
                 self.has_coordinator,
                 self.frontend_stats_publish_address,
             )
-            # Only publish request queue stats to coordinator for "internal"
-            # and "hybrid" LB modes .
-            self.publish_dp_lb_stats = (
+            internal_dp_balancing = (
                 self.has_coordinator
                 and not vllm_config.parallel_config.data_parallel_external_lb
             )
+            # Only publish request queue stats to coordinator for "internal"
+            # and "hybrid" LB modes.
+            self.publish_dp_lb_stats = internal_dp_balancing
 
             self._init_data_parallel(vllm_config)
 
             super().__init__(
-                vllm_config, executor_class, log_stats, executor_fail_callback
+                vllm_config,
+                executor_class,
+                log_stats,
+                executor_fail_callback,
+                internal_dp_balancing,
             )
 
             # Background Threads and Queues for IO. These enable us to
@@ -856,18 +863,29 @@ def signal_handler(signum, frame):
 
         engine_core: EngineCoreProc | None = None
         try:
-            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
-            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
+            vllm_config: VllmConfig = kwargs["vllm_config"]
+            parallel_config: ParallelConfig = vllm_config.parallel_config
+            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
+            if data_parallel:
+                parallel_config.data_parallel_rank_local = local_dp_rank
                 set_process_title("EngineCore", f"DP{dp_rank}")
-                decorate_logs()
+            else:
+                set_process_title("EngineCore")
+            decorate_logs()
+
+            parallel_config.data_parallel_index = dp_rank
+            if data_parallel and vllm_config.model_config.is_moe:
                 # Set data parallel rank for this engine process.
                 parallel_config.data_parallel_rank = dp_rank
-                parallel_config.data_parallel_rank_local = local_dp_rank
                 engine_core = DPEngineCoreProc(*args, **kwargs)
             else:
-                set_process_title("EngineCore")
-                decorate_logs()
-                engine_core = EngineCoreProc(*args, **kwargs)
+                # Non-MoE DP ranks are completely independent, so treat like DP=1.
+                # Note that parallel_config.data_parallel_index will still reflect
+                # the original DP rank.
+                parallel_config.data_parallel_size = 1
+                parallel_config.data_parallel_size_local = 1
+                parallel_config.data_parallel_rank = 0
+                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
 
             engine_core.run_busy_loop()
 
@@ -1197,6 +1215,10 @@ def __init__(
         log_stats: bool,
         client_handshake_address: str | None = None,
     ):
+        assert vllm_config.model_config.is_moe, (
+            "DPEngineCoreProc should only be used for MoE models"
+        )
+
         # Counts forward-passes of the model so that we can synchronize
         # finished with DP peers every N steps.
         self.step_counter = 0
@@ -1212,7 +1234,7 @@ def __init__(
             executor_class,
             log_stats,
             client_handshake_address,
-            dp_rank,
+            engine_index=dp_rank,
         )
 
     def _init_data_parallel(self, vllm_config: VllmConfig):
@@ -1393,7 +1415,7 @@ def reinitialize_distributed(
             )
 
 
-class DPEngineCoreActor(DPEngineCoreProc):
+class EngineCoreActorMixin:
     """
     Ray actor for running EngineCore in a data parallel context
     """
@@ -1401,15 +1423,12 @@ class DPEngineCoreActor(DPEngineCoreProc):
     def __init__(
         self,
         vllm_config: VllmConfig,
-        local_client: bool,
         addresses: EngineZmqAddresses,
-        executor_class: type[Executor],
-        log_stats: bool,
         dp_rank: int = 0,
         local_dp_rank: int = 0,
     ):
         self.addresses = addresses
-        vllm_config.parallel_config.data_parallel_rank = dp_rank
+        vllm_config.parallel_config.data_parallel_index = dp_rank
         vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank
 
         # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
@@ -1431,8 +1450,6 @@ def __init__(
         # of ray.
         self._set_visible_devices(vllm_config, local_dp_rank)
 
-        super().__init__(vllm_config, local_client, "", executor_class, log_stats)
-
     def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
         from vllm.platforms import current_platform
 
@@ -1493,7 +1510,7 @@ def run(self):
         Run the engine core busy loop.
         """
         try:
-            self.run_busy_loop()
+            self.run_busy_loop()  # type: ignore[attr-defined]
         except SystemExit:
             logger.debug("EngineCore exiting.")
             raise
@@ -1501,4 +1518,58 @@ def run(self):
             logger.exception("EngineCore encountered a fatal error.")
             raise
         finally:
-            self.shutdown()
+            self.shutdown()  # type: ignore[attr-defined]
+
+
+class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
+    """Used for MoE model data parallel cases."""
+
+    def __init__(
+        self,
+        vllm_config: VllmConfig,
+        local_client: bool,
+        addresses: EngineZmqAddresses,
+        executor_class: type[Executor],
+        log_stats: bool,
+        dp_rank: int = 0,
+        local_dp_rank: int = 0,
+    ):
+        vllm_config.parallel_config.data_parallel_rank = dp_rank
+
+        EngineCoreActorMixin.__init__(
+            self, vllm_config, addresses, dp_rank, local_dp_rank
+        )
+        DPEngineCoreProc.__init__(
+            self, vllm_config, local_client, "", executor_class, log_stats
+        )
+
+
+class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
+    """Used for non-MoE and/or non-DP cases."""
+
+    def __init__(
+        self,
+        vllm_config: VllmConfig,
+        local_client: bool,
+        addresses: EngineZmqAddresses,
+        executor_class: type[Executor],
+        log_stats: bool,
+        dp_rank: int = 0,
+        local_dp_rank: int = 0,
+    ):
+        vllm_config.parallel_config.data_parallel_size = 1
+        vllm_config.parallel_config.data_parallel_size_local = 1
+        vllm_config.parallel_config.data_parallel_rank = 0
+
+        EngineCoreActorMixin.__init__(
+            self, vllm_config, addresses, dp_rank, local_dp_rank
+        )
+        EngineCoreProc.__init__(
+            self,
+            vllm_config,
+            local_client,
+            "",
+            executor_class,
+            log_stats,
+            engine_index=dp_rank,
+        )
diff --git a/vllm/v1/engine/core_client.py b/vllm/v1/engine/core_client.py
index cacbc805e84f..f74e90abc906 100644
--- a/vllm/v1/engine/core_client.py
+++ b/vllm/v1/engine/core_client.py
@@ -502,7 +502,7 @@ def __init__(
 
             parallel_config = vllm_config.parallel_config
             dp_size = parallel_config.data_parallel_size
-            dp_rank = parallel_config.data_parallel_rank
+            dp_rank = parallel_config.data_parallel_index
             dp_local_size = parallel_config.data_parallel_size_local
             offline_mode = parallel_config.data_parallel_rank_local is not None
             # Client manages local+remote EngineCores in pure internal LB case.
diff --git a/vllm/v1/engine/llm_engine.py b/vllm/v1/engine/llm_engine.py
index 7c3f9a0e868b..c02143c7295e 100644
--- a/vllm/v1/engine/llm_engine.py
+++ b/vllm/v1/engine/llm_engine.py
@@ -65,8 +65,9 @@ def __init__(
 
         self.log_stats = log_stats
 
-        executor_backend = self.vllm_config.parallel_config.distributed_executor_backend
         parallel_config = vllm_config.parallel_config
+        executor_backend = parallel_config.distributed_executor_backend
+
         self.external_launcher_dp = (
             parallel_config.data_parallel_size > 1
             and executor_backend == "external_launcher"
diff --git a/vllm/v1/engine/utils.py b/vllm/v1/engine/utils.py
index 24bf66c42f31..66212ed7cd5e 100644
--- a/vllm/v1/engine/utils.py
+++ b/vllm/v1/engine/utils.py
@@ -75,7 +75,6 @@ class EngineHandshakeMetadata:
 
     addresses: EngineZmqAddresses
     parallel_config: dict[str, int | str | list[int]]
-    parallel_config_hash: str | None = None
 
 
 class CoreEngineProcManager:
@@ -249,12 +248,19 @@ def __init__(
         from ray.runtime_env import RuntimeEnv
         from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
 
-        from vllm.v1.engine.core import DPEngineCoreActor
+        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor
+
+        dp_size = vllm_config.parallel_config.data_parallel_size
+        actor_class = (
+            DPMoEEngineCoreActor
+            if dp_size > 1 and vllm_config.model_config.is_moe
+            else EngineCoreActor
+        )
 
         self.local_engine_actors: list[ray.ActorHandle] = []
         self.remote_engine_actors: list[ray.ActorHandle] = []
 
-        env_vars_list = get_env_vars_to_copy(destination="DPEngineCoreActor")
+        env_vars_list = get_env_vars_to_copy(destination=actor_class.__name__)
         self.env_vars_dict = {
             name: os.environ[name] for name in env_vars_list if name in os.environ
         }
@@ -263,7 +269,6 @@ def __init__(
         self.addresses = addresses
         self.executor_class = executor_class
         self.log_stats = log_stats
-        dp_size = vllm_config.parallel_config.data_parallel_size
         local_engine_count = vllm_config.parallel_config.data_parallel_size_local
         world_size = vllm_config.parallel_config.world_size
 
@@ -314,7 +319,7 @@ def __init__(
                 runtime_env = RuntimeEnv(env_vars=actor_env_vars)
 
             actor = (
-                ray.remote(DPEngineCoreActor)
+                ray.remote(actor_class)
                 .options(
                     scheduling_strategy=PlacementGroupSchedulingStrategy(
                         placement_group=pg,
@@ -624,7 +629,13 @@ def scale_up_elastic_ep(
         from ray.runtime_env import RuntimeEnv
         from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
 
-        from vllm.v1.engine.core import DPEngineCoreActor
+        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor
+
+        actor_class = (
+            DPMoEEngineCoreActor
+            if cur_vllm_config.model_config.is_moe
+            else EngineCoreActor
+        )
 
         cur_data_parallel_size = len(self.local_engine_actors) + len(
             self.remote_engine_actors
@@ -667,7 +678,7 @@ def scale_up_elastic_ep(
                 )
 
             actor = (
-                ray.remote(DPEngineCoreActor)
+                ray.remote(actor_class)
                 .options(
                     scheduling_strategy=PlacementGroupSchedulingStrategy(
                         placement_group=pg,
@@ -804,12 +815,19 @@ def launch_core_engines(
         ],
     )
 
-    # Run the DP Coordinator process with rank 0 when in
-    # online DP mode.
-    run_coordinator = dp_size > 1 and not offline_mode and dp_rank == 0
+    # Run the DP Coordinator process with rank 0 when in online DP mode.
+    # The coordinator is needed for:
+    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
+    # 2. MoE models: wave coordination in addition to stats
+    run_coordinator = (
+        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
+    )
 
     if run_coordinator:
-        coordinator = DPCoordinator(parallel_config)
+        coordinator = DPCoordinator(
+            parallel_config,
+            enable_wave_coordination=vllm_config.model_config.is_moe,
+        )
 
         addresses.coordinator_input, addresses.coordinator_output = (
             coordinator.get_engine_socket_addresses()
@@ -905,6 +923,7 @@ def launch_core_engines(
             addresses,
             engines_to_handshake,
             parallel_config,
+            dp_size > 1 and vllm_config.model_config.is_moe,
             vllm_config.cache_config,
             local_engine_manager,
             coordinator.proc if coordinator else None,
@@ -916,6 +935,7 @@ def wait_for_engine_startup(
     addresses: EngineZmqAddresses,
     core_engines: list[CoreEngine],
     parallel_config: ParallelConfig,
+    coordinated_dp: bool,
     cache_config: CacheConfig,
     proc_manager: CoreEngineProcManager | None,
     coord_process: Process | None,
@@ -997,8 +1017,7 @@ def wait_for_engine_startup(
                 )
 
         if status == "HELLO" and engine.state == CoreEngineState.NEW:
-            # Send init message with DP config info and config hash.
-            # The config hash ensures all DP workers have compatible configs.
+            # Send init message with DP config info.
             init_message = msgspec.msgpack.encode(
                 EngineHandshakeMetadata(
                     addresses=addresses,
@@ -1010,10 +1029,9 @@ def wait_for_engine_startup(
                             "_data_parallel_master_port_list",
                             "data_parallel_size",
                         )
-                    },
-                    parallel_config_hash=parallel_config.compute_hash()
-                    if parallel_config.data_parallel_size > 1
-                    else None,
+                    }
+                    if coordinated_dp
+                    else {},
                 )
             )
             handshake_socket.send_multipart((eng_identity, init_message), copy=False)
@@ -1034,8 +1052,8 @@ def wait_for_engine_startup(
             if addresses.frontend_stats_publish_address is None:
                 addresses.frontend_stats_publish_address = msg.get("dp_stats_address")
 
-            # Validate config hash consistency across DP workers
-            if parallel_config.data_parallel_size > 1:
+            # Validate config hash consistency across DP workers for MoE models.
+            if coordinated_dp:
                 worker_config_hash = msg.get("parallel_config_hash")
                 expected_hash = parallel_config.compute_hash()
                 if worker_config_hash != expected_hash:
diff --git a/vllm/v1/worker/gpu/model_runner.py b/vllm/v1/worker/gpu/model_runner.py
index 9f4c6edfb6aa..636d2e381732 100644
--- a/vllm/v1/worker/gpu/model_runner.py
+++ b/vllm/v1/worker/gpu/model_runner.py
@@ -98,9 +98,6 @@ def __init__(
         self.max_num_reqs = self.scheduler_config.max_num_seqs
         self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
 
-        self.dp_size = self.parallel_config.data_parallel_size
-        self.dp_rank = self.parallel_config.data_parallel_rank
-
         self.use_async_scheduling = self.scheduler_config.async_scheduling
         self.output_copy_stream = torch.cuda.Stream(self.device)
         self.output_copy_event = torch.cuda.Event()
@@ -268,7 +265,8 @@ def _dummy_run(
         if not skip_attn:
             self.prepare_dummy_attn_metadata(input_batch)
 
-        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)
+        dp_size = self.parallel_config.data_parallel_size
+        num_tokens_across_dp = make_num_tokens_across_dp(dp_size, num_tokens)
         num_sampled_tokens = np.ones(input_batch.num_reqs, dtype=np.int32)
         with (
             self.maybe_dummy_run_with_lora(
@@ -312,7 +310,7 @@ def profile_run(self) -> None:
         self._dummy_sampler_run(sample_hidden_states)
         if self.do_spec_decode:
             num_tokens_across_dp = make_num_tokens_across_dp(
-                self.dp_size, self.max_num_tokens
+                self.parallel_config.data_parallel_size, self.max_num_tokens
             )
             self.speculator.run_model(
                 self.max_num_tokens,
@@ -807,7 +805,8 @@ def get_cudagraph_and_dp_padding(
         scheduler_output: SchedulerOutput,
     ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
         total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
-        if self.dp_size == 1:
+        dp_size = self.parallel_config.data_parallel_size
+        if dp_size == 1:
             # No DP. Only consider CUDA graphs.
             if total_num_scheduled_tokens == 0:
                 # Special case: no tokens to run.
@@ -835,11 +834,12 @@ def get_cudagraph_and_dp_padding(
                 cudagraph_size_before_dp = -1
 
         assert cudagraph_size_before_dp is not None
+        dp_rank = self.parallel_config.data_parallel_rank
         num_tokens_across_dp, cudagraph_size_across_dp = get_batch_metadata_across_dp(
             total_num_scheduled_tokens,
             cudagraph_size_before_dp,
-            self.dp_size,
-            self.dp_rank,
+            dp_size,
+            dp_rank,
         )
         if all(cudagraph_size_across_dp >= 0):
             # If all ranks can use CUDA graph, pad to the maximum number of tokens
@@ -850,7 +850,7 @@ def get_cudagraph_and_dp_padding(
             # If any of the ranks cannot use CUDA graph, use eager mode for all ranks.
             # No padding is needed except for ranks that have no tokens to run.
             num_tokens_across_dp = torch.clamp(num_tokens_across_dp, min=1)
-            num_tokens_after_padding = num_tokens_across_dp[self.dp_rank]
+            num_tokens_after_padding = num_tokens_across_dp[dp_rank]
             cudagraph_mode = CUDAGraphMode.NONE
         return cudagraph_mode, num_tokens_after_padding, num_tokens_across_dp
 
diff --git a/vllm/v1/worker/gpu_worker.py b/vllm/v1/worker/gpu_worker.py
index fae7fa620a20..c84244f78e31 100644
--- a/vllm/v1/worker/gpu_worker.py
+++ b/vllm/v1/worker/gpu_worker.py
@@ -179,22 +179,20 @@ def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
         self.cache_config.num_cpu_blocks = num_cpu_blocks
 
     def init_device(self):
-        device = self.device_config.device
-        if isinstance(device, torch.device) and device.type == "cuda":
+        if self.device_config.device_type == "cuda":
             # This env var set by Ray causes exceptions with graph building.
             os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
+            parallel_config = self.parallel_config
             if (
-                self.parallel_config.data_parallel_size > 1
-                and self.parallel_config.data_parallel_size_local > 0
-                and self.parallel_config.distributed_executor_backend
-                not in ["ray", "external_launcher"]
-                and self.vllm_config.parallel_config.data_parallel_backend != "ray"
-                and self.vllm_config.parallel_config.nnodes_within_dp == 1
+                parallel_config.distributed_executor_backend
+                not in ("ray", "external_launcher")
+                and parallel_config.data_parallel_backend != "ray"
+                and parallel_config.nnodes_within_dp == 1
             ):
                 # Use local DP rank if available, otherwise use global DP rank.
                 dp_local_rank = self.parallel_config.data_parallel_rank_local
                 if dp_local_rank is None:
-                    dp_local_rank = self.parallel_config.data_parallel_rank
+                    dp_local_rank = self.parallel_config.data_parallel_index
 
                 tp_pp_world_size = (
                     self.parallel_config.pipeline_parallel_size

```
