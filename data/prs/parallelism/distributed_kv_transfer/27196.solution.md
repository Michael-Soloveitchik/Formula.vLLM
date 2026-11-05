# PR #27196: [Bugfix][P/D] Reduce num_threads used by nixl ucx backend (#27196)

# Issue

## Original PR description (captured source)

This PR adds the ability to configure the number of UCX threads used for the NIXL backend agents. 

Example usage:
`--kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"num_threads": 2}}'`

## Purpose
This PR fixes an issue with WideEP in RDMA environments where we ran out of UAR space while initializing nvshmem for multi-node DeepEP, when used in conjunction with P/D. 

This issue would result in logs like
```
/tmp/nvshmem_src/src/modules/transport/ibgda/ibgda.cpp 2939 Creating 1 DCI QPs (shared: 1, exclusive: 0)
...
/tmp/nvshmem_src/src/modules/transport/ibgda/ibgda.cpp 2967 Creating 8 RC QPs
/tmp/nvshmem_src/src/modules/transport/ibgda/ibgda.cpp 2967 Creating 8 RC QPs
...
/tmp/nvshmem_src/src/modules/transport/ibgda/ibgda.cpp:1830: NULL value cannot allocate mlx5dv_devx_uar

/tmp/nvshmem_src/src/modules/transport/ibgda/ibgda.cpp:1951: non-zero status: 12 ibgda_alloc_and_map_qp_uar failed
...
/tmp/nvshmem_src/src/modules/transport/ibgda/ibgda.cpp:2978: non-zero status: 7 ibgda_create_dci failed on RC #157.
/tmp/nvshmem_src/src/host/transport/transport.cpp:420: non-zero status: 7 connect EPS failed
```

This issue is not seen prior to https://github.com/vllm-project/vllm/commit/8ce5d3198d00631a76e1aa02a57947b46bc7218c where num_threads=8 was introduced. Some investigation with bpftrace revealed that in vLLM versions since this change, there are more UAR allocations by NIXL (from ~550 to ~1640), which explains the resource shortage later on in the start-up process when nvshmem is being bootstrapped.

## Test Plan
Tested `VLLM_ALL2ALL_BACKEND=deepep_low_latency` with this change on a 3x8xH100 node environment on IBM Cloud. 

## Test Result
The above error is resolved and the model (Qwen-235B-A22B) loaded successfully.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [X] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [X] The test plan, such as providing test command.
- [X] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Full discussion

See [27196.md](./27196.md) and [27196.json](./27196.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/27196
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/27196.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `240a1ca6de6ba293c84eadc709af2247fb0431f2bc94412ee91baa4fbee2aa2d`

Existing semantic notes in [27196.sem.json](./27196.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`

### Diff

```diff
diff --git a/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py b/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
index ede5db646f3d..d73e05562951 100644
--- a/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
+++ b/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
@@ -607,15 +607,25 @@ def __init__(self, vllm_config: VllmConfig, engine_id: str):
         # TODO temporary, once nixl allows for telemetry flag in config
         # (next release), we can remove this env var.
         os.environ["NIXL_TELEMETRY_ENABLE"] = "1"
+
         # Agent.
         non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
+        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
+        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
+        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
+        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
+        # initialization with "mlx5dv_devx_alloc_uar" errors.
+        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
+        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
+            "num_threads", 4
+        )
         if nixl_agent_config is None:
             config = None
         else:
             config = (
                 nixl_agent_config(backends=self.nixl_backends)
                 if len(non_ucx_backends) > 0
-                else nixl_agent_config(num_threads=8)
+                else nixl_agent_config(num_threads=num_threads)
             )
 
         self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), config)
```
