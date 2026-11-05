# PR #27196: [Bugfix][P/D] Reduce num_threads used by nixl ucx backend

- field: parallelism  | area: distributed/kv_transfer
- author: dagrayvid
- state: MERGED  created: 2025-10-20T13:39:48Z  merged: 2025-10-21T17:41:52Z
- url: https://github.com/vllm-project/vllm/pull/27196
- labels: ready, kv-connector

## Body

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



## Comments

### github-actions @ 2025-10-20T13:39:58Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### markmc @ 2025-10-21T06:27:40Z

This is the failing tests:

```
FAILED v1/distributed/test_async_llm_dp.py::test_load[True-ray-RequestOutputKind.DELTA] - ValueError: Currently, async scheduling only supports `mp` or `uni` distributed executor backend, but you choose `ray`.
[2025-10-20T13:11:22Z] !!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Fixed by #27195 

You can rebase to pick up the fix or a committer can force merge :+1: 
---
### dagrayvid @ 2025-10-21T14:37:45Z

Rebased on main
---