# Task PR #29499 — [EPLB] Optimize EPLB with numpy (#29499)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
A system that spreads MoE experts across GPUs periodically rebalances them by moving expert weights between devices to even out load. The rebalancing code was slow: it copied data between GPU and CPU more than needed, ran its bookkeeping on the GPU in slow torch/Python rather than vectorized code, and physically moved some experts even when they stayed on the same GPU, wasting memory copies. It also logged balance statistics every step, adding cross-GPU communication overhead.

### Issue — detailed
The EPLB rearrange step in eplb_state.py and its helper primitives in rebalance_execute.py (get_ep_ranks_with_expert, move_to_buffer, move_from_buffer) ran on torch tensors on the GPU. move_from_buffer did multiple GPU-CPU transfers per call. The intermediate state (is_unchanged, is_received_locally, experts_recv_loc) was passed as Python lists and a dict. rearrange moved physical_to_logical_map onto the device and called new_indices[layer_idx].tolist() each layer, and timing used torch.cuda.synchronize() plus time.perf_counter(), forcing host-device sync. The rebalance result could reassign an expert to a different physical slot on the same GPU even when no cross-GPU move was needed, forcing an unnecessary GPU memcopy. Balancedness logging ran every step and each log triggered a collective (_sync_load_pass). EPLBConfig post-processing lacked validation for incompatible settings and had a bug around default values.

## PR's solution (reference — what was merged)
**General:** Reimplement the hot rebalancing primitives in numpy on the CPU so they avoid repeated GPU-CPU copies and run faster. Add a post-processing step that keeps experts which stay on the same GPU in their existing slots, skipping needless weight copies. Add a config knob to log balance statistics less often, cutting the collectives it triggers, plus config validation.

**Detailed:** In rebalance_execute.py, move_to_buffer, move_from_buffer, and get_ep_ranks_with_expert were rewritten in numpy and move_from_buffer's multiple GPU-CPU transfers were removed; its signature now takes a RecvMetadata object (recv_primary_mask, recv_count, recv_expert_ids, recv_dst_rows) plus a CPU new_indices tensor and ep_rank, instead of the old experts_recv_loc dict, ep_group, and a .tolist() call. The RecvMetadata dataclass is threaded through async_worker.py and eplb_state.py, and EplbModelState fields became np.ndarray/RecvMetadata instead of lists/dict. A new DefaultEplbPolicy.preserve_intragpu_slots(new_phy2log, phy_replicas_idx, num_ranks, old_phy2log) keeps experts remaining on a GPU in their old slots (tested in test_eplb_algo.py). new_physical_to_logical_map is moved to CPU once in advance rather than per-transfer, and timing in rearrange switched from time.perf_counter() to torch.cuda.Event. EPLBConfig in vllm/config/parallel.py gained log_balancedness_interval (default 1) gating the logging-and-sync block in step(), plus a _validate_eplb_config model_validator that rejects async with a non-default policy and non-positive intervals.

**Author's words:** Removes multiple gpu-cpu transfers in move_from_buffer. Implements get_ep_ranks_with_expert, move_to_buffer, move_from_buffer in numpy which gives 20-25% on average for each primitive. Adds preserve_intragpu_slots primitive that post-processes rebalance algo results, ensuring experts assigned to be moved within the same gpu stay in their slots, avoiding unnecessary gpu memcopies. Adds log_balancedness_interval config parameter to tune the frequency of balancedness logs and possibly reduce the number of collectives required for this logging. Improves performance of sync and async eplb. Fixes bug in EPLB config post processing.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **move_from_buffer** (us): no author number. workload: move_from_buffer
- **get_ep_ranks_with_expert** (us): no author number. workload: get_ep_ranks
- **move_to_buffer** (us): no author number. workload: move_to_buffer

## Reported numbers (PR author)
- metric: EPLB rearrange time for the profiled third EPLB call (sync mode, Qwen3-Next-80B-A3B)
- value: 1.370 -> 1.077 (~27%)
- unit: seconds

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
