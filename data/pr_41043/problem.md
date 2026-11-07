# Task PR #41043 — [Perf][Spec Decode] Avoid per-step numpy allocation in prepare_next_t… (#41043)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> In the speculative decoding path, `prepare_next_token_ids_padded` rebuilds the backup next-token IDs on every decode step using `.tolist()`, a Python list comprehension, and `np.array()`. These three temporaries are redundant because a pre-allocated pinned `CpuGpuBuffer` (`backup_next_token_ids`) already exists to hold the values. The extra per-step allocation wastes CPU work and creates garbage that triggers occasional Python GC pauses.

### Issue — detailed

> `prepare_next_token_ids_padded` in `vllm/v1/spec_decode/llm_base_proposer.py` built `seq_lens_list = (num_tokens_no_spec[:num_reqs]-1).tolist()`, then a list comprehension over `requests[...].get_token_id(...)`, then wrapped it in `np.array(dtype=np.int32)` and assigned into the buffer. Three temporary objects are created and immediately copied/discarded each step, when the result could be written straight into the existing pinned numpy buffer. Because the method lives on `SpecDecodeBaseProposer`, the waste applies to all spec-decode methods.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff deletes the `.tolist()` call, the list comprehension, and the `np.array()` wrapper. It replaces them with a simple `for i in range(num_reqs)` loop that writes each `requests[gpu_input_batch.req_ids[i]].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)` directly into `self.backup_next_token_ids.np[i]`, the pre-allocated pinned buffer. No temporary Python lists or numpy array are created per step. Since spec-decode batches are small (1–16 requests) the loop is trivially cheap, and eliminating the allocation removes the per-step GC pauses that inflated P99 TPOT.

**Author's words:**

> The author writes directly into the pre-allocated pinned `CpuGpuBuffer` via a small Python loop over the requests, computing each backup token ID with `get_token_id(num_tokens_no_spec[i] - 1)` and storing it into `backup_next_token_ids.np[i]`. This removes the intermediate `.tolist()`, list comprehension, and `np.array()` allocation. They note the loop is trivially fast for typical spec-decode batch sizes (1–16) and the change applies to all spec-decode methods via `SpecDecodeBaseProposer`.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **prepare_next_token_ids_b16** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **prepare_next_token_ids_b256** (us): no number reported in the PR.
- **mean_tpot** (ms): reported 2.6 -> 2.58 ms.
- **p99_tpot** (ms): reported 3.24 -> 2.94 ms.
- note: SpecDecodeBaseProposer.prepare_next_token_ids_padded (unbound, real CpuGpuBuffer, stand-in InputBatch/requests) for decode batches of 16 and 256 with 3 spec tokens; PR drops the per-step tolist/list-comp/np.array temporaries. Wall-clock us per call (host prep + Triton launch). Author numbers are e2e TPOT with Eagle3 (reported-only).

## Reported numbers (PR author)
- gain: 0.8%, 9.3% (pct)
- evidence: | | Mean TPOT (ms) | P99 TPOT (ms) | |---|---|---| | Before (3 runs avg) | 2.60 | 3.24 | | After (3 runs avg) | 2.58 | 2.94 | | **Improvement** | **0.8%** | **9
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
