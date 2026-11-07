# PR #41269: [Bugfix][KV Transfer][NIXL] Notify P node on pre-admission rejection to free stranded KV blocks

- field: scheduling  | area: entrypoints/openai
- author: Dao007forever
- state: MERGED  created: 2026-04-29T18:46:55Z  merged: 2026-05-10T05:52:09Z
- url: https://github.com/vllm-project/vllm/pull/41269
- labels: bug, frontend, ready, v1, kv-connector

## Body

## Summary

Fixes a bug where prefill KV blocks on the **P (prefill) node** are stranded for up to `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` (default 480s) when the **D (decode) node** rejects an incoming request *before* it is admitted to the engine scheduler.

When a request that carries KV-transfer params (`do_remote_prefill=True`, `remote_block_ids`, `remote_engine_id`, etc.) is rejected on D for reasons like:

- Render / chat-template error
- Model existence check failure (`_check_model`)
- Input validation error
- Engine errored
- Beam-search-with-stream rejection
- `previous_response_id` not found (responses API)

…D never opens a NIXL transfer for it, so P never receives the implicit "transfer complete → free blocks" signal. The blocks linger until the abort timeout fires.

This PR adds an explicit early-rejection notification:

1. The OpenAI-compatible serving layer (`chat_completion`, `completion`, `responses`) calls `engine_client.notify_kv_transfer_request_rejected(...)` on every pre-admission rejection path that has `kv_transfer_params.do_remote_prefill=True`.
2. `AsyncLLM` → `EngineCoreClient` (in-process / sync MP / async MP / DPLB-async MP) → `EngineCore` → `Scheduler` → `KVConnectorBase_V1.request_rejected_before_admission(...)`.
3. `NixlConnectorScheduler` recognizes the params and enqueues an empty `_reqs_need_recv` entry. On the next scheduler tick the worker side issues the notification that releases the remote blocks. To make sure that tick actually happens when D has no other in-flight work, `Scheduler.has_requests()` now also reports pending connector metadata.
4. `MultiConnector` fans out to its child connectors and accepts the first one that recognizes the params.
5. `DPLBAsyncMPClient` broadcasts the notification to all local DP engines when no `data_parallel_rank` header is present (the rejection happens before admission, so the request isn't yet tracked in `reqs_in_flight`).

The `_reqs_need_recv` value type changed from `(Request, BlockIds)` to `(dict[str, Any], BlockIds)` because pre-admission rejections do not have a `Request` object — only the raw `kv_transfer_params` dict — and that's all the existing code on the `_build_kv_connector_meta` side actually consumed (`req.kv_transfer_params`).

If `do_remote_prefill` is set but the required `remote_*` metadata is incomplete, the connector logs a warning and returns False (no-op) rather than guessing.

## Why this is not duplicating an existing PR

I checked open PRs and issues with searches like `stranded prefill`, `stranded KV blocks rejected NIXL`, `request_rejected_before_admission`, `kv_transfer_params reject`, and `decode reject prefill`. The closest matches are:

- **#38027 [Nixl][PD] Lease renewal TTL KV blocks on P** — bounds stranding via heartbeat-based TTL refresh. Complementary, not a duplicate: it shortens the window for crashes / disconnects, but pre-admission rejections still need an explicit "this transfer will never start" signal so blocks free *immediately* instead of after the (shortened) lease.
- **#35764 [Feat][NIXL] Add KV lease refresh mechanism** — same area, same rationale as #38027.
- **#41237 [Bugfix][KV Transfer] Reject NixlConnector + expandable_segments:True** — unrelated config-validation fix.

No existing PR covers the rejected-before-admission notification path.

## Test plan

- [x] New unit tests added:
  - `tests/v1/kv_connector/unit/test_nixl_connector.py::test_rejected_remote_prefill_request_enqueues_empty_recv` — verifies the connector enqueues an empty recv with the original `remote_*` params and `Scheduler.has_requests()` flips True until the tick flushes it.
  - `tests/v1/kv_connector/unit/test_nixl_connector.py::test_rejected_remote_prefill_request_missing_metadata_is_ignored` — verifies the connector no-ops (and does not mutate `do_remote_prefill`) when required `remote_*` fields are missing.
  - `tests/v1/kv_connector/unit/test_multi_connector.py::test_request_rejected_before_admission_uses_first_accepting_connector` — verifies short-circuit fan-out behavior in `MultiConnector`.
- [x] Local: `.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_multi_connector.py tests/v1/kv_connector/unit/test_nixl_connector.py -v` — 76 passed, 2 skipped, 1 unrelated failure (`test_multi_example_connector_consistency`, which fails on `OSError: gated repo meta-llama/Llama-3.2-1B-Instruct` — unrelated to this change).
- [x] Pre-commit hooks staged / will run on push.
- [ ] Reviewer should confirm behavior end-to-end on a P/D NIXL deployment by sending an oversized prompt (or one whose chat template fails) carrying `kv_transfer_params.do_remote_prefill=True` and observing that the P-side block usage drops immediately rather than after `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`.

## AI assistance

This change was prepared with assistance from Claude (Anthropic). I (the human submitter) reviewed every changed line, ran the tests above, and can defend the design end-to-end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Comments

### liuzijing2014 @ 2026-05-05T19:16:58Z

Q: what's the benefit vs setting `VLLM_NIXL_ABORT_REQUEST_TIMEOUT=60s` and let prefill naturally timeout? 60s feel fine for edge case handling (there are many cases where decode could fail and never fetch from prefill e.g. engine level kv allocation failure).
---
### Dao007forever @ 2026-05-05T19:30:21Z

60s is a very large in our tests, that reduced the throughput significantly as the concurrency was only ~4-5 and 1 hanging request affect ~25% concurrency.
---
### robertgshaw2-redhat @ 2026-05-08T15:33:53Z

> Thanks for optimizing this part of the workflow @Dao007forever , it isn't trivial. To be honest, I don't expect a lot of validation errors above to happen on a healthy deployment (given they should be failing on P first assuming an exact copy of the code is deployed).
> 
> The implementation looks clean as is to me, but I wonder whether we shouldn't be gating this feature behind the presence of a kv connector in the first place. Afaiu, this is currently only checking the params that are present in the request, not the config of the server itself.
> 
> cc @markmc might also be interested in this

Slack discussion for future reference
 
```
Nick Hill  [11:23 AM]
https://github.com/vllm-project/vllm/pull/41269 - only failures are the known ones
21 replies

Robert Shaw  [11:28 AM]
Do you think this is necessary with the TTL stuff nicolo has been working on landing?
[11:28 AM]cc [@Nick Hill](https://vllm-dev.slack.com/team/U07V7KVR47J)
[11:28 AM]cc [@Nicolò Lucchesi](https://vllm-dev.slack.com/team/U07R95Y4QQH)

Nicolò Lucchesi  [11:29 AM]
I reviewed this morning, I think it's pretty niche

Nick Hill  [11:30 AM]
oops sorry I missed your comment

Nicolò Lucchesi  [11:31 AM]
it's an extra fast-path I am just not sure how likely this is in practice. But it's true is complimentary

Nick Hill  [11:31 AM]
the main case we saw for this is min_length / max_length validation against max_model_len
[11:31 AM]we always override min/max to 1 for the prefill request

Robert Shaw  [11:31 AM]
yeah, that seems reasonable
[11:31 AM]ok sounds good

Nick Hill  [11:31 AM]
so that succeeds, then we send to decode with the user's min and/or max
[11:31 AM]and it fails

Robert Shaw  [11:32 AM]
makes sense

Nick Hill  [11:32 AM]
but currently, this leaves the P stranded.. so it's very easy to trigger/DoS kind of thing

Robert Shaw  [11:32 AM]
yeah it makes sense

Nicolò Lucchesi  [11:32 AM]
I think I patched the router for that thign when I was testing multiturn bench

Robert Shaw  [11:32 AM]
but we dont control all routers
[11:32 AM]im going to merge it

Nick Hill  [11:33 AM]
yeah that is a good point. Ideally we have the validation in the router.. but yes exactly then you have to rely on the router, and also for things like max_model_len, it's hard for the router to know.. it would have to query the config etc

Nicolò Lucchesi  [11:33 AM]
I would still gate it behind a connector check though lol

Robert Shaw  [11:33 AM]
should I merge it?
---
### njhill @ 2026-05-08T16:10:45Z

Thanks @NickLucche @robertgshaw2-redhat! I added a check that a connector is configured.
---