# PR #41269: [Bugfix][KV Transfer][NIXL] Notify P node on pre-admission rejection to free stranded KV blocks (#41269)

# Issue

## Original PR description (captured source)

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

## Full discussion

See [41269.md](./41269.md) and [41269.json](./41269.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/41269
- Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/41269
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `aea9c0001d7887c79b256ae6f9e5237b16f4a01c865f291734e05f850593c4ed`

Existing semantic notes in [41269.sem.json](./41269.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/v1/kv_connector/unit/test_nixl_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py`
- `vllm/engine/protocol.py`
- `vllm/entrypoints/openai/chat_completion/serving.py`
- `vllm/entrypoints/openai/completion/serving.py`
- `vllm/entrypoints/openai/engine/serving.py`
- `vllm/entrypoints/openai/responses/serving.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/request.py`

### Diff

```diff
diff --git a/tests/v1/kv_connector/unit/test_nixl_connector.py b/tests/v1/kv_connector/unit/test_nixl_connector.py
index 3803e4fd3869..1a8612cd3fc4 100644
--- a/tests/v1/kv_connector/unit/test_nixl_connector.py
+++ b/tests/v1/kv_connector/unit/test_nixl_connector.py
@@ -321,6 +321,34 @@ def test_prompt_less_than_block_size():
     assert len(scheduler_output.scheduled_new_reqs) == 0
 
 
+def test_abort_immediately_remote_prefill_enqueues_empty_recv():
+    """A remote-prefill request added with abort_immediately=True should
+    be added to the scheduler's waiting queue then immediately aborted, so the
+    NIXL connector's request_finished hook enqueues an empty recv to notify
+    the prefill instance to free its blocks."""
+    from vllm.v1.request import RequestStatus
+
+    scheduler = create_scheduler(create_vllm_config())
+
+    request = create_request(request_id=42, num_tokens=10, do_remote_prefill=True)
+    assert request.kv_transfer_params is not None
+    assert request.kv_transfer_params["do_remote_prefill"] is True
+
+    # Mimic the EngineCore.add_request path for an abort-immediately req.
+    scheduler.add_request(request)
+    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
+
+    scheduler_output = scheduler.schedule()
+    meta = scheduler_output.kv_connector_metadata
+    assert isinstance(meta, NixlConnectorMetadata)
+    assert set(meta.reqs_to_recv) == {request.request_id}
+    req_meta = meta.reqs_to_recv[request.request_id]
+    assert req_meta.local_block_ids == []
+    assert req_meta.remote.request_id == f"prefill-{42}"
+    # do_remote_prefill is consumed by request_finished to prevent re-issuing.
+    assert request.kv_transfer_params["do_remote_prefill"] is False
+
+
 @patch(
     "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker.NixlWrapper",
     FakeNixlWrapper,
diff --git a/vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py b/vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
index 02c418ebd8d7..a053f2c32611 100644
--- a/vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
+++ b/vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py
@@ -525,7 +525,9 @@ def request_finished(
         if params.get("do_remote_prefill"):
             # If do_remote_prefill is still True when the request is finished,
             # update_state_after_alloc must not have been called (the request
-            # must have been aborted before it was scheduled).
+            # must have been aborted before it was scheduled, e.g. via the
+            # abort_immediately path used to clean up KV-transfer requests
+            # rejected at the D-side serving layer).
             # To avoid stranding the prefill blocks in the prefill instance,
             # we must add empty block_ids to _reqs_need_recv so that our
             # worker side will notify and free blocks in the prefill instance.
diff --git a/vllm/engine/protocol.py b/vllm/engine/protocol.py
index 1471dd1c566d..3f83734a5b78 100644
--- a/vllm/engine/protocol.py
+++ b/vllm/engine/protocol.py
@@ -108,6 +108,20 @@ async def abort(self, request_id: str | Iterable[str]) -> None:
         """
         ...
 
+    @abstractmethod
+    async def notify_kv_transfer_request_rejected(
+        self,
+        request_id: str,
+        kv_transfer_params: dict[str, Any],
+        *,
+        data_parallel_rank: int | None = None,
+    ) -> None:
+        """Notify the engine that a KV-transfer request was rejected before
+        engine admission, so connector-side cleanup can run (e.g. free
+        prefill blocks pinned on the P node).
+        """
+        ...
+
     @abstractmethod
     async def is_tracing_enabled(self) -> bool: ...
 
diff --git a/vllm/entrypoints/openai/chat_completion/serving.py b/vllm/entrypoints/openai/chat_completion/serving.py
index 1026e0a1e3f7..375fe9794c1e 100644
--- a/vllm/entrypoints/openai/chat_completion/serving.py
+++ b/vllm/entrypoints/openai/chat_completion/serving.py
@@ -234,6 +234,15 @@ async def create_chat_completion(
         for the API specification. This API mimics the OpenAI
         Chat Completion API.
         """
+        return await self._with_kv_transfer_rejection_cleanup(
+            self._create_chat_completion(request, raw_request), request, raw_request
+        )
+
+    async def _create_chat_completion(
+        self,
+        request: ChatCompletionRequest,
+        raw_request: Request | None = None,
+    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
         # Streaming response
         tokenizer = self.renderer.tokenizer
         assert tokenizer is not None
diff --git a/vllm/entrypoints/openai/completion/serving.py b/vllm/entrypoints/openai/completion/serving.py
index ee4ca9f3ada3..05efe86466d4 100644
--- a/vllm/entrypoints/openai/completion/serving.py
+++ b/vllm/entrypoints/openai/completion/serving.py
@@ -118,6 +118,15 @@ async def create_completion(
             - suffix (the language models we currently support do not support
             suffix)
         """
+        return await self._with_kv_transfer_rejection_cleanup(
+            self._create_completion(request, raw_request), request, raw_request
+        )
+
+    async def _create_completion(
+        self,
+        request: CompletionRequest,
+        raw_request: Request | None = None,
+    ) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse:
         if request.stream and request.use_beam_search:
             return self.create_error_response(
                 "Streaming is not currently supported with beam search"
diff --git a/vllm/entrypoints/openai/engine/serving.py b/vllm/entrypoints/openai/engine/serving.py
index f0f84a82204c..2a51cc0bfac0 100644
--- a/vllm/entrypoints/openai/engine/serving.py
+++ b/vllm/entrypoints/openai/engine/serving.py
@@ -4,7 +4,7 @@
 import contextlib
 import json
 import time
-from collections.abc import AsyncGenerator, Mapping
+from collections.abc import AsyncGenerator, Awaitable, Mapping
 from dataclasses import dataclass, field
 from http import HTTPStatus
 from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar
@@ -118,6 +118,7 @@ def build_chat_params(
 )
 
 RequestT = TypeVar("RequestT", bound=AnyRequest)
+_T = TypeVar("_T")
 
 
 @dataclass(kw_only=True)
@@ -156,6 +157,9 @@ def __init__(
         self.model_config = engine_client.model_config
         self.renderer = engine_client.renderer
         self.input_processor = engine_client.input_processor
+        vllm_config = getattr(engine_client, "vllm_config", None)
+        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
+        self.has_kv_connector = kv_transfer_config is not None
 
         # Computed once at startup (cached by ``vllm_config`` identity) and
         # stamped on non-streaming responses. Streaming chunks deliberately
@@ -616,6 +620,40 @@ def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
         except ValueError:
             return None
 
+    async def _with_kv_transfer_rejection_cleanup(
+        self,
+        awaitable: Awaitable[_T],
+        request: ChatCompletionRequest | CompletionRequest | ResponsesRequest,
+        raw_request: Request | None,
+    ) -> _T:
+        """Wrap a `create_*` coroutine so that, if it raises or returns an
+        ErrorResponse (i.e. the request never reached the engine), the KV
+        connector is notified to free any pinned remote-prefill blocks."""
+        kv_transfer_params = self.has_kv_connector and request.kv_transfer_params
+        if not kv_transfer_params or not kv_transfer_params.get("do_remote_prefill"):
+            return await awaitable
+
+        notify = True
+        try:
+            result = await awaitable
+            if not isinstance(result, ErrorResponse):
+                notify = False
+            return result
+        finally:
+            if notify:
+                try:
+                    await self.engine_client.notify_kv_transfer_request_rejected(
+                        request.request_id,
+                        kv_transfer_params,
+                        data_parallel_rank=self._get_data_parallel_rank(raw_request),
+                    )
+                except Exception:
+                    logger.warning(
+                        "Failed to notify KV connector about rejected request %s",
+                        request.request_id,
+                        exc_info=True,
+                    )
+
     @staticmethod
     def _parse_tool_calls_from_content(
         request: ResponsesRequest | ChatCompletionRequest,
diff --git a/vllm/entrypoints/openai/responses/serving.py b/vllm/entrypoints/openai/responses/serving.py
index 9c4dc48589ff..fb28a2256ad0 100644
--- a/vllm/entrypoints/openai/responses/serving.py
+++ b/vllm/entrypoints/openai/responses/serving.py
@@ -323,6 +323,17 @@ async def create_responses(
         AsyncGenerator[StreamingResponsesResponse, None]
         | ResponsesResponse
         | ErrorResponse
+    ):
+        return await self._with_kv_transfer_rejection_cleanup(
+            self._create_responses(request, raw_request), request, raw_request
+        )
+
+    async def _create_responses(
+        self, request: ResponsesRequest, raw_request: Request | None = None
+    ) -> (
+        AsyncGenerator[StreamingResponsesResponse, None]
+        | ResponsesResponse
+        | ErrorResponse
     ):
         error_check_ret = await self._check_model(request)
         if error_check_ret is not None:
diff --git a/vllm/v1/engine/__init__.py b/vllm/v1/engine/__init__.py
index 8172ead08319..f6c80055b651 100644
--- a/vllm/v1/engine/__init__.py
+++ b/vllm/v1/engine/__init__.py
@@ -122,6 +122,12 @@ class EngineCoreRequest(
     reasoning_ended: bool | None = None
     reasoning_parser_kwargs: dict[str, Any] | None = None
 
+    # If True, the request should be added to the scheduler's waiting queue
+    # and immediately aborted, so connector-side cleanup runs via the standard
+    # request_finished hook. Used to free P-side prefill blocks when a
+    # KV-transfer request is rejected on the D node before engine admission.
+    abort_immediately: bool = False
+
     @property
     def params(self) -> SamplingParams | PoolingParams:
         """Return the processed params (sampling or pooling)."""
diff --git a/vllm/v1/engine/async_llm.py b/vllm/v1/engine/async_llm.py
index 02ed2917c7d6..32cc3c0d2d05 100644
--- a/vllm/v1/engine/async_llm.py
+++ b/vllm/v1/engine/async_llm.py
@@ -720,6 +720,33 @@ async def abort(
         if self.log_requests:
             logger.info("Aborted request(s) %s.", ",".join(request_ids))
 
+    async def notify_kv_transfer_request_rejected(
+        self,
+        request_id: str,
+        kv_transfer_params: dict[str, Any],
+        *,
+        data_parallel_rank: int | None = None,
+    ) -> None:
+        """Submit a pre-aborted request so the connector's request_finished
+        hook runs to free any pre-admission KV-transfer resources (e.g. NIXL
+        prefill blocks pinned on the P node)."""
+        request = EngineCoreRequest(
+            request_id=request_id,
+            prompt_token_ids=[0],
+            mm_features=None,
+            sampling_params=SamplingParams(
+                max_tokens=1,
+                extra_args={"kv_transfer_params": dict(kv_transfer_params)},
+            ),
+            pooling_params=None,
+            arrival_time=time.time(),
+            lora_request=None,
+            cache_salt=None,
+            data_parallel_rank=data_parallel_rank,
+            abort_immediately=True,
+        )
+        await self.engine_core.add_request_async(request)
+
     async def pause_generation(
         self,
         *,
diff --git a/vllm/v1/engine/core.py b/vllm/v1/engine/core.py
index 11c5ee19a664..122cdd8f965d 100644
--- a/vllm/v1/engine/core.py
+++ b/vllm/v1/engine/core.py
@@ -344,6 +344,10 @@ def add_request(self, request: Request, request_wave: int = 0):
             )
 
         self.scheduler.add_request(request)
+        if request.abort_immediately:
+            # Immediately abort so the connector's request_finished hook runs
+            # to free any pre-admission KV-transfer resources.
+            self.abort_requests([request.request_id])
 
     def abort_requests(self, request_ids: list[str]):
         """Abort requests from the scheduler."""
diff --git a/vllm/v1/request.py b/vllm/v1/request.py
index 738a68c83680..0d435deb3f05 100644
--- a/vllm/v1/request.py
+++ b/vllm/v1/request.py
@@ -76,6 +76,7 @@ def __init__(
         resumable: bool = False,
         reasoning_ended: bool | None = None,
         reasoning_parser_kwargs: dict[str, Any] | None = None,
+        abort_immediately: bool = False,
     ) -> None:
         self.request_id = request_id
         self.client_index = client_index
@@ -182,6 +183,10 @@ def __init__(
         # None entry in the queue means finished.
         self.streaming_queue: deque[StreamingUpdate | None] | None = None
 
+        # If True, request should be aborted immediately after being added to
+        # the scheduler so the connector's request_finished hook runs.
+        self.abort_immediately = abort_immediately
+
     @classmethod
     def from_engine_core_request(
         cls,
@@ -206,6 +211,7 @@ def from_engine_core_request(
             resumable=request.resumable,
             reasoning_ended=request.reasoning_ended,
             reasoning_parser_kwargs=request.reasoning_parser_kwargs,
+            abort_immediately=request.abort_immediately,
         )
 
     def append_output_token_ids(
```
