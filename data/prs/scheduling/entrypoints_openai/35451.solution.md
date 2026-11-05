# PR #35451: [Core] Add optional flags to check for repetitive token patterns in engine output (#35451)

# Issue

## Original PR description (captured source)

## Purpose
This PR allows users to end generation early if they see repetitive token patterns (for example, nonsensically repeated sentences) in their output. This is a real failure case for models prone to hallucination. Testing with Qwen3 VL on multimodal input shows a ~33% increase in throughput and near-total reduction in models hitting the configured max_output_len.

This feature is off by default. We include it in the core so that this can be leveraged for non-streaming generation too.

## Test Plan
New test file, online tests

## Test Result
Huge increase in throughput for error-prone multimodal input without sacrificing quality/latency.


## Full discussion

See [35451.md](./35451.md) and [35451.json](./35451.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/35451
- Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/35451
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `650acfcbf9b77032a0d2ed5cebf4360aec320c556f09141330cfb7b70fd6eb03`

Existing semantic notes in [35451.sem.json](./35451.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/v1/core/test_repetition_detection.py`
- `vllm/entrypoints/openai/chat_completion/protocol.py`
- `vllm/entrypoints/openai/completion/protocol.py`
- `vllm/sampling_params.py`
- `vllm/v1/core/sched/utils.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/request.py`

### Diff

```diff
diff --git a/tests/v1/core/test_repetition_detection.py b/tests/v1/core/test_repetition_detection.py
new file mode 100644
index 000000000000..aae6e3b70cae
--- /dev/null
+++ b/tests/v1/core/test_repetition_detection.py
@@ -0,0 +1,290 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+
+from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
+from vllm.v1.core.sched.utils import check_sequence_repetition, check_stop
+from vllm.v1.request import Request, RequestStatus
+
+pytestmark = pytest.mark.cpu_test
+
+# ============================================================================
+# UNIT TESTS - check_sequence_repetition function
+# ============================================================================
+
+
+class TestCheckSequenceRepetition:
+    """Unit tests for the check_sequence_repetition function"""
+
+    def test_simple_repetition_detected(self):
+        """Test detection of simple repetitive patterns"""
+        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 3]
+        params = RepetitionDetectionParams(
+            max_pattern_size=3,
+            min_pattern_size=2,
+            min_count=3,
+        )
+        assert check_sequence_repetition(token_ids, params)
+
+    def test_repetition_below_min_count(self):
+        """Test that pattern below min_count is not detected"""
+        token_ids = [1, 2, 3, 1, 2, 3]
+        params = RepetitionDetectionParams(
+            max_pattern_size=3,
+            min_pattern_size=2,
+            min_count=3,
+        )
+        assert not check_sequence_repetition(token_ids, params)
+
+    def test_two_token_pattern(self):
+        """Test detection of 2-token patterns"""
+        token_ids = [1, 2, 1, 2, 1, 2, 1, 2]
+        params = RepetitionDetectionParams(
+            max_pattern_size=5,
+            min_pattern_size=2,
+            min_count=4,
+        )
+        assert check_sequence_repetition(token_ids, params)
+
+    def test_no_repetition_varied_sequence(self):
+        """Test that non-repetitive sequences are not flagged"""
+        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
+        params = RepetitionDetectionParams(
+            max_pattern_size=5,
+            min_pattern_size=2,
+            min_count=2,
+        )
+        assert not check_sequence_repetition(token_ids, params)
+
+    def test_partial_repetition_not_detected(self):
+        """Test that incomplete repetitions are not detected"""
+        token_ids = [1, 2, 3, 1, 2, 3, 1, 2, 4]
+        params = RepetitionDetectionParams(
+            max_pattern_size=3,
+            min_pattern_size=2,
+            min_count=3,
+        )
+        assert not check_sequence_repetition(token_ids, params)
+
+    def test_empty_token_list(self):
+        """Test with empty token list"""
+        params = RepetitionDetectionParams(
+            max_pattern_size=3,
+            min_pattern_size=2,
+            min_count=2,
+        )
+        assert not check_sequence_repetition([], params)
+
+    def test_detection_disabled_max_size_zero(self):
+        """Test that zero max_pattern_size disables detection"""
+        token_ids = [1, 2, 1, 2, 1, 2]
+        params = RepetitionDetectionParams()
+        assert not check_sequence_repetition(token_ids, params)
+
+    def test_invalid_min_count(self):
+        """Test that min_count < 2 returns False"""
+        token_ids = [1, 2, 1, 2]
+        params = RepetitionDetectionParams()
+        assert not check_sequence_repetition(token_ids, params)
+
+    def test_repetition_at_end_of_sequence(self):
+        """Test detection when repetition occurs at the end"""
+        token_ids = [1, 2, 3, 4, 5, 6, 5, 6, 5, 6]
+        params = RepetitionDetectionParams(
+            max_pattern_size=3,
+            min_pattern_size=2,
+            min_count=3,
+        )
+        assert check_sequence_repetition(token_ids, params)
+
+    def test_large_pattern_many_repetitions(self):
+        """Test large pattern repeated many times"""
+        token_ids = [1, 2, 3, 4, 5, 6, 7, 8] * 5
+        params = RepetitionDetectionParams(
+            max_pattern_size=10,
+            min_pattern_size=2,
+            min_count=3,
+        )
+        assert check_sequence_repetition(token_ids, params)
+
+
+# ============================================================================
+# INTEGRATION TESTS - check_stop with repetition detection
+# ============================================================================
+
+
+class TestRepetitionDetectionIntegration:
+    """Integration tests for repetition detection in check_stop"""
+
+    def test_basic_repetition_stops_generation(self):
+        """Test that repetition is detected and stops generation"""
+        params = SamplingParams(
+            max_tokens=100,
+            repetition_detection=RepetitionDetectionParams(
+                max_pattern_size=5,
+                min_pattern_size=2,
+                min_count=3,
+            ),
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1, 2, 3],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
+        assert check_stop(request, max_model_len=1024)
+        assert request.status == RequestStatus.FINISHED_REPETITION
+        assert request.stop_reason == "repetition_detected"
+
+    def test_detection_disabled_no_stop(self):
+        """Test that disabled detection doesn't stop generation"""
+        params = SamplingParams(
+            max_tokens=100,
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1, 2, 3],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
+        assert not check_stop(request, max_model_len=1024)
+
+    def test_repetition_respects_min_tokens(self):
+        """Test that repetition detection respects min_tokens"""
+        params = SamplingParams(
+            min_tokens=10,
+            max_tokens=100,
+            repetition_detection=RepetitionDetectionParams(
+                max_pattern_size=5,
+                min_pattern_size=2,
+                min_count=3,
+            ),
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1, 2, 3],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
+        assert not check_stop(request, max_model_len=1024)
+
+    def test_no_repetition_continues_generation(self):
+        """Test that non-repetitive tokens don't stop generation"""
+        params = SamplingParams(
+            max_tokens=100,
+            repetition_detection=RepetitionDetectionParams(
+                max_pattern_size=5,
+                min_pattern_size=2,
+                min_count=3,
+            ),
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1, 2, 3],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([10, 20, 30, 40, 50, 60])
+        assert not check_stop(request, max_model_len=1024)
+
+    def test_pattern_at_size_boundary(self):
+        """Test detection at exact pattern size boundary"""
+        params = SamplingParams(
+            max_tokens=100,
+            repetition_detection=RepetitionDetectionParams(
+                max_pattern_size=3,
+                min_pattern_size=3,
+                min_count=2,
+            ),
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1, 2],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([10, 20, 30, 10, 20, 30])
+        assert check_stop(request, max_model_len=1024)
+        assert request.status == RequestStatus.FINISHED_REPETITION
+
+    def test_multiple_pattern_sizes_checked(self):
+        """Test that function checks pattern sizes in range"""
+        params = SamplingParams(
+            max_tokens=100,
+            repetition_detection=RepetitionDetectionParams(
+                max_pattern_size=5,
+                min_pattern_size=2,
+                min_count=3,
+            ),
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10])
+        assert check_stop(request, max_model_len=1024)
+        assert request.status == RequestStatus.FINISHED_REPETITION
+
+    def test_eos_takes_precedence_over_repetition(self):
+        """Test that EOS token stops before repetition check"""
+        params = SamplingParams(
+            max_tokens=100,
+            stop_token_ids=[999],
+            repetition_detection=RepetitionDetectionParams(
+                max_pattern_size=5,
+                min_pattern_size=2,
+                min_count=3,
+            ),
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1, 2, 3],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([10, 20, 10, 20, 999])
+        assert check_stop(request, max_model_len=1024)
+        assert request.status == RequestStatus.FINISHED_STOPPED
+
+    def test_min_pattern_size_filters_small_patterns(self):
+        """Test that min_pattern_size filters out smaller patterns"""
+        params = SamplingParams(
+            max_tokens=100,
+            repetition_detection=RepetitionDetectionParams(
+                max_pattern_size=5,
+                min_pattern_size=3,
+                min_count=3,
+            ),
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
+        assert not check_stop(request, max_model_len=1024)
+
+    def test_high_repetition_threshold(self):
+        """Test that high min_count requires many repetitions"""
+        params = SamplingParams(
+            max_tokens=100,
+            repetition_detection=RepetitionDetectionParams(
+                max_pattern_size=5,
+                min_pattern_size=2,
+                min_count=5,
+            ),
+        )
+        request = Request(
+            request_id="test",
+            prompt_token_ids=[1],
+            sampling_params=params,
+            pooling_params=None,
+        )
+        request.append_output_token_ids([10, 20, 10, 20, 10, 20])
+        assert not check_stop(request, max_model_len=1024)
diff --git a/vllm/entrypoints/openai/chat_completion/protocol.py b/vllm/entrypoints/openai/chat_completion/protocol.py
index 1bf0de53fa3c..69f98b3fa273 100644
--- a/vllm/entrypoints/openai/chat_completion/protocol.py
+++ b/vllm/entrypoints/openai/chat_completion/protocol.py
@@ -38,6 +38,7 @@
 from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
 from vllm.sampling_params import (
     BeamSearchParams,
+    RepetitionDetectionParams,
     RequestOutputKind,
     SamplingParams,
     StructuredOutputsParams,
@@ -336,6 +337,16 @@ class ChatCompletionRequest(OpenAIBaseModel):
         ),
     )
 
+    repetition_detection: RepetitionDetectionParams | None = Field(
+        default=None,
+        description="Parameters for detecting repetitive N-gram patterns "
+        "in output tokens. If such repetition is detected, generation will "
+        "be ended early. LLMs can sometimes generate repetitive, unhelpful "
+        "token patterns, stopping only when they hit the maximum output length "
+        "(e.g. 'abcdabcdabcd...' or '\emoji \emoji \emoji ...'). This feature "
+        "can detect such behavior and terminate early, saving time and tokens.",
+    )
+
     # --8<-- [end:chat-completion-extra-params]
 
     def build_chat_params(
@@ -500,6 +511,7 @@ def to_sampling_params(
             allowed_token_ids=self.allowed_token_ids,
             extra_args=extra_args or None,
             skip_clone=True,  # Created fresh per request, safe to skip clone
+            repetition_detection=self.repetition_detection,
         )
 
     @model_validator(mode="before")
diff --git a/vllm/entrypoints/openai/completion/protocol.py b/vllm/entrypoints/openai/completion/protocol.py
index 226dd6c1a2bb..ce131be0a1df 100644
--- a/vllm/entrypoints/openai/completion/protocol.py
+++ b/vllm/entrypoints/openai/completion/protocol.py
@@ -26,6 +26,7 @@
 from vllm.renderers import TokenizeParams
 from vllm.sampling_params import (
     BeamSearchParams,
+    RepetitionDetectionParams,
     RequestOutputKind,
     SamplingParams,
     StructuredOutputsParams,
@@ -166,6 +167,16 @@ class CompletionRequest(OpenAIBaseModel):
         ),
     )
 
+    repetition_detection: RepetitionDetectionParams | None = Field(
+        default=None,
+        description="Parameters for detecting repetitive N-gram patterns "
+        "in output tokens. If such repetition is detected, generation will "
+        "be ended early. LLMs can sometimes generate repetitive, unhelpful "
+        "token patterns, stopping only when they hit the maximum output length "
+        "(e.g. 'abcdabcdabcd...' or '\emoji \emoji \emoji ...'). This feature "
+        "can detect such behavior and terminate early, saving time and tokens.",
+    )
+
     # --8<-- [end:completion-extra-params]
 
     def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
@@ -311,6 +322,7 @@ def to_sampling_params(
             allowed_token_ids=self.allowed_token_ids,
             extra_args=extra_args or None,
             skip_clone=True,  # Created fresh per request, safe to skip clone
+            repetition_detection=self.repetition_detection,
         )
 
     @model_validator(mode="before")
diff --git a/vllm/sampling_params.py b/vllm/sampling_params.py
index 2f015339e67d..9685d8dd8a74 100644
--- a/vllm/sampling_params.py
+++ b/vllm/sampling_params.py
@@ -107,6 +107,43 @@ def all_non_structural_tag_constraints_none(self) -> bool:
         )
 
 
+@dataclass
+class RepetitionDetectionParams:
+    """Parameters for detecting repetitive N-gram patterns in output tokens."""
+
+    max_pattern_size: int = 0
+    """Maximum size of N-gram pattern to detect for sequence repetition.
+    Set to 0 to disable. Must be used together with min_count."""
+
+    min_pattern_size: int = 0
+    """Minimum N-gram pattern size to check for sequence repetition.
+    If set to 0, it defaults to 1.
+    Must be <= max_pattern_size."""
+
+    min_count: int = 0
+    """Minimum number of times an N-gram pattern must repeat to trigger
+    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
+    3 times. Must be used together with max_pattern_size."""
+
+    def __post_init__(self):
+        if (
+            self.max_pattern_size < 0
+            or self.min_pattern_size < 0
+            or self.min_pattern_size > self.max_pattern_size
+        ):
+            raise ValueError(
+                "max_pattern_size, min_pattern_size must be >=0, "
+                "with min_pattern_size <= max_pattern_size. "
+                "Set both to 0 to disable repetitive pattern detection."
+            )
+        if self.max_pattern_size > 0 and self.min_count < 2:
+            raise ValueError(
+                "min_count must be >= 2 to detect repetitive patterns "
+                "in engine output. If you do not wish to detect repetitive "
+                "patterns, set max_pattern_size to 0."
+            )
+
+
 class RequestOutputKind(Enum):
     # Return entire output so far in every RequestOutput
     CUMULATIVE = 0
@@ -250,6 +287,14 @@ class SamplingParams(
 
     skip_reading_prefix_cache: bool | None = None
 
+    repetition_detection: RepetitionDetectionParams | None = None
+    """Parameters for detecting repetitive N-gram patterns in output tokens.
+    If such repetition is detected, generation will be ended early. LLMs can
+    sometimes generate repetitive, unhelpful token patterns, stopping only
+    when they hit the maximum output length (e.g. 'abcdabcdabcd...' or
+    '\\emoji \\emoji \\emoji ...'). This feature can detect such behavior
+    and terminate early, saving time and tokens."""
+
     @staticmethod
     def from_optional(
         n: int | None = 1,
@@ -280,6 +325,7 @@ def from_optional(
         allowed_token_ids: list[int] | None = None,
         extra_args: dict[str, Any] | None = None,
         skip_clone: bool = False,
+        repetition_detection: RepetitionDetectionParams | None = None,
     ) -> "SamplingParams":
         if logit_bias is not None:
             # Convert token_id to integer
@@ -320,6 +366,7 @@ def from_optional(
             allowed_token_ids=allowed_token_ids,
             extra_args=extra_args,
             skip_clone=skip_clone,
+            repetition_detection=repetition_detection,
         )
 
     def __post_init__(self) -> None:
diff --git a/vllm/v1/core/sched/utils.py b/vllm/v1/core/sched/utils.py
index 22e3aefb69d5..c7cb6b94367e 100644
--- a/vllm/v1/core/sched/utils.py
+++ b/vllm/v1/core/sched/utils.py
@@ -1,10 +1,64 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 import contextlib
+from collections.abc import Sequence
 
+from vllm.sampling_params import RepetitionDetectionParams
 from vllm.v1.request import Request, RequestStatus
 
 
+def _has_repeating_pattern(
+    token_ids: Sequence[int],
+    pattern_len: int,
+    repetition_min_count: int,
+) -> bool:
+    """Check if the tail of token_ids contains a repeating pattern.
+
+    Compares the last pattern_len tokens against the preceding
+    (repetition_min_count - 1) repetitions of the same length.
+    """
+    for n in range(1, pattern_len + 1):
+        target_token = token_ids[-n]
+        for m in range(1, repetition_min_count):
+            if token_ids[-(pattern_len * m + n)] != target_token:
+                return False
+    return True
+
+
+def check_sequence_repetition(
+    token_ids: Sequence[int],
+    params: RepetitionDetectionParams,
+) -> bool:
+    """Check if a sequence of token IDs has a repetition pattern.
+    Args:
+        token_ids: List of token IDs
+        params: Repetition detection parameters.
+    Returns:
+        True if a repetition pattern is found, False otherwise.
+    """
+    max_pattern_size = params.max_pattern_size
+    min_pattern_size = params.min_pattern_size
+    min_count = params.min_count
+
+    if min_pattern_size <= 0:
+        min_pattern_size = 1
+
+    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
+        return False
+
+    for pattern_len in range(
+        min_pattern_size,
+        max_pattern_size + 1,
+    ):
+        if pattern_len * min_count > len(token_ids):
+            return False
+
+        if _has_repeating_pattern(token_ids, pattern_len, min_count):
+            return True
+
+    return False
+
+
 def remove_all(lst: list, items_to_remove: set) -> list:
     """Remove all items from a list that are in the items_to_remove set.
 
@@ -61,4 +115,16 @@ def check_stop(request: Request, max_model_len: int) -> bool:
     ):
         request.status = RequestStatus.FINISHED_LENGTH_CAPPED
         return True
+
+    repetition_detection = sampling_params.repetition_detection
+    if repetition_detection is not None and (
+        check_sequence_repetition(
+            request.output_token_ids,
+            repetition_detection,
+        )
+    ):
+        request.status = RequestStatus.FINISHED_REPETITION
+        request.stop_reason = "repetition_detected"
+        return True
+
     return False
diff --git a/vllm/v1/engine/__init__.py b/vllm/v1/engine/__init__.py
index 1dd9f64f877d..e225a4297a6f 100644
--- a/vllm/v1/engine/__init__.py
+++ b/vllm/v1/engine/__init__.py
@@ -27,12 +27,12 @@
 
 # These are possible values of RequestOutput.finish_reason,
 # so form part of the external API.
-FINISH_REASON_STRINGS = ("stop", "length", "abort", "error")
+FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")
 
 
 class FinishReason(enum.IntEnum):
     """
-    Reason a request finished - stop, length, abort, or error.
+    Reason a request finished - stop, length, abort, error, or repetition.
 
     Int rather than Str for more compact serialization.
 
@@ -41,6 +41,7 @@ class FinishReason(enum.IntEnum):
     abort - aborted by client
     error - retryable request-level internal error (e.g., KV load failure).
             Invariant: always converted to 500 Internal Server Error.
+    repetition - repetitive token pattern detected (hallucination)
 
     """
 
@@ -48,6 +49,7 @@ class FinishReason(enum.IntEnum):
     LENGTH = 1
     ABORT = 2
     ERROR = 3
+    REPETITION = 4
 
     def __str__(self):
         return FINISH_REASON_STRINGS[self.value]
diff --git a/vllm/v1/request.py b/vllm/v1/request.py
index 7d8254e3505d..85ca90d9957c 100644
--- a/vllm/v1/request.py
+++ b/vllm/v1/request.py
@@ -320,6 +320,7 @@ class RequestStatus(enum.IntEnum):
     FINISHED_ABORTED = enum.auto()
     FINISHED_IGNORED = enum.auto()
     FINISHED_ERROR = enum.auto()
+    FINISHED_REPETITION = enum.auto()
 
     def __str__(self) -> str:
         return self.name
@@ -344,4 +345,5 @@ def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
     RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
     RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
     RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
+    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
 }
```
