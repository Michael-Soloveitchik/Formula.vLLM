#!/usr/bin/env python3
"""Microbench for vLLM PR #33593: [Perf] Optimize Python Slice for Structured Output using islice.

Two real call sites change from a copying list slice to a lazy islice:
  * StructuredOutputManager.should_advance: the per-step "did reasoning end in this step" check
    passed `all_token_ids[delta_from:]` (a copy) to is_reasoning_end_streaming; now an islice.
  * BaseThinkingReasoningParser.extract_content_ids: `end_token_id not in input_ids[:-1]`
    (copy, then scan) -> membership over `islice(input_ids, 0, len-1)` (scan only).

The bench calls the arm's own code for both: should_advance (unbound, with a minimal self holding
the reasoner and a stand-in request whose delta covers the whole sequence) and
extract_content_ids on a DeepSeekR1ReasoningParser built over a stub tokenizer vocabulary.
Sequence length N = 1M tokens with no end token, as in the author's microbench
(SIZE 2M / N 1M, best of R=5). Metric names follow that microbench:
  create_slice - should_advance per call (slice/islice creation + membership scan)
  sum_slice    - extract_content_ids per call (slice/islice + membership scan)
Seconds per call, best of R. CPU only. Runs inside the task image: python3 /task/bench.py
"""
import time
from types import SimpleNamespace

from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.v1.structured_output import StructuredOutputManager

N = 1_000_000
R = 5
THINK_START, THINK_END = 1, 2


class StubTokenizer:
    def get_vocab(self):
        return {"<think>": THINK_START, "</think>": THINK_END, "<unk>": 0}


def best_of(fn):
    best = float("inf")
    for _ in range(R):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    parser = DeepSeekR1ReasoningParser(StubTokenizer())
    assert parser.end_token_id == THINK_END
    tokens = [1000 + (i % 5000) for i in range(N)]          # no </think> anywhere

    manager = SimpleNamespace(reasoner=parser, enable_in_reasoning=False)
    request = SimpleNamespace(
        use_structured_output=True,
        structured_output_request=SimpleNamespace(reasoning_ended=False),
        num_computed_tokens=0,          # delta_from = 0 -> the delta is the whole sequence
        num_output_placeholders=0,
        all_token_ids=tokens,
    )

    def advance():
        return StructuredOutputManager.should_advance(manager, request)

    def extract():
        return parser.extract_content_ids(tokens)

    assert advance() is False and request.structured_output_request.reasoning_ended is False
    assert extract() == []
    s_advance = best_of(advance)
    s_extract = best_of(extract)
    print(f"create_slice {s_advance:.6f}")
    print(f"sum_slice {s_extract:.6f}")
    print(f"# N={N} R={R} end_token_present=False (should_advance / extract_content_ids per call)")


if __name__ == "__main__":
    main()
