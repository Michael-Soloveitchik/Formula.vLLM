# PR #32975: [Perf] Optimize detokenizer python logic

- field: scheduling  | area: v1/engine
- author: yewentao256
- state: MERGED  created: 2026-01-23T22:45:06Z  merged: 2026-02-10T07:54:41Z
- url: https://github.com/vllm-project/vllm/pull/32975
- labels: ready, v1

## Body

## Purpose

1. adding a `num_output_tokens` function, to avoid `len(self.output_token_ids)` might do a slice in SlowIncrementalDetokenizer

```py
class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    @property
    def output_token_ids(self) -> list[int]:
        return (
            self.token_ids
            if not self.prompt_len
            else (self.token_ids[self.prompt_len :])
        )
```

~2. accumulate pieces and join once instead of string add~

## Test

Should be covered in unit test
```bash
  tests/tokenizers_/test_detokenize.py \
  tests/detokenizer/test_min_tokens.py \
  tests/detokenizer/test_stop_string_while_stop_model_terminates.py \
  tests/v1/engine/test_fast_incdec_prefix_err.py \
  tests/entrypoints/openai/test_serving_tokens.py
```


CC: @WoosukKwon  @njhill 

## Comments

### yewentao256 @ 2026-02-03T20:41:27Z

@njhill Could you help review/approve this PR? Thanks!
---
### yewentao256 @ 2026-02-05T21:55:24Z

> Thanks @yewentao256, have you done any benchmarks or microbenchmarks to measure the improvement?

Hi @njhill , the e2e perf is smaller than 0.5% so hidden in fluctuations, so I let AI generates a micro bench script

```py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbench for detokenizer `min_tokens` stop_check_offset (old vs new)."""

import argparse
import gc
import time


def _old(num_tokens: int, min_tokens: int, piece: str) -> int:
    base_len = 0
    new_chars_len = 0
    stop_check_offset = base_len
    out_tokens = 0

    for _ in range(num_tokens):
        out_tokens += 1
        new_chars_len += len(piece)
        if min_tokens and out_tokens <= min_tokens:
            stop_check_offset = base_len + new_chars_len

    return stop_check_offset


def _new(num_tokens: int, min_tokens: int, piece: str) -> int:
    stop_check_offset = 0
    out_tokens = 0

    for _ in range(num_tokens):
        out_tokens += 1
        if min_tokens and out_tokens <= min_tokens:
            stop_check_offset += len(piece)

    return stop_check_offset


def _median_ns(fn, *, repeats: int, num_tokens: int, min_tokens: int, piece: str) -> int:
    fn(10_000, min_tokens, piece)  # warmup
    times: list[int] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(num_tokens, min_tokens, piece)
        times.append(time.perf_counter_ns() - t0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbench: detokenizer min_tokens stop_check_offset (old vs new)."
    )
    parser.add_argument("--num-tokens", type=int, default=20_000_000)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument(
        "--piece-len",
        type=int,
        default=1,
        help="Length of the decoded 'piece' (fixed string) used in the benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    piece = "x" * max(0, args.piece_len)
    assert _old(args.num_tokens, args.min_tokens, piece) == _new(
        args.num_tokens, args.min_tokens, piece
    )

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        old_ns = _median_ns(
            _old,
            repeats=args.repeats,
            num_tokens=args.num_tokens,
            min_tokens=args.min_tokens,
            piece=piece,
        )
        new_ns = _median_ns(
            _new,
            repeats=args.repeats,
            num_tokens=args.num_tokens,
            min_tokens=args.min_tokens,
            piece=piece,
        )
    finally:
        if gc_was_enabled:
            gc.enable()

    denom = max(1, args.num_tokens)
    old_ns_per_tok = old_ns / denom
    new_ns_per_tok = new_ns / denom
    speedup = old_ns / new_ns if new_ns else float("inf")
    pct = (old_ns - new_ns) / old_ns * 100.0 if old_ns else 0.0

    print(f"old: {old_ns/1e6:.3f} ms ({old_ns_per_tok:.2f} ns/token)")
    print(f"new: {new_ns/1e6:.3f} ms ({new_ns_per_tok:.2f} ns/token)")
    print(f"speedup: {speedup:.3f}x (time -{pct:.2f}%)")


if __name__ == "__main__":
    main()
```

And get:

```bash
yewentao256@dgx-b200-02:~/vllm-source$ python benchmarks/overheads/benchmark_detokenizer_min_tokens.py 
old: 787.270 ms (39.36 ns/token)
new: 483.633 ms (24.18 ns/token)
speedup: 1.628x (time -38.57%)
```
---
### njhill @ 2026-02-05T22:22:56Z

Actually @yewentao256 I don't think this change makes sense because in the most common case (without spec decoding), `new_token_ids` will only contain one token. And it actually is more work/allocation for this case. The benchmark doesn't reflect that.
---
### yewentao256 @ 2026-02-06T16:18:52Z

> Actually @yewentao256 I don't think this change makes sense because in the most common case (without spec decoding), `new_token_ids` will only contain one token. And it actually is more work/allocation for this case. The benchmark doesn't reflect that.

I agree with you, so I updated the code, for common decode path we do as usual, and for spec decoding or perhaps jump decoding later, we will use the optimized path.
---
### njhill @ 2026-02-06T17:00:30Z

@yewentao256 TBH I don't think this is worth the added complexity. Maybe just keep it to the num_output_tokens change?
---
### yewentao256 @ 2026-02-06T21:47:34Z

> @yewentao256 TBH I don't think this is worth the added complexity. Maybe just keep it to the num_output_tokens change?

OK, I reverted the change, please take another look
---