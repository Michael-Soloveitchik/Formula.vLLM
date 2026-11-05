#!/usr/bin/env python3
"""Benchmark for vLLM PR #32975 "[Perf] Optimize detokenizer python logic".

The patch adds IncrementalDetokenizer.num_output_tokens() so the hot
BaseIncrementalDetokenizer.update() loop stops calling
len(self.output_token_ids). For SlowIncrementalDetokenizer, output_token_ids
is a property that recomputes a list slice `self.token_ids[self.prompt_len:]`
on every access; update() hits it per decoded token plus once for the stop
check. The patch replaces those with num_output_tokens() == len(token_ids) -
prompt_len, eliminating the per-token slice allocation.

PREFERRED (real-path): import the real SlowIncrementalDetokenizer from the
mounted vllm tree (PYTHONPATH=/src), build it against a real 'gpt2' tokenizer,
and drive the real update() loop over a long token sequence. The base tree has
no num_output_tokens() so update() takes the slice path; the gold tree defines
num_output_tokens() so update() takes the tracked path. Which tree is mounted
at /src is what makes an arm "base" vs "gold" -- the driver code is identical.
This measures the FULL detokenizer python path (real decode_next included),
matching what the PR reports.

FALLBACK (honest, self-contained): if the import chain or the gpt2 tokenizer
is unavailable (e.g. no network in container), reproduce the exact changed
inner comparison both ways in one process over synthetic token lists.
"""
import os
import sys
import time
import statistics

N_SEQS = int(os.environ.get("N_SEQS", "200"))
N_TOKS = int(os.environ.get("N_TOKS", "2000"))
# Long prompt so the SlowIncrementalDetokenizer.output_token_ids slice
# `token_ids[prompt_len:]` (recomputed per update in the base path) is large --
# this is the cost the PR eliminates with num_output_tokens().
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "512"))
# Tokens delivered per update() call. The PR's removed TODO noted the loop is
# "very inefficient when the number of new_token_ids is more than 1".
BATCH = int(os.environ.get("BATCH", "8"))
REPEATS = int(os.environ.get("REPEATS", "7"))


# --------------------------------------------------------------------------- #
# real-path driver
# --------------------------------------------------------------------------- #
def _build_real_detokenizer():
    """Return a constructed real SlowIncrementalDetokenizer bound to gpt2, or
    raise if the real path is not usable (caller falls back)."""
    from transformers import AutoTokenizer
    from vllm.v1.engine.detokenizer import (
        SlowIncrementalDetokenizer,
        BaseIncrementalDetokenizer,
    )
    from vllm.tokenizers.detokenizer_utils import convert_prompt_ids_to_tokens

    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt_ids = list(range(PROMPT_LEN))

    # Construct without going through EngineCoreRequest (msgspec Struct,
    # version-fragile). We set exactly the attributes __init__ would set.
    d = SlowIncrementalDetokenizer.__new__(SlowIncrementalDetokenizer)
    # IncrementalDetokenizer.__init__
    d.token_ids = []
    # BaseIncrementalDetokenizer.__init__ attrs
    d.stop = []
    d.min_tokens = 0
    d.include_stop_str_in_output = False
    d.stop_buffer_length = 0
    d._last_output_text_offset = 0
    d.output_text = ""
    # SlowIncrementalDetokenizer.__init__ attrs
    d.tokenizer = tok
    d.prompt_len = PROMPT_LEN
    d.tokens, d.prefix_offset, d.read_offset = convert_prompt_ids_to_tokens(
        tokenizer=tok, prompt_ids=prompt_ids, skip_special_tokens=False
    )
    d.token_ids.extend(prompt_ids)
    d.skip_special_tokens = False
    d.spaces_between_special_tokens = True
    return SlowIncrementalDetokenizer, d


def _time_real(cls_and_factory) -> float:
    """Drive the real update() loop one token at a time; return us/tok.
    base tree -> slice path; gold tree -> tracked path. Identical driver."""
    from transformers import AutoTokenizer
    from vllm.v1.engine.detokenizer import SlowIncrementalDetokenizer
    from vllm.tokenizers.detokenizer_utils import convert_prompt_ids_to_tokens

    tok = AutoTokenizer.from_pretrained("gpt2")
    vocab = tok.vocab_size
    prompt_ids = list(range(PROMPT_LEN))
    total = 0
    t0 = time.perf_counter()
    for s in range(N_SEQS):
        d = SlowIncrementalDetokenizer.__new__(SlowIncrementalDetokenizer)
        d.token_ids = []
        d.stop = []
        d.min_tokens = 0
        d.include_stop_str_in_output = False
        d.stop_buffer_length = 0
        d._last_output_text_offset = 0
        d.output_text = ""
        d.tokenizer = tok
        d.prompt_len = PROMPT_LEN
        d.tokens, d.prefix_offset, d.read_offset = convert_prompt_ids_to_tokens(
            tokenizer=tok, prompt_ids=prompt_ids, skip_special_tokens=False
        )
        d.token_ids.extend(prompt_ids)
        d.skip_special_tokens = False
        d.spaces_between_special_tokens = True
        base = (s * 7 + 100) % (vocab - N_TOKS - 1)
        i = 0
        while i < N_TOKS:
            n = min(BATCH, N_TOKS - i)
            d.update([base + i + j for j in range(n)], stop_terminated=False)
            i += n
            total += n
    dt = time.perf_counter() - t0
    return dt / total * 1e6


# --------------------------------------------------------------------------- #
# fallback driver (self-contained, both arms in one process)
# --------------------------------------------------------------------------- #
MIN_TOKENS = 16


def _time_fallback(use_tracking: bool) -> float:
    total = 0
    t0 = time.perf_counter()
    for _ in range(N_SEQS):
        token_ids = list(range(PROMPT_LEN))
        prompt_len = PROMPT_LEN
        output_text = ""
        for i in range(N_TOKS):
            token_ids.append(i & 0xFFFF)
            output_text += "x"
            if use_tracking:
                n_out = len(token_ids) - prompt_len
            else:
                n_out = len(token_ids[prompt_len:])
            if MIN_TOKENS and n_out <= MIN_TOKENS:
                stop_check_offset = len(output_text)  # noqa: F841
            total += 1
        if use_tracking:
            _ = (len(token_ids) - prompt_len) > MIN_TOKENS
        else:
            _ = len(token_ids[prompt_len:]) > MIN_TOKENS
    dt = time.perf_counter() - t0
    return dt / total * 1e6


# --------------------------------------------------------------------------- #
def _try_real():
    try:
        cf = _build_real_detokenizer()
        # smoke: one real update must work
        cls, d = cf
        d.update([d.token_ids[-1] + 1 if d.token_ids else 100], False)
        return cf
    except Exception as e:
        print(f"# real-path unavailable, using fallback: {e!r}", file=sys.stderr)
        return None


def main():
    arm = os.environ.get("ARM", "").strip().lower()
    real = _try_real()

    if real is not None and arm in ("base", "gold"):
        # Real path: this container has exactly one tree mounted; measure it.
        runs = [_time_real(real) for _ in range(REPEATS)]
        for v in runs:
            print(f"detokenize {arm} {v:.6f}")
        print(f"# approach=real", file=sys.stderr)
        print(f"# approach=real")
        print(f"# {arm}_mean_us_per_tok={statistics.mean(runs):.6f}")
        return

    # Fallback: run BOTH arms in one process (honest, same CPU/process).
    base_runs = [_time_fallback(False) for _ in range(REPEATS)]
    gold_runs = [_time_fallback(True) for _ in range(REPEATS)]
    for v in base_runs:
        print(f"detokenize base {v:.6f}")
    for v in gold_runs:
        print(f"detokenize gold {v:.6f}")
    ratios = [b / g for b, g in zip(base_runs, gold_runs)]
    print("# approach=fallback")
    print(f"# base_mean_us_per_tok={statistics.mean(base_runs):.6f}")
    print(f"# gold_mean_us_per_tok={statistics.mean(gold_runs):.6f}")
    print(f"# speedup_x={statistics.mean(ratios):.6f}")


if __name__ == "__main__":
    main()
