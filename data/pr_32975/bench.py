#!/usr/bin/env python3
"""Microbench for vLLM PR #32975: [Perf] Optimize detokenizer python logic.

BaseIncrementalDetokenizer.update() checks `len(self.output_token_ids)` per decoded token
(min_tokens) and once per update (stop strings). For SlowIncrementalDetokenizer,
`output_token_ids` is a property that materialises `self.token_ids[self.prompt_len:]` - a copy of
all generated tokens on every call. The PR adds `num_output_tokens()` (= len(token_ids) -
prompt_len) and uses it in update() and in OutputProcessor.make_request_output.

The bench drives the arm's real `SlowIncrementalDetokenizer.update()` one token at a time (decode)
with stop strings set (so the per-update length check runs, as in any request with `stop`),
over N_SEQS generations of OUTPUT_LEN tokens after a PROMPT_LEN prompt. The tokenizer is a small
byte-level BPE trained in-process (no download, no network). us per generated token, mean over
REPEATS passes; total work ~N_SEQS*OUTPUT_LEN*REPEATS updates (bounded, well under a minute).
Author gives a 1.628x micro ratio only (no absolute time).
Runs inside the task image: python3 /task/bench.py
"""
import random
import time
from types import SimpleNamespace

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.detokenizer import SlowIncrementalDetokenizer

N_SEQS = 8
PROMPT_LEN = 256
OUTPUT_LEN = 2048
REPEATS = 5
VOCAB_SIZE = 2000


def build_tokenizer():
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "<s>", "</s>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    rng = random.Random(0)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "vllm", "token",
             "stream", "detokenizer", "python", "logic", "fast", "slow", "cache", "block",
             "model", "serve", "request", "output", "prompt", "stop", "string"]
    corpus = [" ".join(rng.choice(words) for _ in range(40)) + "." for _ in range(400)]
    tok.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   bos_token="<s>", eos_token="</s>")


def make_detokenizer(tokenizer, prompt_ids):
    request = SimpleNamespace(
        prompt_token_ids=prompt_ids,
        prompt_embeds=None,
        sampling_params=SamplingParams(stop=["<<never>>"], min_tokens=0, max_tokens=OUTPUT_LEN),
    )
    return SlowIncrementalDetokenizer(tokenizer, request)


def main():
    tokenizer = build_tokenizer()
    vocab = tokenizer.get_vocab()
    ids = [i for i in range(len(vocab)) if i not in tokenizer.all_special_ids]
    rng = random.Random(1)
    prompts = [[rng.choice(ids) for _ in range(PROMPT_LEN)] for _ in range(N_SEQS)]
    outputs = [[rng.choice(ids) for _ in range(OUTPUT_LEN)] for _ in range(N_SEQS)]

    d = make_detokenizer(tokenizer, prompts[0])
    for t in outputs[0][:64]:
        d.update([t], False)
    assert len(d.output_text) > 0 and len(d.output_token_ids) == 64

    per_tok = []
    for _ in range(REPEATS):
        dets = [make_detokenizer(tokenizer, p) for p in prompts]
        t0 = time.perf_counter()
        for d, out in zip(dets, outputs):
            for t in out:
                d.update([t], False)
        per_tok.append((time.perf_counter() - t0) * 1e6 / (N_SEQS * OUTPUT_LEN))
    per_tok.sort()
    print(f"detokenize {per_tok[len(per_tok) // 2]:.4f}")
    print(f"# seqs={N_SEQS} prompt={PROMPT_LEN} output={OUTPUT_LEN} repeats={REPEATS} "
          f"stop_strings=1 runs_us_per_tok={[round(x, 3) for x in per_tok]}")


if __name__ == "__main__":
    main()
