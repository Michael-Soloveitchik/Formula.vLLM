# PR #27869: [Perf] Support stream interval for reducing host overhead

- field: scheduling  | area: v1/engine
- author: elvischenv
- state: MERGED  created: 2025-10-31T08:26:08Z  merged: 2025-11-13T18:21:25Z
- url: https://github.com/vllm-project/vllm/pull/27869
- labels: performance, frontend, ready, v1

## Body

## Purpose

Support `--stream-interval <num_of_tokens>`. The flag will buffer a number of tokens for a single request before sending back to client. The buffering can reduce the time needed to send responses back to clients, thereby saving host overhead like network/IO.

cc @benchislett @pavanimajety @nvpohanh 

## Testing with `openai/gpt-oss-20b`:
B200 server cmd:
```
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
vllm serve openai/gpt-oss-20b \
--kv-cache-dtype fp8 \
--tensor-parallel-size 1 \
--max-num-seqs 1024 \
--max-model-len 10240 \
--max-num-batched-tokens 8192 \
--max-cudagraph-capture-size 2048 \
-O.pass_config.enable_noop=True \
--async-scheduling \
--no-enable-prefix-caching
```

## Accuracy Test
```
OPENAI_API_KEY="test" \
python -m gpt_oss.evals \
--sampler chat_completions \
--model openai/gpt-oss-20b \
--reasoning-effort medium \
--n-threads 512 \
--eval aime25
```
Without `--stream-interval`(by default is set to `1`):
```
[{'eval_name': 'aime25', 'model_name': 'gpt-oss-20b-medium_temp1.0_20251031_095139', 'metric': 0.7083333333333334}]
```
With `--stream-interval 10`:
```
[{'eval_name': 'aime25', 'model_name': 'gpt-oss-20b-medium_temp1.0_20251031_150201', 'metric': 0.7291666666666666}]
```

## Perf Test
```
vllm bench serve \
--model openai/gpt-oss-20b \
--trust-remote-code \
--dataset-name random \
--ignore-eos \
--max-concurrency 1024 \
--num-prompts 5120 \
--random-input-len 1024 \
--random-output-len 1024
```
Without `--stream-interval`(by default is set to `1`):
```
============ Serving Benchmark Result ============
Successful requests:                     5120
Failed requests:                         0
Maximum request concurrency:             1024
Benchmark duration (s):                  153.85
Total input tokens:                      5242880
Total generated tokens:                  5242880
Request throughput (req/s):              33.28
Output token throughput (tok/s):         34078.85
Peak output token throughput (tok/s):    42240.00
Peak concurrent requests:                1554.00
Total Token throughput (tok/s):          68157.70
---------------Time to First Token----------------
Mean TTFT (ms):                          2698.80
Median TTFT (ms):                        2300.56
P99 TTFT (ms):                           6826.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.16
Median TPOT (ms):                        27.40
P99 TPOT (ms):                           28.57
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.41
Median ITL (ms):                         25.22
P99 ITL (ms):                            89.70
==================================================
```
With `--stream-interval 10`: **57% e2e perf gain**
```
============ Serving Benchmark Result ============
Successful requests:                     5120
Failed requests:                         0
Maximum request concurrency:             1024
Benchmark duration (s):                  98.17
Total input tokens:                      5242880
Total generated tokens:                  5242880
Request throughput (req/s):              52.15
Output token throughput (tok/s):         53403.94
Peak output token throughput (tok/s):    8216.00
Peak concurrent requests:                1180.00
Total Token throughput (tok/s):          106807.88
---------------Time to First Token----------------
Mean TTFT (ms):                          1002.64
Median TTFT (ms):                        262.37
P99 TTFT (ms):                           6749.87
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.09
Median TPOT (ms):                        18.68
P99 TPOT (ms):                           18.85
---------------Inter-token Latency----------------
Mean ITL (ms):                           179.65
Median ITL (ms):                         137.12
P99 ITL (ms):                            512.61
==================================================
```


---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### elvischenv @ 2025-10-31T08:55:52Z

@codex review
---
### elvischenv @ 2025-10-31T09:03:47Z

/gemini review
---
### vadiklyutiy @ 2025-10-31T13:31:21Z

Could you add to perf results command line that use used for server and client and what GPU it was tested
---
### hmellor @ 2025-10-31T15:48:24Z

Potentially a duplicate of https://github.com/vllm-project/vllm/pull/27376
---
### elvischenv @ 2025-10-31T16:15:48Z

@codex review
---
### chatgpt-codex-connector @ 2025-10-31T16:20:34Z

Codex Review: Didn't find any major issues. Keep it up!

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

Codex has been enabled to automatically review pull requests in this repo. Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.




When you [sign up for Codex through ChatGPT](https://openai.com/codex), Codex can also answer questions or update the PR, like "@codex address that feedback".
            
</details>
---
### elvischenv @ 2025-10-31T16:21:04Z

/gemini review
---
### mergify @ 2025-11-03T01:20:34Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @elvischenv.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### heheda12345 @ 2025-11-04T04:42:46Z

CC @aarnphm @chaunceyjiang is it possible to implement all logic in api server?
---
### FENP @ 2025-11-04T06:32:59Z

Hi @elvischenv. Thanks for proposing this exciting feature! 
I'm trying to understand how the TTFT gain is achieved with `--stream-interval`. IIUC, first token should be sended back to client as soon as possible to reduce TTFT.

Would you mind sharing some insight into the key contributing factors? Any before/after profiling numbers would also be super helpful!

Thanks again for your work on this! 🙌

---
### elvischenv @ 2025-11-04T09:45:13Z

> I'm trying to understand how the TTFT gain is achieved with --stream-interval. IIUC, first token should be sended back to client as soon as possible to reduce TTFT.
>
> Would you mind sharing some insight into the key contributing factors? Any before/after profiling numbers would also be super helpful!

@FENP Thanks for the question. Initially I don't expect this PR will bring so much TTFT perf gain but I do get similar perf numbers again after redoing the benchmarking. My implementation will just send the response when `self.sent_tokens_offset == 0`, that means it is the first response to send back to client. Then the offset will be moved forward and the tokens are batched by stream_interval to send.
```
            # should send output when it is the first token or reach the stream interval
            should_send_output = (
                self.sent_tokens_offset == 0
                or self.total_num_output_tokens - self.sent_tokens_offset
                >= self.stream_interval
            )
```
One possible reason is system resource contention. If stream_interval=1 and concurrency=1024, it means that 1024 generated responses will simultaneously enter the output processing. However, if stream_interval=10, the number of responses will be reduced by 10x, and the output processing speed may be significantly improved.
---
### nvpohanh @ 2025-11-05T08:51:36Z

@FENP I think what @elvischenv said makes sense. Basically, in the original case where output token throughput is very large, we are actually bounded by output http issue rate, so TTFT looks bad because the first output http message is stuck in the long queue (I think). 
---
### FENP @ 2025-11-06T08:03:33Z

@elvischenv @nvpohanh Got it — thank you for clarifying!
FYI, I noticed a difference in `Peak concurrent requests` in the benchmark test results. I suspect that `--stream-interval` indirectly affects it by introducing delays in token streaming, similar to how `--request-rate` works in the bench serve ?


---
### nvpohanh @ 2025-11-06T08:25:59Z

> I suspect that --stream-interval indirectly affects it by introducing delays in token streaming

@FENP Do you mean the "input" token streaming or "output" token streaming? `--stream-interval` controls how the server streams the output tokens to the client. It is not related to how the input tokens are streamed into the server or how the requests are submitted to the server.
---