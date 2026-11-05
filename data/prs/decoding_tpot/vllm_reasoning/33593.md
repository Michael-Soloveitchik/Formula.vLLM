# PR #33593: [Perf] Optimize Python Slice for Structured Output using `islice` instead of [:]

- field: decoding  | area: vllm/reasoning
- author: yewentao256
- state: MERGED  created: 2026-02-02T21:19:02Z  merged: 2026-02-24T17:02:36Z
- url: https://github.com/vllm-project/vllm/pull/33593
- labels: structured-output, ready, v1, deepseek

## Body

## Purpose

Python split -> islice (iterator) and get perf improvement.

## Test

### Micro benchmark

```py
import time, tracemalloc
from itertools import islice

SIZE, N, R = 2_000_000, 1_000_000, 5
lst = list(range(SIZE))

def bench(name, f):
    t_best = 1e9
    for _ in range(R):
        tracemalloc.start()
        t0 = time.perf_counter()
        x = f()
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        t_best = min(t_best, dt)
    print(f"{name:22}  best={t_best:.6f}s  peak={peak/1024/1024:.2f} MiB  (sink={int(x)&0xffff})")

bench("create slice",  lambda: id(lst[:N]))
bench("create islice", lambda: id(islice(lst, N)))
bench("sum slice",     lambda: sum(lst[:N]))
bench("sum islice",    lambda: sum(islice(lst, N)))
```

```bash
create slice            best=0.003368s  peak=7.63 MiB  (sink=53504)
create islice           best=0.000001s  peak=0.00 MiB  (sink=17888)
sum slice               best=0.005554s  peak=7.63 MiB  (sink=59104)
sum islice              best=0.004699s  peak=0.00 MiB  (sink=59104)
```

### E2E

`vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   --reasoning-parser deepseek_r1   --port 8000`
`python benchmarks/benchmark_serving_structured_output.py   --backend vllm   --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   --dataset xgrammar_bench   --structured-output-ratio 1.0   --request-rate 5   --num-prompts 600   --output-len 1024   --ignore-eos   --metric-percentiles 50,90,99   --percentile-metrics ttft,tpot,itl,e2el   --port 8000`

```bash
# now
============ Serving Benchmark Result ============
Successful requests:                     600       
Request rate configured (RPS):           5.00      
Benchmark duration (s):                  129.53    
Total input tokens:                      172662    
Total generated tokens:                  610202    
Request throughput (req/s):              4.63      
Output token throughput (tok/s):         4711.02   
Total token throughput (tok/s):          6044.05   
---------------Time to First Token----------------
Mean TTFT (ms):                          38.12     
Median TTFT (ms):                        24.58     
P50 TTFT (ms):                           24.58     
P90 TTFT (ms):                           72.82     
P99 TTFT (ms):                           162.13    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.82      
Median TPOT (ms):                        6.49      
P50 TPOT (ms):                           6.49      
P90 TPOT (ms):                           10.89     
P99 TPOT (ms):                           11.83     
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.77      
Median ITL (ms):                         5.50      
P50 ITL (ms):                            5.50      
P90 ITL (ms):                            5.93      
P99 ITL (ms):                            90.92     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          7982.39   
Median E2EL (ms):                        6625.19   
P50 E2EL (ms):                           6625.19   
P90 E2EL (ms):                           11124.37  
P99 E2EL (ms):                           12082.27  
==================================================
correct_rate(%) 5.666666666666666 
# main
============ Serving Benchmark Result ============
Successful requests:                     600       
Request rate configured (RPS):           5.00      
Benchmark duration (s):                  129.92    
Total input tokens:                      172662    
Total generated tokens:                  609643    
Request throughput (req/s):              4.62      
Output token throughput (tok/s):         4692.52   
Total token throughput (tok/s):          6021.52   
---------------Time to First Token----------------
Mean TTFT (ms):                          47.37     
Median TTFT (ms):                        26.91     
P50 TTFT (ms):                           26.91     
P90 TTFT (ms):                           103.25    
P99 TTFT (ms):                           228.21    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.52      
Median TPOT (ms):                        7.91      
P50 TPOT (ms):                           7.91      
P90 TPOT (ms):                           11.74     
P99 TPOT (ms):                           12.06     
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.45      
Median ITL (ms):                         5.49      
P50 ITL (ms):                            5.49      
P90 ITL (ms):                            5.92      
P99 ITL (ms):                            96.20     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          8686.66   
Median E2EL (ms):                        8065.08   
P50 E2EL (ms):                           8065.08   
P90 E2EL (ms):                           11973.36  
P99 E2EL (ms):                           12284.38  
==================================================
correct_rate(%) 6.5 
```


## Comments

### mergify @ 2026-02-02T23:07:39Z

Hi @yewentao256, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mgoin @ 2026-02-03T16:39:44Z

Please give a more descriptive title and show perf improvement. The downside of this is that it is much less readable to me
---
### yewentao256 @ 2026-02-03T17:02:55Z

> Please give a more descriptive title and show perf improvement. The downside of this is that it is much less readable to me

OK added in the title, the e2e perf improvement is smaller than < 0.5% so hidden in fluctuations.

But certainly good for perf, eg

```py
import time, tracemalloc
from itertools import islice

SIZE, N, R = 2_000_000, 1_000_000, 5
lst = list(range(SIZE))

def bench(name, f):
    t_best = 1e9
    for _ in range(R):
        tracemalloc.start()
        t0 = time.perf_counter()
        x = f()
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        t_best = min(t_best, dt)
    print(f"{name:22}  best={t_best:.6f}s  peak={peak/1024/1024:.2f} MiB  (sink={int(x)&0xffff})")

bench("create slice",  lambda: id(lst[:N]))
bench("create islice", lambda: id(islice(lst, N)))
bench("sum slice",     lambda: sum(lst[:N]))
bench("sum islice",    lambda: sum(islice(lst, N)))
```

```bash
create slice            best=0.003368s  peak=7.63 MiB  (sink=53504)
create islice           best=0.000001s  peak=0.00 MiB  (sink=17888)
sum slice               best=0.005554s  peak=7.63 MiB  (sink=59104)
sum islice              best=0.004699s  peak=0.00 MiB  (sink=59104)
```
---
### mgoin @ 2026-02-03T21:02:28Z

I meant adding in the title "what did you optimize it for" rather than generic "optimized slice". I guess we can merge this if you really want, but I struggle to see the point of micro-optimizations like this without clear purpose
---
### yewentao256 @ 2026-02-03T21:10:32Z

> I meant adding in the title "what did you optimize it for" rather than generic "optimized slice". I guess we can merge this if you really want, but I struggle to see the point of micro-optimizations like this without clear purpose

Ohh I see, this is for structured output related logic.

We do hope to land this change as it both save memory and improve performance (although small), but small optimizations gathered can make a difference I think.
---
### mgoin @ 2026-02-06T16:34:46Z

cc @njhill @chaunceyjiang do you think these optimizations are worth the trouble? I'm leaning towards no unless we have a workload that benefits
---
### njhill @ 2026-02-09T16:51:06Z

> cc @njhill @chaunceyjiang do you think these optimizations are worth the trouble? I'm leaning towards no unless we have a workload that benefits

Agree... maybe the `reversed()` is fine. IIRC others are looking at more holistic optimization of the structured outputs / reasoning logic.
---
### yewentao256 @ 2026-02-19T19:37:29Z

@mgoin ok, I have updated the benchmark in the PR description, it is relatively small but we could get some benefit from
---