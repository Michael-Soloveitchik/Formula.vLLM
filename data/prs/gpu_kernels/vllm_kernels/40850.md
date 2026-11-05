# PR #40850: [Kernel][Helion] Optimize Helion config parsing latency

- field: gpu_kernels  | area: vllm/kernels
- author: gmagogsfm
- state: MERGED  created: 2026-04-25T01:36:21Z  merged: 2026-05-08T03:27:35Z
- url: https://github.com/vllm-project/vllm/pull/40850
- labels: ready

## Body

This PR optimizes Helion compilation time.

Previously config picking is not cached, so every cuda graph capture size (per num_token) needs to go through all stored configs for matching, and each matching requires expensive RegEx operations due to free-form string-based config key.

Two improvements are made:
- Replace regex-based config key parsing with structured dict config keys.  eliminating the per-call regex overhead 
- Deserializes keys once on load and passes parsed dicts to kernel pick_config functions with caching enabled, so for different shapes config parsing doesn't need to be done repeatedly


Benchmark: 80,000 calls to `pick_config` with 300 config keys, 8 different input shapes. Measured on NVIDIA H100.

| Approach | us/call | Speedup |
|---|---|---|
| Old (regex, no cache) | 1289 | 1.0× |
| New (CaseKey, no cache) | 46 | 28× |
| New (CaseKey, cached) | 1.8 | 719× |


## Comments

### ProExpertProg @ 2026-04-25T02:07:42Z

Nice! Will this mean we can use helion without cudagraphs? 👀
---
### gmagogsfm @ 2026-04-25T02:12:21Z

> Nice! Will this mean we can use helion without cudagraphs? 👀

Unfortunately still not yet, but I will look at Helion's compilation and dispatching overhead deeper to optimize as much as possible.
---
### gmagogsfm @ 2026-04-27T16:24:16Z

@zou3519 @BoyuanFeng @ProExpertProg @xiaohongchen1991 Could you take a look when you get a chance? Thanks

Note to @xiaohongchen1991 this will change serialization format of configs, so you may need to update your inflight PRs. Thankfully claude can do a decent job in updating at scale.
---