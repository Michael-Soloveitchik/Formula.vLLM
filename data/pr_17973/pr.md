# PR #17973: [PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…

- field: attention  | area: model_executor/models
- author: vadiklyutiy
- state: MERGED  created: 2025-05-12T01:42:53Z  merged: 2025-05-16T06:31:02Z
- url: https://github.com/vllm-project/vllm/pull/17973
- labels: ready, qwen

## Body

### Description of Problem
In Qwen2.5-VL rotary position embedding constant tensors creates in the beginning of model's `forward`. 
Before this PR there were a mix of CPU and GPU tensors and (small) data pieces transferred back and forward to device. 
Profile looked like below 

![image](https://github.com/user-attachments/assets/777fd03d-8eb7-4f87-b008-50fe5400fc46)
pink `tmp` is begining of `Qwen2_5_VisionTransformer.forward()` before main transformer started.   

### Solution
This PR:
 - makes a refactoring and put all tensors necessary to create constant mrope data to CPU (similar to how it works for mrope for language (part of) models)
 - regroup calculation by `grid_thw` line and cache results 

Now profile looks like below 
![image](https://github.com/user-attachments/assets/1e98a81d-9cd6-4642-8a74-c8590cbc81eb)



### Performance results
Run Qwen2.5-3B-VL on H100 with following command line 
`vllm serve Qwen/Qwen2.5-VL-3B-Instruct --disable-log-requests --max-num-seqs 1024  --block-size 16 --max-num-batched-tokens 2048`

Construction of constant mrope tensors itself speeded up **5+ times**. 

E2E measured with https://github.com/CentML/flexible-inference-bench
```
fib benchmark -rps 50 --input-token-distribution uniform 250 300 --output-token-distribution uniform 150 250 --num-of-imgs-per-req 1 --img-ratios-per-req 512x512 -n 1000 --base-url http://localhost:8000 --endpoint v1/chat/completions --backend openai-chat
```
The above runs 1000 requests, 50 reqs/sec, every request has one 512x512 image. Measured average reqs/s. Made 11 runs and took median

Before: 25.99 reqs/s
After: 26.63 req/s
**Speed up: 2.46%**

### Correctness
Run `lm_eval` with chartqa and mmmu
```
lm_eval --model vllm-vlm --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,model=Qwen/Qwen2.5-VL-3B-Instruct"  --tasks mmmu_val,chartqa  --batch_size 32 --apply_chat_template
```

**Before**
```
|                 Tasks                 |Version|Filter|n-shot|     Metric      |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|-----------------|---|-----:|---|-----:|
|chartqa                                |      0|none  |     0|anywhere_accuracy|↑  |0.8072|±  |0.0079|
|                                       |       |none  |     0|exact_match      |↑  |0.5712|±  |0.0099|
|                                       |       |none  |     0|relaxed_accuracy |↑  |0.8040|±  |0.0079|

|             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------------------------|------:|------|------|------|---|-----:|---|-----:|
|mmmu_val                        |      0|none  |      |acc   |↑  |0.4567|±  |0.0159|
| - Art and Design               |      0|none  |      |acc   |↑  |0.5583|±  |0.0437|
| - Business                     |      0|none  |      |acc   |↑  |0.3733|±  |0.0395|
| - Health and Medicine          |      0|none  |      |acc   |↑  |0.5267|±  |0.0406|
| - Humanities and Social Science|      0|none  |      |acc   |↑  |0.7000|±  |0.0412|
| - Science                      |      0|none  |      |acc   |↑  |0.3267|±  |0.0386|
| - Tech and Engineering         |      0|none  |      |acc   |↑  |0.3619|±  |0.0326|
```

**After**
```
|                 Tasks                 |Version|Filter|n-shot|     Metric      |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|-----------------|---|-----:|---|-----:|
|chartqa                                |      0|none  |     0|anywhere_accuracy|↑  |0.8032|±  |0.0080|
|                                       |       |none  |     0|exact_match      |↑  |0.5756|±  |0.0099|
|                                       |       |none  |     0|relaxed_accuracy |↑  |0.8016|±  |0.0080|

|             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------------------------|------:|------|------|------|---|-----:|---|-----:|
|mmmu_val                        |      0|none  |      |acc   |↑  |0.4544|±  |0.0159|
| - Art and Design               |      0|none  |      |acc   |↑  |0.5583|±  |0.0443|
| - Business                     |      0|none  |      |acc   |↑  |0.3733|±  |0.0395|
| - Health and Medicine          |      0|none  |      |acc   |↑  |0.5067|±  |0.0407|
| - Humanities and Social Science|      0|none  |      |acc   |↑  |0.7083|±  |0.0411|
| - Science                      |      0|none  |      |acc   |↑  |0.3267|±  |0.0386|
| - Tech and Engineering         |      0|none  |      |acc   |↑  |0.3619|±  |0.0327|
```







 


## Comments

### github-actions @ 2025-05-12T01:43:01Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### vadiklyutiy @ 2025-05-15T01:02:03Z

> thank you for the optimization, please run a mmmu or chartqa evaluation to verify the correctness of the changes.

I added to description results of mmmu and chartqa "before" and "after"
---
### WoosukKwon @ 2025-05-15T01:39:08Z

@imkero Could you please take a final look? I'm not sure if this overlaps with #14684 
---
### mergify @ 2025-05-15T05:27:21Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @vadiklyutiy.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### WoosukKwon @ 2025-05-16T05:46:46Z

@vadiklyutiy QQ: Why does this PR change the accuracy (though the diff is small)? I thought the PR doesn't change the computation at all. Can we somehow strictly match the accuracy? I'm a bit careful about this because we've seen a few bugs regarding m-rope.
---
### simon-mo @ 2025-05-16T05:59:39Z

<!-- linear:replyTo:2841667391 -->

@WoosukKwon these tests are not deterministic due to temperature, I read values and apply the stderr; seems no change to accuracy to me.  
---
### imkero @ 2025-05-16T06:10:24Z

The idea of this PR is similar to #14684. And it is verified by both #14684 and this PR that such approach will gain some performance improvement.

If the inference result slightly changed in this PR, maybe we should compare the generated m-rope pos seq and window_index seq  output with those generated by main branch. Also check if we are testing with greedy decoding.

By the way I suggest that we can keep `image_grid_thw` and `video_grid_thw` in CPU all the time by modifying `vllm/multimodal/inputs.py::MultiModalKwargs::as_kwargs` (here vLLM move all mm data to device by default, and still needed to move them back to host later)

```diff
  @staticmethod
  def as_kwargs(
      batched_inputs: BatchedTensorInputs,
      *,
      device: torch.types.Device,
  ) -> BatchedTensorInputs:
      json_inputs = cast(JSONTree[torch.Tensor], batched_inputs)

+     # keep Qwen2/2.5-VL's image_grid_thw and video_grid_thw in cpu
+     image_grid_thw = None
+     video_grid_thw = None
+     if isinstance(json_inputs, dict):
+         image_grid_thw = json_inputs.pop("image_grid_thw", None)
+         video_grid_thw = json_inputs.pop("video_grid_thw", None)

      json_mapped = json_map_leaves(
          lambda x: x.to(device, non_blocking=True),
          json_inputs,
      )

+     if image_grid_thw is not None:
+         json_mapped["image_grid_thw"] = image_grid_thw  # type: ignore
+     if video_grid_thw is not None:
+         json_mapped["video_grid_thw"] = video_grid_thw  # type: ignore

      return cast(BatchedTensorInputs, json_mapped)
```
---
### WoosukKwon @ 2025-05-16T06:30:28Z

@simon-mo @imkero Thanks for the explanation. Ok let's merge this PR for v0.9.0 and further improve it with @imkero's suggestion
---
### vadiklyutiy @ 2025-05-16T15:44:33Z

@WoosukKwon 
As @simon-mo said `lm_eval` isn't deterministic. 

To dispel doubts in correctness I wrote the following test that compare "before" and "after" implementations. 
In test I took `Qwen2_5_VisionTransformer` before and after and copy to test. Clean both to calculate only `rotary_pos_emb`, `window_index`, `cu_window_seqlens`, and `cu_seqlens`. Test takes arbitrary `grid_thw`, run both version and compare results. 
Test accept following args
`--samples` number of different grid to test 
`--max-t`  max value of t
`--max-h` max value of h
`--max-w` max value of w
`--max-images` - `len(grid_thw)`

The following runs successfully passed:
```
$ python test_qwen25_vl_transformer.py --mass-test --samples 10000 --max-t 50 --max-h 100 --max-w 100 --max-images 5
```
```
$python test_qwen25_vl_transformer.py --mass-test --samples 10000 --max-t 100 --max-h 250 --max-w 250 --max-images 10
```

Hope that resolved worries about correctness 


<details>

<summary>Test source</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
import argparse
import numpy as np
import random
import tqdm
import sys

class TestFailureException(Exception):
    """Exception raised when the test results don't match between old and new implementations."""
    pass

class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta**(
            torch.arange(0, dim, 2, dtype=torch.float, device='cpu') / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (self.theta**(torch.arange(
                0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
                                                / self.dim))
            seq = torch.arange(seqlen,
                               device=self.inv_freq.device,
                               dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]

class Qwen2_5_VisionTransformer_New(nn.Module):
    def __init__(
        self,
        hidden_size=1152,
        num_heads=16,
        window_size=32,
        patch_size=14,
        spatial_merge_size=2,
        fullatt_block_indexes=[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2
        
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device('cpu')

    def rotary_pos_emb_thw(self, t, h, w):
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).permute(0, 2, 1, 3).flatten()
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).permute(0, 2, 1, 3).flatten()
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit, -1)

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
        index_padded = index_padded.reshape(grid_t, num_windows_h,
                                            vit_merger_window_size,
                                            num_windows_w,
                                            vit_merger_window_size)
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
            vit_merger_window_size)
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.to(dtype=torch.int32)
        cu_seqlens_tmp = torch.unique_consecutive(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(
            t, h, w)
        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)
        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.flatten(start_dim=0, end_dim=1)
        cu_seqlens_thw = torch.repeat_interleave(
            torch.tensor([h * w], dtype=torch.int32), t)
        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw,
                cu_seqlens_thw)

    def process_grid_thw(self, grid_thw):
        rotary_pos_emb = []
        window_index = []
        cu_window_seqlens = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens = []

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            cu_seqlens_window_thw = (cu_seqlens_window_thw +
                                     cu_window_seqlens_last)
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = torch.cat(rotary_pos_emb)
        window_index = torch.cat(window_index)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        return rotary_pos_emb, window_index, cu_window_seqlens, cu_seqlens

class Qwen2_5_VisionTransformer_Old(nn.Module):
    def __init__(
        self,
        hidden_size=1152,
        num_heads=16,
        window_size=32,
        patch_size=14,
        spatial_merge_size=2,
        fullatt_block_indexes=[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2
        
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device('cpu')

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
            index_padded = index_padded.reshape(grid_t, num_windows_h,
                                                vit_merger_window_size,
                                                num_windows_w,
                                                vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
                vit_merger_window_size)
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(
                0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[None, None]:
        return None, None

    def process_grid_thw(self, grid_thw_list):
        # Convert list to tensor for compatibility with old model
        grid_thw = torch.tensor(grid_thw_list, dtype=torch.int32)
        
        # Compute positional embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        
        # Compute window indices and seqlens
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, 
                                         device=window_index.device, 
                                         dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        
        # Compute sequence lengths
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                            grid_thw[:, 0]).cumsum(
                                                dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
        
        return rotary_pos_emb, window_index, cu_window_seqlens, cu_seqlens

def tensor_equals(t1, t2, name=None, rtol=1e-5, atol=1e-5):
    if t1.shape != t2.shape:
        if name:
            print(f"✗ {name} shapes differ: {t1.shape} vs {t2.shape}")
        return False
    
    equal = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    if not equal:
        # Find the positions where they differ
        diff_mask = ~torch.isclose(t1, t2, rtol=rtol, atol=atol)
        if diff_mask.sum() > 0:
            diff_pos = diff_mask.nonzero()
            first_diff = diff_pos[0].tolist()
            t1_val = t1[tuple(first_diff)]
            t2_val = t2[tuple(first_diff)]
            if name:
                print(f"✗ {name} values differ at {first_diff}: {t1_val} vs {t2_val}")
                print(f"Total number of different values: {diff_mask.sum().item()}/{t1.numel()}")
        else:
            if name:
                print(f"✗ {name} values differ but couldn't identify position")
            
        # Print some stats about the differences
        if name and t1.numel() < 100:
            print(f"Old: {t1.flatten().tolist()}")
            print(f"New: {t2.flatten().tolist()}")
        return False
    
    if name:
        print(f"✓ {name} matched")
    return True

def run_test(grid_thw, verbose=True):
    # Create models
    new_model = Qwen2_5_VisionTransformer_New()
    old_model = Qwen2_5_VisionTransformer_Old()
    
    if verbose:
        print("\nTesting with grid_thw:", grid_thw)
    
    # Test the new model
    rotary_pos_emb_new, window_index_new, cu_window_seqlens_new, cu_seqlens_new = new_model.process_grid_thw(grid_thw)
    
    if verbose:
        print("\nNew model outputs:")
        print(f"rotary_pos_emb shape: {rotary_pos_emb_new.shape}")
        print(f"window_index shape: {window_index_new.shape}")
        print(f"cu_window_seqlens shape: {cu_window_seqlens_new.shape}")
        print(f"cu_seqlens shape: {cu_seqlens_new.shape}")
    
    # Test the old model
    rotary_pos_emb_old, window_index_old, cu_window_seqlens_old, cu_seqlens_old = old_model.process_grid_thw(grid_thw)
    
    if verbose:
        print("\nOld model outputs:")
        print(f"rotary_pos_emb shape: {rotary_pos_emb_old.shape}")
        print(f"window_index shape: {window_index_old.shape}")
        print(f"cu_window_seqlens shape: {cu_window_seqlens_old.shape}")
        print(f"cu_seqlens shape: {cu_seqlens_old.shape}")
    
    # Compare outputs
    if verbose:
        print("\nComparing outputs:")
    match_rotary = tensor_equals(rotary_pos_emb_old, rotary_pos_emb_new, "rotary_pos_emb" if verbose else None)
    match_window = tensor_equals(window_index_old, window_index_new, "window_index" if verbose else None)
    match_cu_window = tensor_equals(cu_window_seqlens_old, cu_window_seqlens_new, "cu_window_seqlens" if verbose else None)
    match_cu_seq = tensor_equals(cu_seqlens_old, cu_seqlens_new, "cu_seqlens" if verbose else None)
    
    all_match = match_rotary and match_window and match_cu_window and match_cu_seq
    if verbose:
        print(f"\nAll outputs match: {all_match}")
    
    if not all_match:
        error_msg = f"Test failed for grid_thw={grid_thw}: Outputs between old and new implementations do not match"
        raise TestFailureException(error_msg)
        
    return all_match

def run_mass_test(t_range=(1, 50), h_range=(1, 250), w_range=(1, 250), 
                num_samples=100, max_images_per_sample=1, seed=42):
    """
    Run mass testing by sampling grid_thw configurations from the specified ranges.
    
    Args:
        t_range: Tuple of (min_t, max_t)
        h_range: Tuple of (min_h, max_h)
        w_range: Tuple of (min_w, max_w)
        num_samples: Number of random samples to test
        max_images_per_sample: Maximum number of images per sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Ensure minimum h and w values are at least 2 (spatial_merge_size)
    # This is required by the model architecture
    min_t = max(1, t_range[0])
    min_h = max(2, h_range[0])  # Minimum must be at least spatial_merge_size
    min_w = max(2, w_range[0])  # Minimum must be at least spatial_merge_size
    max_t = t_range[1]
    max_h = h_range[1]
    max_w = w_range[1]
    
    t_range = (min_t, max_t)
    h_range = (min_h, max_h)
    w_range = (min_w, max_w)
    
    print(f"Running mass testing with {num_samples} samples")
    print(f"T range: {t_range}")
    print(f"H range: {h_range}")
    print(f"W range: {w_range}")
    print(f"Max images per sample: {max_images_per_sample}")
    
    # Include edge cases
    edge_cases = [
        # Smallest valid values
        [[min_t, min_h, min_w]],
        # Largest values
        [[max_t, max_h, max_w]],
        # Min t, max h, w
        [[min_t, max_h, max_w]],
        # Max t, min h, w
        [[max_t, min_h, min_w]],
        # Mixed values
        [[min_t, max_h, min_w]],
        [[max_t, min_h, max_w]],
        # Values divisible by window_size/spatial_merge_size/patch_size
        [[min_t, 16, 16]],  # 16 = 32/2/1 (window_size/spatial_merge_size/1)
        [[min_t, 32, 32]],  # 32 = 32/2/0.5 (window_size/spatial_merge_size/0.5)
    ]
    
    # Add multi-image edge cases if max_images_per_sample > 1
    if max_images_per_sample > 1:
        multi_image_edge_cases = [
            # Multiple small images
            [[min_t, min_h, min_w], [min_t, min_h, min_w]],
            # One small, one large
            [[min_t, min_h, min_w], [max_t, max_h, max_w]],
            # Maximum number of images with varied sizes
            [[min_t, min_h, min_w]] * max_images_per_sample,
        ]
        edge_cases.extend(multi_image_edge_cases)
    
    # Test edge cases first
    print("\nTesting edge cases:")
    for i, grid_thw in enumerate(edge_cases):
        try:
            print(f"Edge case {i+1}/{len(edge_cases)}: {grid_thw}")
            run_test(grid_thw, verbose=False)
            print(f"✓ Edge case {i+1} passed")
        except TestFailureException as e:
            print(f"\nERROR: {e}")
            return False
        except Exception as e:
            print(f"\nUnexpected error for grid_thw={grid_thw}: {e}")
            print(f"Exception details: {type(e).__name__}: {e}")
            return False
    
    # Generate random samples for the mass test
    samples = []
    for _ in range(num_samples):
        # Decide how many images to include in this sample
        num_images = random.randint(1, max_images_per_sample)
        
        # Generate grid_thw for each image
        sample = []
        for _ in range(num_images):
            t = random.randint(min_t, max_t)
            h = random.randint(min_h, max_h)
            w = random.randint(min_h, max_w)
            # Ensure h and w are multiples of spatial_merge_size (2)
            h = (h // 2) * 2
            w = (w // 2) * 2
            if h == 0:
                h = 2
            if w == 0:
                w = 2
            sample.append([t, h, w])
        
        samples.append(sample)
    
    # Run the mass test with a progress bar
    print(f"\nRunning {num_samples} random samples:")
    progress_bar = tqdm.tqdm(total=num_samples)
    for i, grid_thw in enumerate(samples):
        try:
            run_test(grid_thw, verbose=False)
            progress_bar.update(1)
        except TestFailureException as e:
            progress_bar.close()
            print(f"\nERROR at sample {i+1}/{num_samples}: {e}")
            return False
        except Exception as e:
            progress_bar.close()
            print(f"\nUnexpected error at sample {i+1}/{num_samples} for grid_thw={grid_thw}: {e}")
            print(f"Exception details: {type(e).__name__}: {e}")
            return False
    
    progress_bar.close()
    print(f"\nAll {num_samples} samples passed successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Qwen2.5-VL Vision Transformer')
    parser.add_argument('--grid_t', type=int, default=1, help='Grid size T')
    parser.add_argument('--grid_h', type=int, default=36, help='Grid size H')
    parser.add_argument('--grid_w', type=int, default=36, help='Grid size W')
    parser.add_argument('--multiple', action='store_true', help='Test with multiple images')
    parser.add_argument('--large', action='store_true', help='Test with many high-resolution images')
    parser.add_argument('--mass-test', action='store_true', help='Run mass testing with many grid configurations')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for mass testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for mass testing')
    parser.add_argument('--max-t', type=int, default=50, help='Maximum T value for mass testing')
    parser.add_argument('--max-h', type=int, default=250, help='Maximum H value for mass testing')
    parser.add_argument('--max-w', type=int, default=250, help='Maximum W value for mass testing')
    parser.add_argument('--max-images', type=int, default=1, help='Maximum number of images per sample for mass testing')
    args = parser.parse_args()
    
    if args.mass_test:
        success = run_mass_test(
            t_range=(1, args.max_t),
            h_range=(1, args.max_h),
            w_range=(1, args.max_w),
            num_samples=args.samples,
            max_images_per_sample=args.max_images,
            seed=args.seed
        )
        sys.exit(0 if success else 1)
    else:
        if args.large:
            # Test with a large number of high-resolution images/videos
            grid_thw = [
                [1, 224, 224],  # High-res image 1
                [1, 112, 112],  # Medium-res image
                [4, 96, 96],    # Video 1
                [1, 168, 168],  # Another image
                [2, 128, 224],  # Video 2
                [1, 224, 224],  # High-res image 2
                [3, 64, 128],   # Video 3
                [1, 96, 96],    # Small image
                [6, 64, 64],    # Longer video
                [1, 192, 192]   # Another image
            ]
            print("Testing with large dataset (many high-resolution images/videos)")
        elif args.multiple:
            # Test with multiple images
            grid_thw = [
                [1, 36, 36],  # First image
                [2, 48, 64],  # Second image (video)
                [1, 24, 24]   # Third image
            ]
            print("Testing with multiple images")
        else:
            # Test with a single image
            grid_thw = [[args.grid_t, args.grid_h, args.grid_w]]
        
        try:
            # Run correctness test
            run_test(grid_thw)
            print("\nTest completed successfully!")
        except TestFailureException as e:
            print(f"\nERROR: {e}")
            sys.exit(1)  # Exit with error code 
```
</details>

---