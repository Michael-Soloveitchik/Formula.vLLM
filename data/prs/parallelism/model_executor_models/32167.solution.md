# PR #32167: [Model] Re-implement Qwen3Omni Audio Encoder

- field: parallelism  | area: model_executor_models
- author: ywang96
- state: MERGED  created: 2026-01-12T10:44:32Z  merged: 2026-01-14T07:40:30Z
- url: https://github.com/vllm-project/vllm/pull/32167
- labels: ready, qwen

# Issue

## Problem

The Qwen3-Omni Audio Encoder was imported directly from HuggingFace transformers (`Qwen3OmniMoeAudioEncoder`) rather than implemented with vLLM primitives. This meant the audio encoder path did not use tensor-parallel linear layers, vLLM's attention backend selection, or vectorized preprocessing, and it hard-required `flash_attn` as an explicit dependency. The HF implementation was also slower at high batch sizes due to non-vectorized mask/chunk construction and per-item tensor creation.

## Example

Profiling with TP=1 on the thinker model: audio processing took 0.3812s with the HF encoder vs 0.3417s after re-implementation — roughly 10% faster at high batch sizes.

## Use case

Serving the Qwen3-Omni multimodal model with audio inputs, where the audio encoder runs on every audio item before the thinker language model.

## Resources

GPU compute and kernel-launch overhead (many small tensor allocations during mask/chunk construction), plus HBM pressure from the CNN downsampling path on large batches.

## Type of users

Users serving Qwen3-Omni with audio modality, especially those running tensor parallelism (TP>1) or on hardware without flash_attn where a vLLM-native attention backend is required.

## Kind of inputs

Batched audio mel-spectrogram features with variable feature lengths requiring chunked attention (`cu_seqlens`), particularly high batch sizes with many audio items.

## Mechanism with the issue

The encoder relied on the transformers `Qwen3OmniMoeAudioEncoder` with its own attention and un-sharded Linear layers, and the module forced `import flash_attn`. Preprocessing built per-chunk masks and tensors in loops, creating many small allocations, and attention could not route through vLLM's backend selector (`get_vit_attn_backend`).

# Solution

## Fix in words

The diff drops the HF `Qwen3OmniMoeAudioEncoder` import and the explicit `flash_attn` try/import, replacing them with vLLM-native modules: `SinusoidsPositionEmbedding`, `Qwen3OmniMoeAudioAttention` (QKV via `QKVParallelLinear`, output via `RowParallelLinear`, attention via `MMEncoderAttention`), `Qwen3OmniMoeAudioEncoderLayer` (TP-aware `ColumnParallelLinear`/`RowParallelLinear` MLP with fp16 clamping), and `Qwen3OmniMoeAudioEncoder` (3× Conv2d downsample → linear projection → transformer → MLP head). Attention now selects a backend via `get_vit_attn_backend` and computes `max_seqlen` only for flash-attn backends. Preprocessing is vectorized: chunk lengths computed with tensor ops, masks built via a single `arange` broadcast instead of per-item tensors, and the CNN path chunked by `conv_chunksize` to avoid OOM. Heads are split across TP ranks (`num_local_heads`), enabling tensor parallelism and ~10% speedup.

## Author's solution

Re-implement the Qwen3-Omni Audio Encoder using vLLM primitives (TP-aware linear layers, `MMEncoderAttention`, vLLM attention backend selection) with vectorization improvements in preprocessing and the CNN path, removing the explicit flash_attn dependency. Profiling showed roughly 10% speedup at high batch sizes with TP=1. Author notes two follow-ups: fixing a memory-profiling bug from `get_mm_max_tokens_per_item` inherited from Qwen2.5-VL, and enabling DP for audio and vision encoders.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/models/qwen3_omni_moe_thinker.py`

### Diff
```diff
diff --git a/vllm/model_executor/models/qwen3_omni_moe_thinker.py b/vllm/model_executor/models/qwen3_omni_moe_thinker.py
index d17ac6ce8cb3..0d13c3fe352b 100755
--- a/vllm/model_executor/models/qwen3_omni_moe_thinker.py
+++ b/vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -31,29 +31,34 @@
 import torch.nn as nn
 import torch.nn.functional as F
 from packaging.version import Version
-from transformers import PretrainedConfig
-from transformers import __version__ as TRANSFORMERS_VERSION
 from transformers.feature_extraction_utils import BatchFeature
 from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
+    Qwen3OmniMoeAudioEncoderConfig,
     Qwen3OmniMoeConfig,
     Qwen3OmniMoeThinkerConfig,
 )
-from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
-    Qwen3OmniMoeAudioEncoder,
-)
 from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
     Qwen3OmniMoeProcessor,
 )
 from transformers.models.whisper import WhisperFeatureExtractor
 
+# isort: off
+from transformers import PretrainedConfig
+from transformers import __version__ as TRANSFORMERS_VERSION
+# isort: on
+
 from vllm.compilation.decorators import support_torch_compile
 from vllm.config import MultiModalConfig, VllmConfig
-from vllm.distributed import get_pp_group
+from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
 from vllm.logger import init_logger
 from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
+from vllm.model_executor.layers.attention.mm_encoder_attention import (
+    MMEncoderAttention,
+)
 from vllm.model_executor.layers.conv import Conv3dLayer
 from vllm.model_executor.layers.linear import (
     ColumnParallelLinear,
+    QKVParallelLinear,
     RowParallelLinear,
 )
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
@@ -104,11 +109,6 @@
     get_vit_attn_backend,
 )
 
-try:
-    import flash_attn
-except (ImportError, ModuleNotFoundError):
-    flash_attn = None
-
 logger = init_logger(__name__)
 
 
@@ -121,6 +121,415 @@ def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
     return output_lengths
 
 
+# ============= Audio Encoder Components =============
+
+
+class SinusoidsPositionEmbedding(nn.Module):
+    """Sinusoidal position embedding for audio encoder."""
+
+    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
+        super().__init__()
+        self.length = length
+        self.channels = channels
+        self.max_timescale = max_timescale
+
+        if channels % 2 != 0:
+            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
+
+        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
+        inv_timescales = torch.exp(
+            -log_timescale_increment * torch.arange(channels // 2).float()
+        )
+        scaled_time = (
+            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
+        )
+        positional_embedding = torch.cat(
+            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
+        )
+        self.register_buffer(
+            "positional_embedding", positional_embedding, persistent=False
+        )
+
+    def forward(self, seqlen: int) -> torch.Tensor:
+        return self.positional_embedding[:seqlen, :]
+
+
+class Qwen3OmniMoeAudioAttention(nn.Module):
+    """Multi-headed attention for Qwen3-Omni Audio Encoder using MMEncoderAttention."""
+
+    def __init__(
+        self,
+        config: Qwen3OmniMoeAudioEncoderConfig,
+        multimodal_config: MultiModalConfig | None = None,
+        prefix: str = "",
+    ):
+        super().__init__()
+        self.embed_dim = config.d_model
+        self.num_heads = config.encoder_attention_heads
+        self.head_dim = self.embed_dim // self.num_heads
+        tp_size = get_tensor_model_parallel_world_size()
+        self.num_local_heads = self.num_heads // tp_size
+
+        if (self.head_dim * self.num_heads) != self.embed_dim:
+            raise ValueError(
+                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
+                f"{self.embed_dim} and `num_heads`: {self.num_heads})."
+            )
+
+        self.scaling = self.head_dim**-0.5
+
+        self.qkv = QKVParallelLinear(
+            hidden_size=self.embed_dim,
+            head_size=self.head_dim,
+            total_num_heads=self.num_heads,
+            total_num_kv_heads=self.num_heads,
+            bias=True,
+            prefix=f"{prefix}.qkv",
+        )
+
+        self.out_proj = RowParallelLinear(
+            input_size=self.embed_dim,
+            output_size=self.embed_dim,
+            bias=True,
+            prefix=f"{prefix}.out_proj",
+        )
+
+        self.attn = MMEncoderAttention(
+            num_heads=self.num_local_heads,
+            head_size=self.head_dim,
+            scale=self.scaling,
+            multimodal_config=multimodal_config,
+        )
+
+    def forward(
+        self,
+        hidden_states: torch.Tensor,
+        cu_seqlens: torch.Tensor,
+        max_seqlen: torch.Tensor | None,
+    ) -> torch.Tensor:
+        seq_length, _ = hidden_states.size()
+        qkv, _ = self.qkv(hidden_states)
+        q, k, v = qkv.chunk(3, dim=-1)
+        q = q.view(1, seq_length, -1, self.head_dim)
+        k = k.view(1, seq_length, -1, self.head_dim)
+        v = v.view(1, seq_length, -1, self.head_dim)
+
+        attn_output = self.attn(
+            query=q,
+            key=k,
+            value=v,
+            cu_seqlens=cu_seqlens,
+            max_seqlen=max_seqlen,
+        )
+
+        attn_output = attn_output.view(seq_length, -1)
+        output, _ = self.out_proj(attn_output)
+        return output
+
+
+class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
+    """Transformer encoder layer for Qwen3-Omni Audio Encoder."""
+
+    def __init__(
+        self,
+        config: Qwen3OmniMoeAudioEncoderConfig,
+        multimodal_config: MultiModalConfig | None = None,
+        prefix: str = "",
+    ):
+        super().__init__()
+        self.embed_dim = config.d_model
+        self.self_attn = Qwen3OmniMoeAudioAttention(
+            config, multimodal_config=multimodal_config, prefix=f"{prefix}.self_attn"
+        )
+        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
+        self.activation_fn = _ACTIVATION_REGISTRY[config.activation_function]
+        self.fc1 = ColumnParallelLinear(
+            self.embed_dim,
+            config.encoder_ffn_dim,
+            bias=True,
+            prefix=f"{prefix}.fc1",
+        )
+        self.fc2 = RowParallelLinear(
+            config.encoder_ffn_dim,
+            self.embed_dim,
+            bias=True,
+            prefix=f"{prefix}.fc2",
+        )
+        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
+
+    def forward(
+        self,
+        hidden_states: torch.Tensor,
+        cu_seqlens: torch.Tensor,
+        max_seqlen: torch.Tensor | None,
+    ) -> torch.Tensor:
+        """
+        Args:
+            hidden_states: Input tensor of shape (seq_len, hidden_size)
+            cu_seqlens: Cumulative sequence lengths
+            max_seqlen: Maximum sequence length in the batch
+        """
+        residual = hidden_states
+        hidden_states = self.self_attn_layer_norm(hidden_states)
+        hidden_states = self.self_attn(
+            hidden_states=hidden_states,
+            cu_seqlens=cu_seqlens,
+            max_seqlen=max_seqlen,
+        )
+        hidden_states = residual + hidden_states
+
+        residual = hidden_states
+        hidden_states = self.final_layer_norm(hidden_states)
+        hidden_states, _ = self.fc1(hidden_states)
+        hidden_states = self.activation_fn(hidden_states)
+        hidden_states, _ = self.fc2(hidden_states)
+        hidden_states = residual + hidden_states
+
+        # Clamp for numerical stability with fp16
+        if hidden_states.dtype == torch.float16:
+            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
+            hidden_states = torch.clamp(
+                hidden_states, min=-clamp_value, max=clamp_value
+            )
+
+        return hidden_states
+
+
+class Qwen3OmniMoeAudioEncoder(nn.Module):
+    """vLLM-native Qwen3-Omni Audio Encoder."""
+
+    def __init__(
+        self,
+        config: Qwen3OmniMoeAudioEncoderConfig,
+        multimodal_config: MultiModalConfig | None = None,
+        prefix: str = "",
+    ):
+        super().__init__()
+
+        embed_dim = config.d_model
+        self.num_mel_bins = config.num_mel_bins
+        self.max_source_positions = config.max_source_positions
+        self.n_window = config.n_window
+        self.n_window_infer = config.n_window_infer
+        self.conv_chunksize = config.conv_chunksize
+
+        # Position embedding
+        self.positional_embedding = SinusoidsPositionEmbedding(
+            self.max_source_positions, embed_dim
+        )
+
+        # Convolutional layers for mel-spectrogram processing
+        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
+        self.conv2d2 = nn.Conv2d(
+            config.downsample_hidden_size,
+            config.downsample_hidden_size,
+            3,
+            2,
+            padding=1,
+        )
+        self.conv2d3 = nn.Conv2d(
+            config.downsample_hidden_size,
+            config.downsample_hidden_size,
+            3,
+            2,
+            padding=1,
+        )
+
+        conv_out_dim = config.downsample_hidden_size * (
+            (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
+        )
+        self.conv_out = nn.Linear(conv_out_dim, config.d_model, bias=False)
+
+        # Transformer encoder layers
+        self.layers = nn.ModuleList(
+            [
+                Qwen3OmniMoeAudioEncoderLayer(
+                    config,
+                    multimodal_config=multimodal_config,
+                    prefix=f"{prefix}.layers.{i}",
+                )
+                for i in range(config.encoder_layers)
+            ]
+        )
+
+        # Output layers
+        self.ln_post = nn.LayerNorm(config.d_model)
+        self.proj1 = nn.Linear(config.d_model, config.d_model)
+        self.act = _ACTIVATION_REGISTRY[config.activation_function]
+        self.proj2 = nn.Linear(config.d_model, config.output_dim)
+
+        # Get attention backend
+        attn_backend_override = (
+            multimodal_config.mm_encoder_attn_backend
+            if multimodal_config is not None
+            else None
+        )
+        self.attn_backend = get_vit_attn_backend(
+            head_size=config.d_model // config.encoder_attention_heads,
+            dtype=torch.get_default_dtype(),
+            attn_backend_override=attn_backend_override,
+        )
+
+    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> torch.Tensor | None:
+        """Compute max_seqlen only for flash attention backends."""
+        max_seqlen = None
+        if self.attn_backend in {
+            AttentionBackendEnum.FLASH_ATTN,
+            AttentionBackendEnum.ROCM_AITER_FA,
+        }:
+            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
+        return max_seqlen
+
+    @property
+    def dtype(self) -> torch.dtype:
+        return self.conv2d1.weight.dtype
+
+    @property
+    def device(self) -> torch.device:
+        return self.conv2d1.weight.device
+
+    def forward(
+        self,
+        input_features: torch.Tensor,
+        feature_lens: torch.Tensor,
+        aftercnn_lens: torch.Tensor,
+    ):
+        # Compute chunk information
+        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
+
+        chunk_lengths = torch.tensor(
+            [self.n_window * 2] * chunk_num.sum(),
+            dtype=torch.long,
+            device=feature_lens.device,
+        )
+        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
+        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
+        chunk_lengths[chunk_lengths == 0] = self.n_window * 2
+
+        # Split input features into chunks and pad
+        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
+        padded_feature = nn.utils.rnn.pad_sequence(
+            chunk_list, batch_first=True
+        ).transpose(1, 2)
+
+        # Compute feature lengths after CNN
+        feature_lens_after_cnn = self._get_cnn_output_lengths(chunk_lengths)
+        # Vectorized mask creation: avoid creating many small tensors
+        max_len_after_cnn = feature_lens_after_cnn.max().item()
+        indices = torch.arange(max_len_after_cnn, device=padded_feature.device)
+        padded_mask_after_cnn = indices.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(
+            1
+        )
+
+        # Add channel dimension for conv2d
+        padded_feature = padded_feature.unsqueeze(1)
+
+        # Apply convolutional layers (chunk if needed to avoid OOM)
+        if padded_feature.size(0) <= self.conv_chunksize:
+            # Fast path: no chunking needed
+            padded_embed = F.gelu(self.conv2d1(padded_feature))
+            padded_embed = F.gelu(self.conv2d2(padded_embed))
+            padded_embed = F.gelu(self.conv2d3(padded_embed))
+        else:
+            # Chunked processing to avoid OOM
+            padded_embeds = []
+            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
+                padded_embed = F.gelu(self.conv2d1(chunk))
+                padded_embed = F.gelu(self.conv2d2(padded_embed))
+                padded_embed = F.gelu(self.conv2d3(padded_embed))
+                padded_embeds.append(padded_embed)
+            padded_embed = torch.cat(padded_embeds, dim=0)
+
+        # (batch, channels, freq, time) -> (batch, time, channels*freq)
+        b, c, f, t = padded_embed.size()
+        padded_embed = self.conv_out(
+            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
+        )
+
+        # Add positional embedding
+        positional_embedding = (
+            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
+            .unsqueeze(0)
+            .to(padded_embed.dtype)
+        )
+        padded_embed = padded_embed + positional_embedding
+
+        # Extract valid hidden states and compute cu_seqlens
+        hidden_states = padded_embed[padded_mask_after_cnn]
+
+        # Compute cumulative sequence lengths for chunked attention
+        cu_chunk_lens = [0]
+        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
+            self.n_window_infer // (self.n_window * 2)
+        )
+        # Use tolist() for efficient batch conversion from tensor to Python
+        for cnn_len in aftercnn_lens.tolist():
+            num_full_chunks = cnn_len // window_aftercnn
+            remainder = cnn_len % window_aftercnn
+            cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
+            if remainder:
+                cu_chunk_lens.append(remainder)
+        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(
+            -1, dtype=torch.int32
+        )
+
+        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
+
+        # Apply transformer layers
+        for encoder_layer in self.layers:
+            hidden_states = encoder_layer(
+                hidden_states,
+                cu_seqlens,
+                max_seqlen,
+            )
+
+        # Apply output layers
+        hidden_states = self.ln_post(hidden_states)
+        hidden_states = self.proj1(hidden_states)
+        hidden_states = self.act(hidden_states)
+        hidden_states = self.proj2(hidden_states)
+
+        return hidden_states
+
+    def _get_cnn_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
+        """Compute output lengths after the three conv2d layers."""
+        lengths = input_lengths
+        for _ in range(3):
+            lengths = (lengths - 1) // 2 + 1
+        return lengths
+
+    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        """Load weights with mapping from HuggingFace format."""
+        stacked_params_mapping = [
+            # (param_name, shard_name, shard_id)
+            ("self_attn.qkv.", "self_attn.q_proj.", "q"),
+            ("self_attn.qkv.", "self_attn.k_proj.", "k"),
+            ("self_attn.qkv.", "self_attn.v_proj.", "v"),
+        ]
+        params_dict = dict(self.named_parameters(remove_duplicate=False))
+        loaded_params: set[str] = set()
+
+        for name, loaded_weight in weights:
+            for param_name, weight_name, shard_id in stacked_params_mapping:
+                if weight_name not in name:
+                    continue
+                name = name.replace(weight_name, param_name)
+
+                param = params_dict[name]
+                weight_loader = param.weight_loader
+                weight_loader(param, loaded_weight, shard_id)
+                break
+            else:
+                param = params_dict.get(name)
+                if param is not None:
+                    weight_loader = getattr(
+                        param, "weight_loader", default_weight_loader
+                    )
+                    weight_loader(param, loaded_weight)
+            loaded_params.add(name)
+        return loaded_params
+
+
 class Qwen3_VisionPatchEmbed(nn.Module):
     def __init__(
         self,
@@ -144,7 +553,7 @@ def __init__(
         )
 
     def forward(self, x: torch.Tensor) -> torch.Tensor:
-        L, C = x.shape
+        L, _ = x.shape
         x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
         x = self.proj(x).view(L, self.hidden_size)
         return x
@@ -224,7 +633,7 @@ def forward(
         cu_seqlens: torch.Tensor,
         rotary_pos_emb_cos: torch.Tensor,
         rotary_pos_emb_sin: torch.Tensor,
-        max_seqlen: torch.Tensor,  # Only used for Flash Attention
+        max_seqlen: torch.Tensor | None,  # Only used for Flash Attention
     ) -> torch.Tensor:
         x = x + self.attn(
             self.norm1(x),
@@ -1142,12 +1551,11 @@ def _process_audio_input(
 
         audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)
 
-        audio_outputs = self.audio_tower(
+        audio_features = self.audio_tower(
             input_features.to(self.audio_tower.dtype),
             feature_lens=audio_feature_lengths,
             aftercnn_lens=audio_output_lengths,
         )
-        audio_features = audio_outputs.last_hidden_state
         return audio_features.split(audio_output_lengths.tolist())
 
 
@@ -1205,20 +1613,11 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
         self.config = thinker_config
         self.multimodal_config = multimodal_config
 
-        # force "use_flash_attention_2=True" to audio tower to align
-        # the results.
-        if flash_attn is not None:
-            audio_config = thinker_config.audio_config
-            audio_config._attn_implementation_autoset = True
-            audio_config._attn_implementation = "flash_attention_2"
-        else:
-            logger.warning(
-                "flash_attn is not available, the model may not yield the "
-                "exactly same result as the transformers implementation "
-                "in the audio tower part."
-            )
-
-        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)
+        self.audio_tower = Qwen3OmniMoeAudioEncoder(
+            thinker_config.audio_config,
+            multimodal_config=multimodal_config,
+            prefix=maybe_prefix(prefix, "audio_tower"),
+        )
 
         self.visual = Qwen3Omni_VisionTransformer(
             vision_config=thinker_config.vision_config,

```
