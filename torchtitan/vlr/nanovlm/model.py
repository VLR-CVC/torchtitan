import json
import os
from dataclasses import asdict
from typing import Optional

import math  # TODO: take out

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from .utils import top_k_top_p_filtering
from .config import VLMConfig


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/modeling_idefics3.py#L107
class Idefics3VisionEmbeddings(nn.Module):
    """
    Allows for images of varying dims and aspect ratios.
    """

    def __init__(self, cfg):
        super().__init__()

        self.embed_dim = cfg.vit_hidden_dim
        self.image_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size

        self.conv = nn.Conv2d(
            in_channels=3,  # hardcoded
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    # TODO: what is this mask
    def forward(
        self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor
    ) -> torch.Tensor:
        bsz, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.conv(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # we get the num of patches for each side
        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
        )

        position_ids = torch.full(
            size=(bsz, max_nb_patches_h * max_nb_patches_w), fill_value=0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            h_indices = torch.arange(nb_patches_h, dtype=pixel_values.dtype)
            w_indices = torch.arange(nb_patches_w, dtype=pixel_values.dtype)

            # for stability????
            fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
            fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

            bucket_coords_h = torch.bucketize(
                fractional_coords_h, boundaries, right=True
            )
            bucket_coords_w = torch.bucketize(
                fractional_coords_w, boundaries, right=True
            )

            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim

        # Conv layer to extract the patches
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(
                torch.rand(1, self.num_patches + 1, self.embd_dim)
            )
        else:
            self.position_embedding = nn.Parameter(
                torch.rand(1, self.num_patches, self.embd_dim)
            )

    def forward(self, x):
        x = self.conv(x)  # extract patches
        x = x.flatten(2)  # flatten the patches into a single dimension
        x = x.transpose(1, 2)  # transpose to (batch_size, num_patches, hidden_dim)

        # Add CLS token (according to original ViT Paper) and position embeddings
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L381
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        assert self.embd_dim % self.n_heads == 0, (
            "embd_dim must be divisible by num_heads"
        )
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.vit_dropout

        # Combined projections for all heads
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)

        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.sdpa:
            print(
                "Warning: scaled dot product attention not available. Using standard attention in ViT."
            )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        # Reshape  [B, T, C] -> [B, T, n_heads, head_dim] and transpose -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, T, head_dim)

        if self.sdpa:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,  # ViT attention is bidirectional
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = (
                attn @ v
            )  # (B, n_heads, T, T) x (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)

        # Transpose back from [B, n_heads, T, head_dim] to [B, T, n_heads * head_dim] and combine all heads to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L453
class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# https://github.com/karpathy/nanoGPT/blob/master/model.py#L94
class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        if self.cls_flag:
            x = self.layer_norm(x[:, 0])
        else:
            x = self.layer_norm(x)
            # x = x.mean(dim=1)

        return x

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Vision Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        """
        TODO: move to another file
        """
        from transformers import SiglipVisionConfig
        from huggingface_hub import hf_hub_download
        import safetensors

        hf_config = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)
        cfg.vit_dropout = hf_config.attention_dropout
        cfg.vit_hidden_dim = hf_config.hidden_size
        cfg.vit_img_size = hf_config.image_size
        cfg.vit_inter_dim = hf_config.intermediate_size
        cfg.vit_ln_eps = hf_config.layer_norm_eps
        cfg.vit_n_heads = hf_config.num_attention_heads
        cfg.vit_n_blocks = hf_config.num_hidden_layers
        cfg.vit_patch_size = hf_config.patch_size
        model = cls(cfg)
        safetensors_file = hf_hub_download(
            repo_id=cfg.vit_model_type, filename="model.safetensors"
        )

        sd = model.state_dict()

        mapping = {
            "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "layer_norm.weight",
            "vision_model.post_layernorm.bias": "layer_norm.bias",
        }

        for i in range(cfg.vit_n_blocks):
            # Layer norms
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = (
                f"blocks.{i}.ln1.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = (
                f"blocks.{i}.ln1.bias"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = (
                f"blocks.{i}.ln2.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = (
                f"blocks.{i}.ln2.bias"
            )

            # MLP
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = (
                f"blocks.{i}.mlp.fc1.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = (
                f"blocks.{i}.mlp.fc1.bias"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = (
                f"blocks.{i}.mlp.fc2.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = (
                f"blocks.{i}.mlp.fc2.bias"
            )

            # Output projection
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = (
                f"blocks.{i}.attn.out_proj.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = (
                f"blocks.{i}.attn.out_proj.bias"
            )

        with safetensors.safe_open(
            filename=safetensors_file, framework="pt", device="cpu"
        ) as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    if tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        if "position_embedding" in hf_key:
                            sd[our_key].copy_(tensor.unsqueeze(0))
                        else:
                            print(
                                f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}"
                            )
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")

            # Manually handle QKV concatenation since our implementation combines Q, K, V into one
            for i in range(model.cfg.vit_n_blocks):
                q_weight = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
                )
                k_weight = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
                )
                v_weight = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
                )

                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(qkv_weight)

                q_bias = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
                )
                k_bias = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
                )
                v_bias = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
                )

                qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(qkv_bias)

        model.load_state_dict(sd)
        print(
            f"Successfully loaded {cfg.vit_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters."
        )
        return model


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    Args:
        cfg: A configuration object containing:
            - lm_hidden_dim (int): The dimensionality of the model hidden states.
            - lm_rms_eps (float): A small constant to avoid division by zero.
    """

    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, lm_hidden_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute inverse of RMS: square the tensor element-wise, mean is computed across lm_hidden_dim.
        irms = torch.rsqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        )  # inverse of RMS
        x = x * irms * self.weight

        return x


# Multiple derivates of Rotary Embeddings by now, this is a basic one with linear scaling to context length
# e.g. https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190
class RotaryEmbedding(nn.Module):
    """
    Compute Rotary Embedding to introduce positional dependency to input sequence without additional training parameters and
    relative distance of token position ids through angle rotation.

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): Hidden dimension size.
            - lm_n_heads (int): Number of attention heads.
            - lm_re_base (float): Base for rotary embedding frequencies.
            - lm_max_position_embeddings (int): Max sequence length supported for rotary embedding.
            - lm_attn_scaling (float): Attention scaling factor.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, (
            "Hidden dimension must be divisible by number of heads"
        )

        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads  # dim of each head
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape
                                  (batch_size, seq_len, dim), representing rotary embeddings.
        """

        batch_size, seq_len = position_ids.shape
        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq

        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()

        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)

        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)

        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1)

        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input by dividing the hidden dimension to two, then swapping and negating dimensions.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors in attention mechanisms.

    Rotary positional embeddings inject position-dependent rotations into query and key vectors,
    enabling transformers to encode positional information effectively without explicit positional encoding.

    Args:
        q (torch.Tensor): Query tensor with shape [batch_size, num_heads, seq_len, head_dim].
        k (torch.Tensor): Key tensor with shape [batch_size, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Precomputed cosine positional embeddings with shape [batch_size, seq_len, head_dim].
        sin (torch.Tensor): Precomputed sine positional embeddings with shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int, optional): Dimension index to unsqueeze `cos` and `sin` to enable broadcasting.
                                      Defaults to 1 (typically the heads dimension).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors (`q_embed`, `k_embed`),
                                           each with the same shape as the input tensors.

    How it works:
        - `cos` and `sin` tensors are unsqueezed at `unsqueeze_dim` to broadcast across attention heads.
        - Rotary embeddings apply a complex number rotation in the embedding space using:
            rotated = (original * cos) + (rotate_half(original) * sin)
        - `rotate_half` performs a specific half-dimension rotation on the input tensor.
        - This operation encodes relative position information in q and k without adding explicit positional vectors.

    Example:
        q_embed, k_embed = apply_rotary_pos_embd(q, k, cos, sin)

    """

    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]

    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as used in some transformer-based language models.

    GQA reduces computation by using fewer key-value heads than query heads,
    grouping multiple query heads to share the same key-value heads.

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Number of query heads.
            - lm_n_kv_heads (int): Number of key-value heads.
            - lm_hidden_dim (int): Hidden embedding dimension.
            - lm_dropout (float): Dropout rate.
    """

    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )
        assert self.embd_dim % self.n_heads == 0, (
            "embd_dim must be divisible by num_heads"
        )

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(
            self.embd_dim, self.head_dim * self.n_kv_heads, bias=False
        )
        self.v_proj = nn.Linear(
            self.embd_dim, self.head_dim * self.n_kv_heads, bias=False
        )
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.sdpa:
            print(
                "Warning: scaled dot product attention not available, using standard attention in LM."
            )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask=None,
        block_kv_cache=None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass for grouped query attention.

        Args:
            x (Tensor): Input tensor of shape (B, T_curr, C), where
                        B = batch size,
                        T_curr = current sequence length,
                        C = embedding dimension.
            cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
            sin (Tensor): Rotary embedding sines, shape compatible with q and k.
            attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                               with 1 for tokens to attend to and 0 for padding.
            block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.

        Returns:
            tuple[Tensor, dict]:
                - Output tensor after attention and projection, shape (B, T_curr, C).
                - Updated block_kv_cache dict for caching key-value states.
        """
        is_prefill = block_kv_cache is None

        B, T_curr, C = x.size()  # T_curr is the sequence length of the current input x

        q_curr = (
            self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B, n_heads, T_curr, head_dim)
        k_curr = (
            self.k_proj(x)
            .view(B, T_curr, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, n_kv_heads, T_curr, head_dim)
        v_curr = (
            self.v_proj(x)
            .view(B, T_curr, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, n_kv_heads, T_curr, head_dim)

        # Apply rotary embeddings to the current q and k
        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        # Check if we can use cached keys and values
        if not is_prefill and block_kv_cache["key"] is not None:
            # Concatenate with cached K, V
            # k_rotated and v_curr are for the new token(s)
            k = block_kv_cache["key"]
            v = block_kv_cache["value"]
            k = torch.cat([k, k_rotated], dim=2)
            v = torch.cat([v, v_curr], dim=2)
            block_kv_cache["key"] = k
            block_kv_cache["value"] = v
        else:
            # No cache, this is the first pass (prefill)
            k = k_rotated
            v = v_curr
            block_kv_cache = {"key": k, "value": v}

        # Repeat K, V for Grouped Query Attention
        k_exp = k.repeat_interleave(
            self.n_kv_groups, dim=1
        )  # (B, n_heads, T_kv, head_dim)
        v_exp = v.repeat_interleave(
            self.n_kv_groups, dim=1
        )  # (B, n_heads, T_kv, head_dim)

        T_kv = k_exp.size(2)  # Total sequence length of keys/values

        # Prepare attention mask for SDPA or manual path
        # attention_mask is (B, T_kv_total_length), 1 for attend, 0 for pad
        additive_attn_mask = None
        if attention_mask is not None:
            # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
            # Let's make it `[B, 1, 1, T_kv]` for SDPA.
            mask_for_keys = attention_mask[
                :, :T_kv
            ]  # Ensure mask matches key length [B, T_kv]
            additive_attn_mask = (
                1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()
            ) * torch.finfo(q.dtype).min
            # This additive_attn_mask shape is [B, 1, 1, T_kv]

        if self.sdpa and x.device.type != "mps":
            # During decode, no additional masking needed as [1, T_kv] is naturally causal
            is_causal = T_curr == T_kv and T_curr > 1
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k_exp,
                v_exp,
                attn_mask=additive_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Manual attention implementation
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )  # (B, n_heads, T_curr, T_kv)
            # During decode: no additional masking needed as [1, T_kv] is naturally causal
            if T_curr == T_kv and T_curr > 1:
                causal_mask_val = torch.tril(
                    torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)
                ).view(1, 1, T_curr, T_curr)
                attn = attn.masked_fill(~causal_mask_val, float("-inf"))

            if additive_attn_mask is not None:  # Additive padding mask
                # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, T_curr, T_kv]
                attn = attn + additive_attn_mask

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp

        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    """
    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embd_dim),
                    after gated MLP transformation.
        """
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x


# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg)  # Input Norm
        self.norm2 = RMSNorm(cfg)  # Post Attention Norm

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_kv_cache: Optional[dict] = None,
    ):
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape
                matching sequence length and head dimension.
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional): Key-value cache dict for cached keys and values
                during decoding. If None, no cache is used.

        Returns:
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache


# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList(
            [LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)]
        )
        self.norm = RMSNorm(cfg)  # Final Norm
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list[dict]] = None,
        start_pos: int = 0,
    ):
        """
        Performs a forward pass through the language model.

        Args:
            x (Tensor): Input tensor. If `lm_use_tokens` is True, this should be
                token indices with shape (batch_size, sequence_length).
                If False, it should be embeddings of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (Tensor, optional): Mask tensor for attention to
                specify which tokens to attend to, typically of shape
                (batch_size, sequence_length). Default is None.
            kv_cache (list[dict], optional): List of key-value caches for each transformer
                block to enable efficient autoregressive decoding.
                If None, no cache is used and new ones are created. Default is None.
            start_pos (int, optional): The starting position index for the current input
                sequence. Used to compute rotary positional embeddings correctly,
                especially for cached sequences during generation. Default is 0.

        Returns:
            Tuple:
                - Tensor: Output logits with shape (batch_size, sequence_length, vocab_size)
                if `lm_use_tokens` is True, otherwise the hidden state embeddings
                (batch_size, sequence_length, hidden_dim).
                - list: Updated list of key-value caches, one for each transformer block,
                useful for autoregressive decoding and incremental generation.

        Behavior:
            - If `lm_use_tokens` is True, the input token indices are first embedded.
            - Rotary positional embeddings are generated for the current input positions,
            which are passed along to each transformer block.
            - For each transformer block, the input is processed along with
            rotary embeddings, attention mask, and optional cached key-values.
            - After processing all blocks, a final RMS normalization is applied.
            - If tokens are used, the normalized hidden states are projected to logits
            over the vocabulary.
            - The method returns the logits or embeddings along with the updated
            cache for efficient decoding.
        """
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        # T_curr is the length of the current input sequence
        B, T_curr, _ = x.size()

        # Create position_ids for the current sequence based on start_pos
        current_position_ids = (
            torch.arange(start_pos, start_pos + T_curr, device=x.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        cos, sin = self.rotary_embd(
            current_position_ids
        )  # Get rotary position embeddings for current tokens

        # Initialize new KV cache if none provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])

        x = self.norm(x)

        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens:
            x = self.head(x)

        return x, kv_cache

    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int = 20):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list = self.forward(
            generated_outputs, attention_mask=None, kv_cache=None, start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)

            # The token being processed is `next_token`. Its position is `generated_outputs.size(1) - 1`.
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1:
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output,
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
            )
            last_output = decode_step_output[:, -1, :]

        return generated_outputs

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        """
        TODO: move to another file
        """
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
        import json
        from huggingface_hub.utils import EntryNotFoundError

        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)

        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")

        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, "lm_vocab_size"):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(
                    f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})"
                )
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")

        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers

        # Create our model with potentially larger vocabulary
        model = cls(cfg)

        try:
            index_path = hf_hub_download(
                repo_id=cfg.lm_model_type, filename="model.safetensors.index.json"
            )
            with open(index_path, "r") as f:
                index = json.load(f)
            # Get unique filenames from weight map
            safetensors_filenames = sorted(list(set(index["weight_map"].values())))
            # Download all the sharded files
            safetensors_files = [
                hf_hub_download(repo_id=cfg.lm_model_type, filename=fn)
                for fn in safetensors_filenames
            ]
        except EntryNotFoundError:
            safetensors_files = [
                hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")
            ]

        sd = model.state_dict()

        mapping = {
            "model.embed_tokens.weight": "token_embedding.weight",
            "model.norm.weight": "norm.weight",
        }

        for i in range(cfg.lm_n_blocks):
            layer_prefix = f"model.layers.{i}."
            block_prefix = f"blocks.{i}."

            mapping.update(
                {
                    f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                    f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                    f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                    f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                    f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                    f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                    f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                    f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                    f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight",
                }
            )

        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        loaded_keys = set()

        for safetensors_file in safetensors_files:
            with safetensors.safe_open(
                filename=safetensors_file, framework="pt", device="cpu"
            ) as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue

                    if hf_key in f.keys() and our_key in sd:
                        tensor = f.get_tensor(hf_key)

                        # Special handling for token embeddings if vocab sizes differ
                        if (
                            hf_key == "model.embed_tokens.weight"
                            and tensor.shape[0] != sd[our_key].shape[0]
                        ):
                            has_extended_embeddings = True
                            print(
                                f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}"
                            )

                            # Copy existing embeddings to the beginning of our larger embedding matrix
                            sd[our_key][: tensor.shape[0]].copy_(tensor)

                            # Initialize the new embeddings using the same approach as the original model
                            std = 0.02  # Common value, but you might want to adjust based on model
                            init.normal_(
                                sd[our_key][tensor.shape[0] :], mean=0.0, std=std
                            )

                            print(
                                f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings"
                            )
                            sd["head.weight"].copy_(
                                sd[our_key]
                            )  # Update the head weights as well
                        elif tensor.shape == sd[our_key].shape:
                            sd[our_key].copy_(tensor)
                        else:
                            print(
                                f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}"
                            )

                        loaded_keys.add(our_key)

        for hf_key, our_key in mapping.items():
            if our_key not in loaded_keys:
                if our_key in sd:
                    print(
                        f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})"
                    )

        # Load the state dict
        model.load_state_dict(sd)

        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, "head") and "head.weight" in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(
                    filename=safetensors_file, framework="pt", device="cpu"
                ) as f:
                    if "lm_head.weight" in f.keys():
                        lm_head = f.get_tensor("lm_head.weight")
                        if lm_head.shape[0] != sd["head.weight"].shape[0]:
                            print(
                                f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}"
                            )
                            # Copy existing weights
                            sd["head.weight"][: lm_head.shape[0]].copy_(lm_head)
                            # Initialize new weights
                            std = 0.02
                            init.normal_(
                                sd["head.weight"][lm_head.shape[0] :], mean=0.0, std=std
                            )
                            # Load updated weights
                            model.load_state_dict(sd)
                        lm_head_loaded = True
                        break

        # Handle weight tying (if needed)
        if (
            cfg.lm_tie_weights
            and hasattr(model, "head")
            and hasattr(model, "token_embedding")
        ):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")

        print(
            f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters."
        )
        return model


class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert (
            seq_root**2 == seq
        )  # Sequence length must be a perfect square for pixel shuffle
        assert (
            seq_root % self.scale_factor == 0
        )  # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(
            bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)

        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x


def create_pixel_attention_mask_vectorized(
    image_sizes: list[tuple[int, int]], device=None
) -> torch.Tensor:
    """
    To allow images of different dimns and aspect sized into the same batch, we need to add padding
    up to the maximum size found in each side (H and W).

    For that we need to create a mask to distinguish to pad patches from the real ones.
    """
    if not image_sizes:
        return torch.empty(0, 0, 0, dtype=torch.bool, device=device)

    batch_size = len(image_sizes)
    max_h = max(h for h, _ in image_sizes)
    max_w = max(w for _, w in image_sizes)

    heights = torch.tensor([h for h, w in image_sizes], device=device).view(
        batch_size, 1, 1
    )
    widths = torch.tensor([w for h, w in image_sizes], device=device).view(
        batch_size, 1, 1
    )

    h_range = torch.arange(max_h, device=device).view(1, max_h, 1)
    w_range = torch.arange(max_w, device=device).view(1, 1, max_w)

    h_mask = h_range < heights
    w_mask = w_range < widths

    return h_mask & w_mask


class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone

        # TODO: what to do with the tokenizer
        self.tokenizer = get_tokenizer(
            cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template
        )

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()

        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = input_ids == self.tokenizer.image_token_id
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(
            updated_token_embd.dtype
        )  # torch flattens before assigning

        return updated_token_embd

    """
    The images should come in as tensors [B, 3, H, W].
    """

    def _process_images(self, images, device):
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]

            if not images:  # Handle cases with no images
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        return images  # Already a tensor

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)  # [B, T_sequence, D_lm]

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(
                image_embd
            )  # [num_images, mp_image_token_length, D_lm]
            token_embd = self._replace_img_tokens_with_embd(
                input_ids, token_embd, image_embd
            )

        logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)  # Apply LM head
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        images,
        attention_mask=None,
        max_new_tokens=5,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False,
    ):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)  # [B, T_prompt_text, D_lm]

        if images_tensor is not None:
            # 1. Process image if present
            image_embd = self.vision_encoder(images_tensor)  # [B, T_img_feat, D_model]
            image_embd = self.MP(image_embd)  # [B, mp_image_token_length, D_lm]
            # 2. Combine image and text embeddings
            token_embd = self._replace_img_tokens_with_embd(
                input_ids, token_embd, image_embd
            )

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0)  # Or token_embd.size(0)

        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            token_embd,
            attention_mask=attention_mask,  # Use the provided attention mask
            kv_cache=None,
            start_pos=0,
        )

        last_token_output_from_prefill = prefill_output[:, -1, :]

        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill)
        else:
            current_logits = last_token_output_from_prefill

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(
                    current_logits, top_k=top_k, top_p=top_p
                )
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)

            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(
                next_token_id
            )  # [B, 1, D_lm]

            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ),
                    dim=1,
                )

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
            )

            last_token_output = decode_step_output[:, -1, :]

            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output

        if not newly_generated_ids_list:  # Handle case where max_new_tokens might be 0
            return torch.empty(
                (batch_size, 0), dtype=torch.long, device=input_ids.device
            )

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if (
            self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0
        ):  # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (
                generated_ids == self.tokenizer.eos_token_id
            )  # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(
                seq_len, device=device
            )  # Create column indices [0, 1, ..., seq_len-1]

            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(
                eos_mask,
                col_indices_for_min.unsqueeze(0).expand_as(generated_ids),
                seq_len + 1,
            )

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values

            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(
                first_eos_indices_values, max=seq_len
            )

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = (
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .expand_as(generated_ids)
            )

            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = (
                col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            )

            generated_ids[replace_mask] = self.tokenizer.eos_token_id

        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    """
    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )
    """
