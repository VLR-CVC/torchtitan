# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import einops as E
import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.attention import build_attention, init_attention_mask

from .args import Siglip2ModelArgs



class SmolVLMVisionEmbeddings(nn.Module):
    """
    Vision embeddings with positional embeddings built-in.

    Taken from SmolVLM transformers implementation.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.n_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def init_weights(self):
        nn.init.trunc_normal_(self.patch_embedding.weight)
        nn.init.normal_(self.position_embedding.weight)

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, channels, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side, device=pixel_values.device
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0, device=pixel_values.device
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            h_indices = torch.arange(nb_patches_h, device=position_ids.device, dtype=pixel_values.dtype)
            w_indices = torch.arange(nb_patches_w, device=position_ids.device, dtype=pixel_values.dtype)

            fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
            fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of query heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, args: Siglip2ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

        self.attn = build_attention(
            use_flex_attn=False, attn_mask_type=args.attn_mask_type
        )

    def forward(self, x: torch.Tensor):
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Use self.head_dim instead of `n_heads` to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = E.rearrange(xq, "b l (h d) -> b h l d", d=self.head_dim)
        xk = E.rearrange(xk, "b l (h d) -> b h l d", d=self.head_dim)
        xv = E.rearrange(xv, "b l (h d) -> b h l d", d=self.head_dim)

        output = self.attn(xq, xk, xv)
        output = E.rearrange(output, "b h l d -> b l (h d)").contiguous()

        return self.out_proj(output)

    def init_weights(self):
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)


class FeedForward(nn.Module):
    def __init__(self, args: Siglip2ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.dim, args.ffn_dim)
        self.fc2 = nn.Linear(args.ffn_dim, args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x

    def init_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=0.02)


class TransformerLayer(nn.Module):
    def __init__(self, args: Siglip2ModelArgs):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)
        self.self_attn = Attention(args)
        self.layer_norm2 = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)
        self.mlp = FeedForward(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

    def init_weights(self):
        self.layer_norm1.reset_parameters()
        self.layer_norm2.reset_parameters()
        self.self_attn.init_weights()
        self.mlp.init_weights()


class VisionTransformer(nn.Module):
    def __init__(self, args: Siglip2ModelArgs):
        super().__init__()
        self.args = args
        self.eos_id = 11

        self.embeddings = SmolVLMVisionEmbeddings(args)
        self.layers = nn.ModuleDict(
            {str(idx): TransformerLayer(args) for idx in range(args.n_layers)}
        )
        self.post_layernorm = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
    ):
        h = self.embeddings(pixel_values, patch_attention_mask)

        for layer in self.layers.values():
            h = layer(h)
        h = self.post_layernorm(h)

        return h

    def init_weights(self):
        self.embeddings.init_weights()
        for layer in self.layers.values():
            layer.init_weights()
        self.post_layernorm.reset_parameters()
