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


def resize_positional_embeddings(
    pos_embs_HWD: torch.Tensor,
    spatial_shapes_N2: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Resize the learned 2D positional embeddings to image-specific size and pad to a fixed size.

    Args:
        pos_embs_HWD (`torch.Tensor`):
            Position embeddings of shape (height, width, embed_dim)
        spatial_shapes (`torch.LongTensor`):
            Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        max_length (`int`):
            Maximum length of the positional embeddings to pad resized positional embeddings to

    Returns:
        `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
    """
    _, _, D = pos_embs_HWD.shape
    B, _ = spatial_shapes_N2.shape

    resized_embs_BLD = torch.empty(
        (B, max_length, D),
        device=pos_embs_HWD.device,
        dtype=pos_embs_HWD.dtype,
    )

    # TODO: group images by size, and do interpolate,
    # or cache the interpolate output so we do this once per size
    for i in range(B):
        height, width = spatial_shapes_N2[i].tolist()
        if (height + width) == 0:  # Skip empty padding images
            continue

        resized_emb = F.interpolate(
            E.rearrange(pos_embs_HWD, "h w d -> 1 d h w"),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        resized_emb_LD = E.rearrange(resized_emb, "1 d h w -> (h w) d")
        resized_embs_BLD[i, : int(height * width)] = resized_emb_LD

    return resized_embs_BLD


class VisionEmbeddings(nn.Module):
    """
    DEPRECATED
    """
    def __init__(self, args: Siglip2ModelArgs):
        super().__init__()
        self.patch_embedding = nn.Linear(
            in_features=args.n_channels * args.patch_size * args.patch_size,
            out_features=args.dim,
        )
        self.position_embedding = nn.Embedding(args.n_pos_embs**2, args.dim)
        self.n_pos_embs = args.n_pos_embs

    def init_weights(self):
        nn.init.trunc_normal_(self.patch_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight)

    def forward(self, pixels_NLD: torch.Tensor, grid_hw: torch.Tensor) -> torch.Tensor:
        # Apply patch embeddings to already patchified pixel values
        patch_embeds_NLD = self.patch_embedding(pixels_NLD)

        # Get positional resized and padded positional embeddings
        pos_emb_HWD = self.position_embedding.weight.reshape(
            self.n_pos_embs, self.n_pos_embs, -1
        )
        spatial_h = E.reduce(grid_hw[:, :, 0], "n l -> n", reduction="max") + 1
        spatial_w = E.reduce(grid_hw[:, :, 1], "n l -> n", reduction="max") + 1
        spatial_shapes = torch.stack([spatial_h, spatial_w], dim=-1).long()
        resized_positional_embeddings = resize_positional_embeddings(
            pos_emb_HWD,
            spatial_shapes,
            max_length=pixels_NLD.shape[1],
        )
        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds_NLD + resized_positional_embeddings
        return embeddings

class SmolVLMVisionEmbeddings(nn.Module):
    """
    Vision embeddings with positional embeddings built-in.

    Taken from SmolVLM transformers implementation.
    """
    def __init__(self):
        self.embed_dim = 1152
        self.image_size = 224
        self.patch_size = 16
        self.num_channels = 3

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

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

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
            use_flex_attn=True, attn_mask_type=args.attn_mask_type
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

        self.embeddings = VisionEmbeddings(args)
        self.layers = nn.ModuleDict(
            {str(idx): TransformerLayer(args) for idx in range(args.n_layers)}
        )
        self.post_layernorm = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)

    def forward(
        self,
        pixel_values_NLD: torch.FloatTensor,
        pixel_masks_NL: torch.BoolTensor,
        grid_hw: torch.LongTensor,
    ):
        init_attention_mask(pixel_masks_NL, eos_id=self.eos_id)

        h = self.embeddings(pixel_values_NLD, grid_hw)

        for layer in self.layers.values():
            h = layer(h)
        h = self.post_layernorm(h)

        return h

    def init_weights(self):
        self.embeddings.init_weights()
        for layer in self.layers.values():
            layer.init_weights()
        self.post_layernorm.reset_parameters()
