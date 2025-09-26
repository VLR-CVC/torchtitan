# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import einops as E
import torch
from torch import nn

from torchtitan.models.llama3 import Transformer as Llama3

from ..datasets.mm_datasets import SpecialTokens

from .args import Llama3Siglip2ModelArgs
from .siglip2 import VisionTransformer


def _scatter_img_tokens(h_BSD, tokens_BS, i_NLD, i_mask_NL, img_id):
    B, S, D = h_BSD.shape
    # Where are the image tokens in LLM input, make broadcastable with h_BSD
    img_mask_h_BSD = E.repeat(tokens_BS == img_id, "b s -> b s 1")
    # Only get valid (non-padded) tokens, result are flatten
    i_flatten = torch.masked_select(i_NLD, mask=i_mask_NL.unsqueeze(-1))

    assert i_flatten.numel() // D == img_mask_h_BSD.sum(), (
        f"Different number of visual embeddings {i_flatten.numel() // D} "
        f"with placeholder in input token embeddings {img_mask_h_BSD.sum()}"
    )
    h_BSD.masked_scatter_(mask=img_mask_h_BSD, source=i_flatten)
    return h_BSD


class Projector(nn.Module):
    """Project the Encoder embedding to the LLM embedding."""

class SmolVLMSimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: scale_factor to config
        input_size = config.encoder.dim * (config.encoder.scale_factor**2)
        output_size = config.dim
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.proj(x)


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.encoder.scale_factor
        self.modality_projection = SmolVLMSimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states

    def init_weights(self):
        self.modality_projection.init_weights()



class Llama3Siglip2Transformer(Llama3):
    def __init__(self, model_args: Llama3Siglip2ModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.encoder = VisionTransformer(model_args.encoder)
        self.projector = Projector(
            in_dim=model_args.encoder.dim, out_dim=model_args.dim
        )

    def init_weights(self, buffer_device=None):
        super().init_weights(buffer_device=buffer_device)
        if self.encoder is not None:
            self.encoder.init_weights()
        if self.projector is not None:
            self.projector.init_weights()

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        special_tokens: SpecialTokens,
        input_batch: torch.Tensor | None = None,
    ):
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        embed_tokens = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if self.encoder is not None:
            #grid_hw = grid_thw[:, :, 1:]  # Siglip2 only support image hw
            #pixel_masks = E.reduce(grid_hw != -1, "n l hw -> n l", reduction="all")
            i_NLD = self.encoder(pixel_values, patch_attention_mask)
            i_NLD = self.projector(i_NLD)
            h_BSD = _scatter_img_tokens(
                h_BSD, tokens, i_NLD, pixel_masks, special_tokens.img_id
            )

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis)

        hidden_states = self.norm(hidden_states)
        output = self.output(hidden_states)
        return output

if __name__ == "__main__":

    siglip2_configs = {
            "debugmodel": Siglip2ModelArgs(
                dim=128,
                ffn_dim=256,
                n_layers=4,
                n_heads=2,
                )
            }
    configs = {
            "256M": Llama3Siglip2ModelArgs(
                encoder=siglip2_configs["debugmodel"],
                dim=576,
                n_layers=30,
                n_heads=9,
                n_kv_heads=3,
                ffn_dim_multiplier=1.3,
                multiple_of=1024,
                rope_theta=500000,
                ),
            }


    args = configs["256M"]
    model = Llama3Siglip2Transformer(args)

    patch_size = 16

    # IMAGES
    img_size = 224
    pixel_values = torch.randn([1, 1, 3, img_size, img_size])
    batch_size, num_images, num_channels, height, width = pixel_values.shape
    # the num_images dim is dropped, increasing the batch size
    pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

    # Remove padding images - padding images are full 0.
    nb_values_per_image = pixel_values.shape[1:].numel()
    real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

    if not any(real_images_inds):
        # no images, leave one empty image.
        real_images_inds[0] = True

    pixel_values = pixel_values[real_images_inds].contiguous()

    pixel_attention_mask = None
    if pixel_attention_mask is None:
            size = [pixel_values.shape[i] for i in (0, 2, 3)]
            pixel_attention_mask = torch.ones(
                size=size,
                dtype=torch.bool,
                device=pixel_values.device,
            )
    else:
        # Remove padding images from the mask
        pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
        pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()
    patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
    patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
    patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

    image_tokens_seq = (torch.ones(int(14 * 14 / 4)) * torch.tensor(49190))

    seq1 = torch.tensor([    1, 11126,    42, 49189]).reshape(-1)
    seq2 = torch.tensor([49189,  7306,   346,  5125,   451, 2443,    47, 49279,   198,  9519,  9531,    42, 49279]).reshape(-1)
    input_ids = torch.cat([seq1, image_tokens_seq, seq2]).long()
    input_ids = torch.tensor(input_ids).reshape(1, -1)

    outputs = model(
        tokens = input_ids,
        patch_attention_mask = patch_attention_mask,
        pixel_values = pixel_values,
    )

    print(outputs)

