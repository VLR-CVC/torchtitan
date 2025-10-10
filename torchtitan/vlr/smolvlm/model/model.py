# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import einops as E
import torch
from torch import nn

from torchtitan.models.attention import init_attention_mask
from torchtitan.models.llama3 import Transformer as Llama3

from .args import Llama3Siglip2ModelArgs, Siglip2ModelArgs
from .siglip2 import VisionTransformer

import lovely_tensors as lt
lt.monkey_patch()

class SmolVLMSimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: scale_factor to config
        input_size = 12288
        output_size = config.dim
        self.proj = nn.Linear(12288, 576, bias=False)

    def init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.proj(x)


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.encoder.scale_factor
        self.modality_projection = SmolVLMSimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=4):
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
        print("image hidden")
        print(image_hidden_states)
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
        self.projector = Projector(model_args)
        self.n_pixels_per_token = model_args.encoder.patch_size**2
        self.init_encoder_weights()

        self.IMAGE_TOKEN_ID = 49190

    def init_encoder_weights(self, buffer_device=None):
        super().init_weights(buffer_device=buffer_device)
        if self.encoder is not None:
            self.encoder.init_weights()
        if self.projector is not None:
            self.projector.init_weights()

    def _fuse_vision_text(self, inputs_embeds, image_hidden_states, input_ids):
        _, patch_size, _ = image_hidden_states.shape
        image_token_id = self.IMAGE_TOKEN_ID

        image_mask = input_ids == image_token_id

        num_image_tokens = image_mask.sum(dim=1)
        """
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("At least one sample has <image> tokens not divisible by patch_size.")
        """

        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds).bfloat16()
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
        return merged_embeds

    def get_image_features(
            self,
            pixel_values,
            pixel_attention_mask
    ):
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.bfloat16()  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        if not any(real_images_inds):
            # no images, leave one empty image.
            real_images_inds[0] = True

        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = 16
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        image_hidden_states = self.encoder(pixel_values, patch_attention_mask)
        #image_hidden_states = image_hidden_states.last_hidden_state
        image_hidden_states = image_hidden_states.bfloat16()

        image_hidden_states = self.projector(image_hidden_states)
        return image_hidden_states

    def forward(
            self,
            input_ids: torch.Tensor,
            eos_id: int | None = None,
            input_batch: torch.Tensor | None = None,
            pixel_values: torch.Tensor | None = None,
            patch_attention_mask: torch.BoolTensor | None = None,
            #grid_thw: torch.Tensor | None = None,
            ):
        if self.model_args.use_flex_attn:
            init_attention_mask(
                    input_batch if input_batch is not None else input_ids, eos_id=self.eos_id
                    )

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        hidden_states = self.tok_embeddings(input_ids) if self.tok_embeddings else input_ids

        """
        if self.encoder is not None and pixel_values is not None:
            vision_tokens = self.get_image_features(pixel_values, patch_attention_mask)
            hidden_states = self._fuse_vision_text(hidden_states, vision_tokens, input_ids)
        else:
            "THERE are not images"
        """

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis)

        hidden_states = self.norm(hidden_states)
        output = self.output(hidden_states)
        return output

if __name__ == "__main__":

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-256M-Video-Instruct')

    device = torch.device('cuda:4')

    siglip2_configs = {
            "debugmodel": Siglip2ModelArgs(
                dim=128,
                ffn_dim=256,
                n_layers=4,
                n_heads=2,
                ),
            "256M": Siglip2ModelArgs(
                dim=768,
                ffn_dim=3072,
                n_layers=12,
                n_heads=12,
                )
            }
    configs = {
            "256M": Llama3Siglip2ModelArgs(
                encoder=siglip2_configs["256M"],
                dim=576,
                n_layers=30,
                n_heads=9,
                n_kv_heads=3,
                ffn_dim=1536,
                use_flex_attn = False,
                attn_mask_type = "causal",
                ),
            }


    args = configs["256M"]

    device = torch.device("cuda:4")
    model = Llama3Siglip2Transformer(args).to(device)

    print(model)

    import numpy as np
    import requests
    from PIL import Image
    from io import BytesIO

    model_id = 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'
    processor = AutoProcessor.from_pretrained(model_id)

    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    image = Image.open(BytesIO(requests.get(image_url).content)).resize((512, 512))

    image = np.array(image)

    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Can you describe this image?"},            
        ]
    },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=torch.bfloat16)

    pixel_attention_mask = inputs['pixel_attention_mask']
    pixel_values = inputs['pixel_values'] 
    input_ids = inputs['input_ids']

    print(pixel_values.plt)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(
            input_ids = input_ids,
            patch_attention_mask = pixel_attention_mask,
            pixel_values = pixel_values,
        )

    print(outputs.plt)
