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

torch.set_printoptions(threshold=10_000)

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

    from transformers import AutoProcessor

    siglip2_configs = {
        "debugmodel": Siglip2ModelArgs(
            dim=128,
            ffn_dim=256,
            n_layers=4,
            n_heads=2,
        ),
        "256M": Siglip2ModelArgs(
            dim=768,
            ffn_dim=2304,
            n_layers=12,
            n_heads=12,
        )
    }

    llama3_siglip2_configs = {
        "debugmodel": Llama3Siglip2ModelArgs(
            encoder=siglip2_configs["debugmodel"],
            dim=256,
            n_layers=6,
            n_heads=16,
            vocab_size=50000,
            rope_theta=500000,
        ),
        "256M": Llama3Siglip2ModelArgs(
            encoder=siglip2_configs["256M"],
            dim=576,
            n_layers=30,
            n_heads=9,
            n_kv_heads=3,
            ffn_dim_multiplier=1.3,
            multiple_of=1024,
            rope_theta=100000,
            vocab_size=49280,
        ),
    }

    args = llama3_siglip2_configs["256M"]

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

    for key, item in inputs.items():
        print(key)

    pixel_attention_mask = inputs['pixel_attention_mask']
    pixel_values = inputs['pixel_values'] 
    input_ids = inputs['input_ids']

    print(pixel_values.shape)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(
            tokens = input_ids,
            patch_attention_mask = pixel_attention_mask,
            pixel_values = pixel_values,
        )

    print(outputs)

