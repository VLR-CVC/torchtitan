# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

logger = logging.getLogger()

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import Llama3Siglip2ModelArgs


class SmolVLMStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_args: Llama3Siglip2ModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {
            "lm_head.weight": "output.weight",

            "model.text_model.embed_tokens.weight": "tok_embeddings.weight", # check

            "model.text_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight", # check
            "model.text_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight", # check
            "model.text_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight", # check
            "model.text_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight", # check

            #"model.layers.{}.self_attn.rotary_emb.inv_freq": None,

            "model.text_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.gate_proj.weight", # check
            "model.text_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.up_proj.weight", # check
            "model.text_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.down_proj.weight", # check

            "model.text_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight", # check
            "model.text_model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight", # check

            "model.text_model.norm.weight": "norm.weight", # check

            "model.vision_model.embeddings.patch_embedding.weight": "encoder.embeddings.patch_embedding.weight",
            "model.vision_model.embeddings.patch_embedding.bias": "encoder.embeddings.patch_embedding.bias",

            "model.vision_model.embeddings.position_embedding.weight": "encoder.embeddings.position_embedding.weight",

            "model.vision_model.post_layernorm.weight": "encoder.post_layernorm.weight",
            "model.vision_model.post_layernorm.bias": "encoder.post_layernorm.bias",

            "model.vision_model.encoder.layers.{}.layer_norm1.weight": "encoder.layers.{}.layer_norm1.weight",
            "model.vision_model.encoder.layers.{}.layer_norm1.bias": "encoder.layers.{}.layer_norm1.bias",
            "model.vision_model.encoder.layers.{}.layer_norm2.weight": "encoder.layers.{}.layer_norm2.weight",
            "model.vision_model.encoder.layers.{}.layer_norm2.bias": "encoder.layers.{}.layer_norm2.bias",

            "model.vision_model.encoder.layers.{}.mlp.fc1.weight": "encoder.layers.{}.mlp.fc1.weight",
            "model.vision_model.encoder.layers.{}.mlp.fc1.bias": "encoder.layers.{}.mlp.fc1.bias",
            "model.vision_model.encoder.layers.{}.mlp.fc2.weight": "encoder.layers.{}.mlp.fc2.weight",
            "model.vision_model.encoder.layers.{}.mlp.fc2.bias": "encoder.layers.{}.mlp.fc2.bias",

            "model.vision_model.encoder.layers.{}.self_attn.k_proj.weight": "encoder.layers.{}.self_attn.k_proj.weight",
            "model.vision_model.encoder.layers.{}.self_attn.k_proj.bias": "encoder.layers.{}.self_attn.k_proj.bias",

            "model.vision_model.encoder.layers.{}.self_attn.out_proj.weight": "encoder.layers.{}.self_attn.out_proj.weight",
            "model.vision_model.encoder.layers.{}.self_attn.out_proj.bias": "encoder.layers.{}.self_attn.out_proj.bias",

            "model.vision_model.encoder.layers.{}.self_attn.q_proj.weight": "encoder.layers.{}.self_attn.q_proj.weight",
            "model.vision_model.encoder.layers.{}.self_attn.q_proj.bias": "encoder.layers.{}.self_attn.q_proj.bias",

            "model.vision_model.encoder.layers.{}.self_attn.v_proj.weight": "encoder.layers.{}.self_attn.v_proj.weight",
            "model.vision_model.encoder.layers.{}.self_attn.v_proj.bias": "encoder.layers.{}.self_attn.v_proj.bias",

            "model.connector.modality_projection.proj.weight": "projector.modality_projection.proj.weight",
        }

    # HuggingFace permutation function (exact copy from their conversion script)
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "layers.{}.attention.wq.weight":
                    value = self._permute(value, n_heads)
                if abstract_key == "layers.{}.attention.wk.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_args.dim
        head_dim = dim // n_heads
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]

            state_dict[new_key] = value
        return state_dict
