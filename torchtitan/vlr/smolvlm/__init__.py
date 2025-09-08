# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer

from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.llama3.infra.pipeline import pipeline_llama
from torchtitan.protocols.train_spec import (
    TrainSpec,
    register_train_spec,
)

from .model import SmolVLModel
from .args import SmolVLMArgs

dim = SmolVLMArgs.hidden_size

smolvlm_configs = {
    "debugmodel": SmolVLMArgs(
        hidden_dim=256,
        num_layers=4,
        num_heads=2,
        vocab_size=10000,
    ),
}

register_train_spec(
    TrainSpec(
        name='smolvlm',
        model_cls=SmolVLModel,
        model_args=smolvlm_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
