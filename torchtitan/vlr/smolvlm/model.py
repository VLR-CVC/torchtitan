# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForVision2Seq

from torchtitan.protocols.model import ModelProtocol

from .args import SmolVLMArgs


class SmolVLModel(AutoModelForVision2Seq, ModelProtocol):
    """
    This is a wrapper around the Hugging Face SmolVLM model that makes it
    compatible with torchtitan's training framework.
    """

    def __init__(self, model_args: SmolVLMArgs) -> None:
        # This will be initialized with the pre-trained model from Hugging Face
        # when the checkpoint is loaded.
        super().__init__(model_args)
