# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torchtitan.tools.logging import logger

from .text_utils import pad_input_ids_and_labels_to_target_batch_size, pad_text_batch


def create_pixel_attention_mask_vectorized(
    image_sizes: list[tuple[int, int]], device=None
) -> torch.Tensor:
    if not image_sizes:
        return torch.empty(0, 0, 0, dtype=torch.bool, device=device)

    batch_size = len(image_sizes)
    max_h = max(h for h, w in image_sizes)
    max_w = max(w for h, w in image_sizes)

    heights = torch.tensor([h for h, _ in image_sizes], device=device).view(batch_size, 1, 1)
    widths = torch.tensor([w for _, w in image_sizes], device=device).view(batch_size, 1, 1)

    h_range = torch.arange(max_h, device=device).view(1, max_h, 1)
    w_range = torch.arange(max_w, device=device).view(1, 1, max_w)

    h_mask = h_range < heights
    w_mask = w_range < widths

    return h_mask & w_mask

# WHAT IS THIS
def _pixel_to_patch_mask(
    pixel_mask: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """Converts a pixel attention mask to a patch attention mask."""
    # Add a channel dimension for pooling
    pixel_mask_float = pixel_mask.unsqueeze(1).float()
    
    # Use average pooling to check if a patch contains any unmasked pixels
    # A patch is considered valid if its average value is > 0
    patch_mask = F.avg_pool2d(
        pixel_mask_float,
        kernel_size=patch_size,
        stride=patch_size
    ) > 0
    
    # Remove the channel dimension
    return patch_mask.squeeze(1)

@dataclass
class MultiModalCollatorNLD:
    batch_size: int  # LLM's batch size
    seq_len: int  # LLM's maximum sequence length

    patch_size: int  # Patch size for converting images to patches
    max_images_per_batch: int  # Vision Encoder's batch size
    max_patches_per_image: int  # Vision Encoder's sequence length

    padding_idx: int = 0
    ignore_idx: int = -100

    def process_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process text inputs and labels from batch.

        Args:
            batch: list of dictionaries containing "input_ids" and "labels"

        Returns:
            input_ids: Tensor of shape (B, L)
            labels: Tensor of shape (B, L)

        Note:
            B = batch size (padded if needed)
            L = sequence length (padded/truncated to seq_len)
        """
        # Pad sequences in batch
        input_ids = pad_sequence(
            [s["input_ids"] for s in batch],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        labels = pad_sequence(
            [s["labels"] for s in batch],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        # Handle sequence length
        input_ids, labels = pad_text_batch(
            input_ids,
            labels,
            self.seq_len + 1,  # Extra token for label shifting
            padding_idx=self.padding_idx,
            ignore_idx=self.ignore_idx,
        )
        input_ids, labels = pad_input_ids_and_labels_to_target_batch_size(
            input_ids,
            labels,
            self.batch_size,
            padding_idx=self.padding_idx,
            ignore_idx=self.ignore_idx,
        )

        return input_ids[:, :-1], labels[:, 1:]  # Shift for next token prediction

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """Encode batch with patch-based approach.

        Args:
            batch: list of dictionaries containing:
                - input_ids: Tensor of shape (S)
                - labels: Tensor of shape (L)
                - images: list of tensors, each (1, 3, H, W)
        """

        pixel_values = [sample['pixel_values'] for sample in batch]
        patch_attention_mask = [sample['patch_attention_mask'] for sample in batch]

        try:
            patch_attention_mask = torch.tensor(patch_attention_mask).squeeze()
        except Exception:
            print(len(patch_attention_mask))
            print(len(patch_attention_mask[0]))
            print(len(patch_attention_mask[0][0]))

        padded_pixels = torch.stack(pixel_values, dim=0)

        input_ids, labels = self.process_text(batch)
        input_dict = {"input": input_ids, "pixel_values": padded_pixels, "patch_attention_mask": patch_attention_mask}

        return input_dict, labels
