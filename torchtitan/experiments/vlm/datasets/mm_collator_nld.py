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
    """Multimodal collator that works with image patches in NLD format.
    N: Number of images (vision encoder's batch size)
    L: Length of patches (vision encoder's sequence length)
    D: Dimension of a patch (3 * spatial_patch_size**2 * temporral patch_size)

    This module provides a collator class that handles both image and text data,
    converting images to patches and preparing text for model input.

    Example:
        >>> # Initialize collator
        >>> collator = MultiModalCollatorNLD(
        ...     batch_size=2,
        ...     seq_len=32,
        ...     max_images_per_batch=4,
        ...     max_patch_per_image=6,
        ...     patch_size=16,
        ...     padding_idx=0,
        ... )
        >>>
        >>> # Create sample batch
        >>> batch = [
        ...     {
        ...         "input_ids": torch.tensor([1, 2, 3]),
        ...         "labels": torch.tensor([2, 3, 4]),
        ...         "pixel_values": [
        ...             torch.randn(1, 32, 32, 3),
        ...             torch.randn(1, 32, 48, 3)
        ...         ]
        ...     },
        ...     {
        ...         "input_ids": torch.tensor([5, 6]),
        ...         "labels": torch.tensor([6, 7]),
        ...         "pixel_values": [
        ...             torch.randn(1, 32, 32, 3)   # One image
        ...         ]
        ...     }
        ... ]
        >>>
        >>> # Collate batch
        >>> outputs = collator(batch)
        >>>
        >>> # Examine outputs
        >>> print(outputs["input_ids"].shape)     # (2, 32)     - Padded to seq_len
        >>> print(outputs["labels"].shape)        # (2, 32)     - Padded to seq_len
        >>> print(outputs["pixel_values"].shape)  # (4, 6, 768) - (N=4 images, L=6 patches, D=16*16*3)
        >>> print(outputs["grid_thw"].shape)      # (4, 6, 3)   - Coordinates for each patch
        >>>
        >>> # The collated batch has:
        >>> # 1. Text tensors padded to max length
        >>> # 2. Images converted to patches in NLD format
        >>> # 3. Grid coordinates for each patch
        >>> # 4. All tensors properly batched and padded
    """

    batch_size: int  # LLM's batch size
    seq_len: int  # LLM's maximum sequence length

    patch_size: int  # Patch size for converting images to patches
    max_images_per_batch: int  # Vision Encoder's batch size
    max_patches_per_image: int  # Vision Encoder's sequence length

    padding_idx: int = 0
    ignore_idx: int = -100

    """
    def process_images(
        self, all_images: list[torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        Process a list of image tensors into patches with coordinate grids.

        Args:
            all_images: list of image tensors, each of shape (T, H, W, 3)
        if not all_images:
            return None, None

        pixel_values_list, grid_list = [], []
        for img in all_images:
            # Convert single image to patches
            #patches, grids = convert_to_patches(img, patch_size=self.patch_size)
            pixel_values, grids = get_grids(img, patch_size=self.patch_size)

            # Pad/truncate to max patches DEPRECATED
            #patches, grids = pad_patches(patches, grids, self.max_patches_per_image)

            pixel_values_list.append(pixel_values)
            grid_list.append(grids)

        # Stack all images
        pixel_values = torch.stack(pixel_values_list)
        grids = torch.stack(grid_list)
        # TODO: need for stack?

        # Pad to max_images_per_batch with empty images
        # DEPRECATED
        #patches, grids = pad_empty_images_to_target_batch_size(patches, grids, self.max_images_per_batch)

        return pixel_values, grids
    """

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

        images = [sample['images'] for sample in batch]
        # WHAT IS THIS INDEXING
        image_shapes = [tuple(img[0].shape[1:]) for img in images]

        pixel_attention_mask = create_pixel_attention_mask_vectorized(image_shapes)

        B, max_h, max_w = pixel_attention_mask.shape
        C = 3 # should be 3
        
        # Pad images in a channels-last format
        padded_pixels = torch.zeros(
            (B, C, max_h, max_w), dtype=images[0][0].dtype
        )
        for i, img in enumerate(images):
            h, w = image_shapes[i]
            padded_pixels[i, :, :h, :w] = img[0].squeeze(0)

        # Convert the pixel mask to the final patch mask
        patch_attention_mask = _pixel_to_patch_mask(
            pixel_attention_mask, self.patch_size
        )

        # Process text and pad to batch size
        input_ids, labels = self.process_text(batch)
        # RETURN PATCH ATTENTION MASK
        input_dict = {"input": input_ids, "pixel_values": padded_pixels, "patch_attention_mask": patch_attention_mask}

        return input_dict, labels
