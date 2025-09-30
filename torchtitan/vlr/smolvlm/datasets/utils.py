# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for image processing in multimodal datasets."""

import math
from io import BytesIO

import einops as E
import numpy as np
import requests
import torch
import torchvision

from typing import Optional, Set, Union
from collections import defaultdict

from PIL import Image

from torchtitan.tools.logging import logger

from torchvision.transforms.v2 import functional as F
import re
from urllib import request


def process_image(
    image: str | bytes | Image.Image,
    patch_size: int = 16,
    merge_size: int = 1,
    max_patch_per_image: int = 256,
    min_patch_per_image: int = 1,
) -> torch.Tensor | None:
    """Process a single image into normalized tensor format.

    Args:
        image: PIL Image, bytes, or URL string
        patch_size: Size of each patch
        merge_size: Spatial Merge size factor
        max_patch_per_image: Maximum patches allowed per image
        min_dimension: Minimum dimension for width/height

    Returns:
        Tensor of shape (1, H, W, 3) or None if processing fails

    Note:
        - Resizes image while maintaining aspect ratio
        - Normalizes using CLIP mean/std values
        - Returns None if any processing step fails
    """
    try:
        # Convert various input formats to PIL Image
        if isinstance(image, str) and image.startswith("http"):
            response = requests.get(image, timeout=10)
            image = Image.open(BytesIO(response.content))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize maintaining aspect ratio
        image = resize_image_by_patch_count(
            image,
            max_patch_per_image=max_patch_per_image,
            patch_size=patch_size,
            merge_size=merge_size,
            min_patch_per_image=min_patch_per_image,
        )

        # Convert to numpy and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0

        # CLIP normalization
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img_array = (img_array - mean) / std

        # Convert to tensor (1, H, W, 3) with dummy temporal dim
        return torch.from_numpy(img_array).float().unsqueeze(0)

    except Exception as e:
        logger.warning(f"Error processing image: {e}")
        return None


def smart_resize(
    height: int,
    width: int,
    factor: int,  # should be equal patch_size * merge_size
    max_patch_per_image: int,
    min_patch_per_image: int = 1,
):
    """Calculate dimensions that maintain aspect ratio and satisfy constraints."""
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} and width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # Calculate patch count from adjusted dimensions
    current_patches = (h_bar * w_bar) // (factor * factor)

    if current_patches > max_patch_per_image:
        # Scale down to fit within max patch limit
        max_area = max_patch_per_image * (factor * factor)
        beta = math.sqrt((h_bar * w_bar) / max_area)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif current_patches < min_patch_per_image:
        beta = math.sqrt(min_patch_per_image / current_patches)
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def resize_image_by_patch_count(
    image: Image.Image,
    max_patch_per_image: int,
    patch_size: int,
    merge_size: int,
    min_patch_per_image: int = 1,
) -> Image.Image:
    """Resize image while maintaining aspect ratio and ensuring patch count is within [min_patch_per_image, max_patch_per_image]."""
    original_width, original_height = image.size
    factor = patch_size * merge_size

    # Calculate current number of patches
    current_patches = (original_height * original_width) // (factor * factor)

    # If patches < min_patch_per_image, scale up proportionally
    if current_patches < min_patch_per_image:
        if current_patches == 0:
            # Special case: image too small to produce any patches
            # Scale to minimum viable size (at least factor x factor)
            scale_factor = max(factor / original_width, factor / original_height)
        else:
            scale_factor = math.sqrt(min_patch_per_image / current_patches)

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_height, resized_width = smart_resize(
            new_height,
            new_width,
            factor,
            max_patch_per_image,
        )
        return image.resize((resized_width, resized_height))

    # If patches are within [min, max] range, just use smart_resize
    elif current_patches <= max_patch_per_image:
        resized_height, resized_width = smart_resize(
            original_height, original_width, factor, max_patch_per_image
        )
        return image.resize((resized_width, resized_height))

    # If patches > max_patch_per_image, scale down proportionally
    else:
        scale_factor = math.sqrt(max_patch_per_image / current_patches)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_height, resized_width = smart_resize(
            new_height, new_width, factor, max_patch_per_image
        )
        return image.resize((resized_width, resized_height))


def calculate_image_tokens(
    image: Image.Image | torch.Tensor,
    patch_size: int,
    spatial_merge_size: int,
) -> tuple[int, int, int]:
    """Calculate number of tokens needed for an image."""
    if isinstance(image, torch.Tensor):
        height, width = image.shape[1:3]
    else:
        width, height = image.size

    tokens_per_row = int(width / (patch_size * spatial_merge_size))
    num_rows = int(height / (patch_size * spatial_merge_size))
    total_tokens = tokens_per_row * num_rows

    return total_tokens, tokens_per_row, num_rows


def convert_to_patches(
    pixel_values: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert single image tensor to patches and generate coordinate grids.

    Args:
        pixel_values: Tensor of shape (T, H, W, C)
        patch_size: Spatial patch size (height and width)
        temporal_patch_size: Temporal patch size (default=1 for no temporal patching)

    Returns:
        patches: Tensor of shape (L, D) where:
            L = (T//temporal_patch_size) * (H//patch_size) * (W//patch_size)
            D = temporal_patch_size * patch_size * patch_size * C
        grid: Tensor of shape (L, 3) containing (t, h, w) coordinates

    Example:
        >>> x = torch.randn(4, 224, 224, 3)  # Single image with 4 frames
        >>> patches, grid = convert_to_patches(x, patch_size=14, temporal_patch_size=2)
        >>> print(patches.shape)  # (512, 1176)  # 512 patches, each 1176-dim
        >>> print(grid.shape)     # (512, 3)     # (t,h,w) coordinates
    """
    T, H, W, C = pixel_values.shape
    ps = patch_size
    ts = temporal_patch_size
    device = pixel_values.device

    # Ensure dimensions are divisible
    if T % ts != 0:
        raise ValueError(
            f"Temporal dimension {T} must be divisible by temporal_patch_size {ts}"
        )
    if H % ps != 0 or W % ps != 0:
        raise ValueError(
            f"Spatial dimensions {H},{W} must be divisible by patch_size {ps}"
        )

    patches = E.rearrange(
        pixel_values,
        "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)",
        pt=ts,
        ph=ps,
        pw=ps,
    )

    # Generate coordinate grid
    coords = torch.meshgrid(
        torch.arange(T // ts, device=device),
        torch.arange(H // ps, device=device),
        torch.arange(W // ps, device=device),
        indexing="ij",
    )
    grid = E.rearrange(torch.stack(coords), "coords t h w -> (t h w) coords")  # (L, 3)

    return patches, grid


def get_grids(
    pixel_values: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    T, H, W, C = pixel_values.shape
    device = pixel_values.device

    coords = torch.meshgrid(
        torch.arange(T // temporal_patch_size, device=device),
        torch.arange(H // patch_size, device=device),
        torch.arange(W // patch_size, device=device),
        indexing="ij",
    )
    grid = E.rearrange(torch.stack(coords), "coords t h w -> (t h w) coords")
    return pixel_values, grid


def pad_patches(
    patches: torch.Tensor,
    grids: torch.Tensor,
    max_patches: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pad or truncate patches and grids to max_patches length for single image.
    Args:
        patches: Image patches of shape SeqLen x Dim [L,D]
        grids: corresponding patch coordinates in 3D grid from top-left
            with shape [L, 3] for temporal and spatial dimension t,h,w.
            Grid of all -1 indicates padding position.
    """
    L, D = patches.shape

    if L == max_patches:
        return patches, grids
    elif L < max_patches:
        # Pad
        pad_len = max_patches - L
        zero_patches = torch.zeros(pad_len, D, device=patches.device)
        invalid_grids = torch.full((pad_len, 3), -1, device=grids.device)
        return (
            torch.cat([patches, zero_patches], 0),
            torch.cat([grids, invalid_grids], 0),
        )
    else:
        # Truncate
        logger.error(
            f"Truncating Image Patches from {L} to {max_patches} should not happen."
        )
        return None, None


def pad_empty_images_to_target_batch_size(
    patches: torch.Tensor,
    grids: torch.Tensor,
    max_images: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad vision encoder batch with blank images if needed."""
    N, L, D = patches.shape
    if N >= max_images:
        return patches, grids

    blank_count = max_images - N
    blank_patches = torch.zeros(blank_count, L, D, device=patches.device)
    blank_grids = torch.full((blank_count, L, 3), -1, device=grids.device)
    return (
        torch.cat([patches, blank_patches], dim=0),
        torch.cat([grids, blank_grids], dim=0),
    )

# Used to check our models performance on multiple choice tasks. This can also be done in a more involved way with e.g. LLM-as-a-judge
def check_multiple_choice_with_regex(model_outputs, correct_answers):
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        # Strip any trailing newlines and convert to uppercase
        correct_answer = correct_answer.rstrip('\n').upper()

        # Look for the answer letter at the beginning of a line or as the last word
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break  # Exit inner loop once a match is found
        results.append(match_found)
    return results


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.
    """
    top_k = min(top_k, logits.size(-1))  # Safety

    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

# NOTE Copied from torchtune.modules.transforms.vision_utils.tile_crop.py
def tile_crop(image: torch.Tensor, tile_size: int) -> torch.Tensor:
    """
    Divides a tensor into equally sized tiles. The tensor should be divisible by tile_size.

    Args:
        image (torch.Tensor): Input image to crop into tiles.
        tile_size (int): Size of each tile.

    Returns:
        torch.Tensor: torch.Tensor of shape [num_tiles, channel_size, tile_size, tile_size]

    Examples:
        >>> image = torch.rand(3, 200, 300)
        >>> tiles = tile_crop(image, tile_size=50)
        >>> tiles.shape # 4x6 = 24 tiles
        torch.Size([24, 3, 50, 50])

        >>> image = torch.rand(3, 400, 600)
        >>> tiles = tile_crop(image, tile_size=200)
        >>> tiles.shape # 2x3 = 6 tiles
        torch.Size([6, 3, 200, 200])
    """

    channel_size, height, width = image.shape

    # assert sizes are divisible
    assert (
        height % tile_size == 0 and width % tile_size == 0
    ), f"Image size {height}x{width} is not divisible by tile size {tile_size}"

    # Reshape to split height and width into tile_size blocks
    tiles_height = height // tile_size
    tiles_width = width // tile_size

    reshaped = image.view(channel_size, tiles_height, tile_size, tiles_width, tile_size)

    # Transpose to bring tiles together
    # We want [tiles_height, tiles_width, channel_size, tile_size, tile_size]
    transposed = reshaped.permute(1, 3, 0, 2, 4)

    # Flatten the tiles
    tiles = transposed.contiguous().view(
        tiles_height * tiles_width, channel_size, tile_size, tile_size
    )

    return tiles


# NOTE Copied from torchtune.modules.transforms.vision_utils.resize_with_pad.py
def resize_with_pad(
    image: torch.Tensor,
    target_size: tuple[int, int],
    resample: torchvision.transforms.InterpolationMode,
    max_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Resizes and pads an image to target_size without causing distortion.
    The user can set max_size to limit upscaling when target_size exceeds image_size.

    Args:
        image (torch.Tensor): The input image tensor in the format [..., H, W].
        target_size (Tuple[int, int]): The desired resolution to fit the image into in the format [height, width].
        resample (torchvision.transforms.InterpolationMode): Resampling method used when resizing images.
            Supports torchvision.transforms.InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT,
            InterpolationMode.BILINEAR and InterpolationMode.BICUBIC.
        max_size (Optional[int]): The maximum size to upscale the image to.
            If None, will upscale up to target_size.

    Returns:
        torch.Tensor: The resized and padded image tensor in the format [..., H, W].

    Examples:

        Example 1: The image will be upscaled from (300, 800) to (448, 1194), since 448 is the limiting side,
        and then padded from (448, 1194) to (448, 1344).

            >>> max_size = None
            >>> image = torch.rand([3, 300, 800])
            >>> target_size = (448, 1344)
            >>> resample = torchvision.transforms.InterpolationMode.BILINEAR
            >>> output = resize_with_pad(image, target_size, resample, max_size)

        Example 2: The image will stay as is, since 800 > 600, and then padded from (300, 800) to (448, 1344).

            >>> max_size = 600
            >>> image = torch.rand([3, 300, 800])
            >>> target_size = (448, 1344)
            >>> resample = torchvision.transforms.InterpolationMode.BILINEAR
            >>> output = resize_with_pad(image, target_size, resample, max_size)

        Example 3: The image will be downscaled from (500, 1000) to (224, 448),
        and padded from (224, 448) to (448, 448).

            >>> max_size = 600
            >>> image = torch.rand([3, 500, 1000])
            >>> target_size = (448, 488)
            >>> resample = torchvision.transforms.InterpolationMode.BILINEAR
            >>> output = resize_with_pad(image, target_size, resample, max_size)

    """

    image_height, image_width = image.shape[-2:]
    image_size = (image_height, image_width)

    # If target_size requires upscaling, we might want to limit the upscaling to max_size
    if max_size is not None:
        new_target_height = min(max(image_height, max_size), target_size[0])
        new_target_width = min(max(image_width, max_size), target_size[1])
        target_size_resize = (new_target_height, new_target_width)
    else:
        target_size_resize = target_size

    # resize to target_size while preserving aspect ratio
    new_size_preserving_aspect_ratio = _get_max_res_without_distortion(
        image_size=image_size,
        target_size=target_size_resize,
    )

    image = F.resize(
        inpt=image,
        size=list(new_size_preserving_aspect_ratio),
        interpolation=resample,
        antialias=True,
    )

    image = _pad_image_top_left(image=image, target_size=target_size)

    return image


# NOTE Copied from torchtune.modules.transforms.vision_utils.resize_with_pad.py
def _pad_image_top_left(
    image: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """
    Places the image at the top left of the canvas and pads with 0 the right and bottom
    to fit to the target resolution. If target_size < image_size, it will crop the image.

    Args:
        image (torch.Tensor): The input image tensor in the format [..., H, W].
        target_size (Tuple[int, int]): The desired resolution to fit the image into in the format [height, width].

    Returns:
        torch.Tensor: The padded image tensor in the format [..., H, W].
    """

    image_size = image.shape[-2:]

    height, width = image_size
    target_height, target_width = target_size

    pad_x = target_width - width
    pad_y = target_height - height

    padding = [0, 0, pad_x, pad_y]
    return F.pad(inpt=image, padding=padding)


# NOTE Copied from torchtune.modules.transforms.vision_utils.resize_with_pad.py
def _get_max_res_without_distortion(
    image_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int]:
    """
    Determines the maximum resolution to which an image can be resized to without distorting its
    aspect ratio, based on the target resolution.

    For example, if image_size = (200,400) and target_size = (600,800),
    scale_h = 600/200 = 3
    scale_w = 800/400 = 2
    So the maximum that we can upscale without distortion is min(scale_h, scale_w) = 2

    Since scale_w is the limiting side, then new_w = target_w, and new_h = old_h*scale_w

    Args:
        image_size (Tuple[int, int]): The original resolution of the image.
        target_size (Tuple[int, int]): The desired resolution to fit the image into.
    Returns:
        Tuple[int, int]: The optimal dimensions to which the image should be resized.
    Examples:
        >>> _get_max_res_without_distortion([200, 300], target_size = (450, 200))
        (133, 200)
        >>> _get_max_res_without_distortion([800, 600], target_size = (450, 1300))
        (450, 337)
    """

    original_height, original_width = image_size
    target_height, target_width = target_size

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.floor(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.floor(original_width * scale_h), target_width)

    return new_height, new_width


# NOTE Copied from torchtune.modules.transforms.vision_utils.get_canvas_best_fit.py
def _get_factors(n: int) -> Set[int]:
    """
    Calculate all factors of a given number, i.e. a divisor that leaves no remainder.

    Args:
        n (int): The number to find factors for.

    Returns:
        set: A set containing all factors of the number.

    Examples:
        >>> _get_factors(n=12)
        {1, 2, 3, 4, 6, 12}
    """
    factors_set = set()

    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors_set.add(i)
            factors_set.add(n // i)
    return factors_set


# NOTE Copied from torchtune.modules.transforms.vision_utils.get_canvas_best_fit.py
def get_canvas_best_fit(
    image: torch.Tensor, possible_resolutions: torch.Tensor, resize_to_max_canvas: bool
) -> tuple[int, int]:
    """
    Determines the best canvas possible from a list of possible resolutions to
    resize an image to, without distortion.

    For each possible resolution, calculates the scaling factors for
    width and height, and selects the smallest one, which is the limiting side.
    E.g. if to match a canvas shape you have to upscale an image's height by 2x, and width by 1.5x,
    then the maximum upscaling without distortion is min(2, 1.5) = 1.5.

    If there are multiple canvases that satisfy the conditions,
    we pick the one with the lowest area to minimize padding.

    Args:
        image (torch.Tensor): The image we want to fit into a canvas.
        possible_resolutions (torch.Tensor): A tensor of shape (N, 2) where each
            row represents a possible canvas.
        resize_to_max_canvas (bool): If True, pick the canvas that allows maximum scaling.
            If False, pick the canvas that minimizes downscaling, including no downscaling at all.

    Returns:
        Tuple[int, int]: The best resolution to fit the image into.

    Examples:
        >>> image = torch.rand(3, 200, 300)
        >>> possible_resolutions = torch.tensor([
        ...     [224, 672],
        ...     [672, 224],
        ...     [224, 448],
        ...     [448, 224],
        ...     [224, 224]
        ... ])
        >>> get_canvas_best_fit(image, possible_resolutions, resize_to_max_canvas=False)
        (224, 448)

        In the example above, we calculate the scaling factors for each possible resolution

        >>> scale_height = torch.tensor([1.1200, 3.3600, 1.1200, 2.2400, 1.1200])
        >>> scale_width = torch.tensor([2.2400, 0.7467, 1.4933, 0.7467, 0.7467])
        >>> scales = torch.tensor([1.1200, 0.7467, 1.1200, 0.7467, 0.7467])

        Two options have scaling_factor > 1, since resize_to_max_canvas is False, we pick the smallest

        >>> upscaling_options = torch.tensor([1.1200, 1.1200])
        >>> selected_scale = torch.tensor(1.1200)

        There are two possible options, so we pick the one with the smallest area

        >>> areas = torch.tensor([150528, 100352])  # for resolutions [672, 224] and [224, 448], respectively
        >>> optimal_canvas = torch.tensor([224, 448])  # resolution with the smallest area
    """

    original_height, original_width = image.shape[-2:]

    # possible resolutions heights/widths
    target_heights, target_widths = (
        possible_resolutions[:, 0],
        possible_resolutions[:, 1],
    )

    # scaling factors to resize the image without distortion
    scale_w = target_widths / original_width
    scale_h = target_heights / original_height

    # get limiting side scaling -> no distortion
    scales = torch.where(scale_w > scale_h, scale_h, scale_w)

    # filter only scales that allow upscaling
    upscaling_options = scales[scales >= 1]
    if len(upscaling_options) > 0:
        if resize_to_max_canvas:
            selected_scale = torch.max(upscaling_options)
        else:
            selected_scale = torch.min(upscaling_options)
    else:
        # no upscaling possible,
        # get the minimum downscaling (max scale for scales<1)
        downscaling_options = scales[scales < 1]
        selected_scale = torch.max(downscaling_options)

    # get all resolutions that support this scaling factor,
    # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
    chosen_canvas = possible_resolutions[scales == selected_scale]

    # if there are multiple resolutions,
    # get the one with minimum area to reduce padding
    if len(chosen_canvas) > 1:
        areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
        optimal_idx = torch.argmin(areas)
        optimal_canvas = chosen_canvas[optimal_idx]
    else:
        optimal_canvas = chosen_canvas[0]

    return tuple(optimal_canvas.tolist())


# NOTE Copied from torchtune.modules.transforms.vision_utils.get_canvas_best_fit.py
def find_supported_resolutions(
    max_num_tiles: int, tile_size: int
) -> list[tuple[int, int]]:
    """
    Computes all combinations of resolutions, multiple of tile_size,
    that contain up to max_num_tiles. Useful for when dividing an image into tiles.

    For example, if we want at most 2 tiles per image, then we can support the
    following resolutions: (1x1, 1x2, 2x1) * tile_size

    Args:
        max_num_tiles (int): Maximum number of tiles.
        tile_size (int): Size of the side of the tile.

    Returns:
        List[Tuple[int, int]]: List of possible resolutions as tuples (height, width).

    Examples:

        >>> max_num_tiles = 4
        >>> tile_size = 224
        >>> find_supported_resolutions(max_num_tiles, tile_size)
        [(224, 896), (448, 448), (224, 224), (896, 224), (224, 672), (672, 224), (224, 448), (448, 224)]
    """

    # create dictionary {aspect_ratio: [resolution1, ..., resolution n]}
    # example {0.25: [(1,4)], 1.0: [(2,2), (1,1)], 4.0: [(4,1)]}
    asp_dict = defaultdict(list)
    for _tile_size in range(max_num_tiles, 0, -1):
        factors = sorted(_get_factors(_tile_size))
        asp_ratios = [(factor, _tile_size // factor) for factor in factors]
        for height, width in asp_ratios:
            ratio_float = height / width
            asp_dict[ratio_float].append((height, width))

    # get the resolutions multiplied by the tile_size
    possible_resolutions = []
    for ar, resolution in asp_dict.items():
        for height, width in resolution:
            possible_resolutions.append((height * tile_size, width * tile_size))

    return possible_resolutions


# NOTE Copied from torchtune.data._utils.py
def load_image(image_loc) -> torch.Tensor:
    """
    Convenience method to load an image in torch.Tensor format from a local file path or remote source.

    Args:
        image_loc (Union[Path, str]): Local file path or remote source pointing to the image
            which will be loaded in PIL format.

    Note:
        If loading an image from a remote source, the function expects the URL provided in ``image_loc``
        to start with "http" or "https" e.g. "https://www.wikipedia.org/en/bird.jpg".

    Raises:
        ValueError: If the image cannot be loaded from remote source, **or**
        if the image cannot be opened as a :class:`~torch.Tensor`.

    Examples:
        >>> # Load from remote source
        >>> image = load_image("https://www.wikipedia.org/en/bird.jpg")

        >>> # Load from local file path
        >>> image = load_image(Path("/home/user/bird.jpg"))

    Returns:
        torch.Tensor: The loaded image.
    """

    # If pointing to remote source, try to load to local
    if isinstance(image_loc, str) and image_loc.startswith("http"):
        try:
            image_loc = request.urlopen(image_loc).read()
            image = torchvision.io.decode_image(
                torch.frombuffer(image_loc, dtype=torch.uint8),
                mode="RGB",
            )
        except Exception as e:
            raise ValueError("Failed to load remote image as torch.Tensor") from e

    # Open the local image as a Tensor image
    else:
        try:
            image = torchvision.io.decode_image(image_loc, mode="RGB")
        except Exception as e:
            raise ValueError("Failed to load local image as torch.Tensor") from e

    return image
