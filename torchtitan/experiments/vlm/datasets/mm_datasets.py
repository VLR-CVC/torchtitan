# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal dataset implementation for vision-language models training.

This module provides dataset classes for handling multimodal data
including images and text. Images are interleaved with text at native aspect ratio and resolution.
It supports both streaming and non-streaming datasets from HuggingFace.
"""

from typing import Any, Callable

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.config import JobConfig
from torchtitan.datasets import DatasetConfig
from torchtitan.tools.logging import logger
from torchtitan.datasets.utils import load_image_PIL

from ..model.args import SpecialTokens
from .mm_collator_nld import MultiModalCollatorNLD
from .utils.image import calculate_image_tokens, process_image
from .utils.packing import SamplePacker
from .utils.text import process_text_with_images


def _process_mm_sample(
    texts: list[str] | str,
    images: list[bytes] | bytes,
    tokenizer: BaseTokenizer,
    patch_size: int,
    max_patch_per_image: int,
    spatial_merge_size: int,
    special_tokens: SpecialTokens,
) -> dict[str, Any] | None:
    """Common processing logic for multimodal samples.

    Args:
        texts: List of strings with None indicating image positions
        images: List of image bytes with None for text positions
        tokenizer: Tokenizer for text processing
        patch_size: Size of image patches
        max_patch_per_image: Maximum patches per image
        spatial_merge_size: merge 2D image patches to reduce LLM's sequence length.
            - if 1 (default): no merge, effectively NoOp
            - if 2: 2x2=4 image patches will be reduced to 1 LLM sequence

    Returns:
        Dict with:
            - input_ids: Tensor of token IDs
            - labels: Tensor of label IDs
            - pixel_values: List of processed image tensors

    Example:
        Interleaved format:
        texts = [text1, None, text2, None, text3]
        images = [None, img1, None, img2, None]

        Image-text pair format as a special case of interleaved:
        texts = [None, text]
        images = [image, None]
    """
    try:
        # Normalize inputs to lists
        texts = [texts] if isinstance(texts, str) else texts
        images = [images] if isinstance(images, bytes) else images

        if not texts or len(texts) != len(images):
            return None

        # Process all images first
        processed_images = []
        image_dimensions = []
        texts_list = list(texts)  # Make mutable copy

        for idx, img in enumerate(images):
            if img is not None:
                processed_img = process_image(
                    img,
                    patch_size=patch_size,
                    merge_size=spatial_merge_size,
                    max_patch_per_image=max_patch_per_image,
                )
                if processed_img is not None:
                    num_tokens, width, height = calculate_image_tokens(
                        processed_img,
                        patch_size=patch_size,
                        spatial_merge_size=spatial_merge_size,
                    )
                    processed_images.append(processed_img)
                    image_dimensions.append((num_tokens, width, height))
                    # Replace None with image token
                    texts_list[idx] = special_tokens.img_token
                else:
                    # Replace None with empty string if processing failed
                    texts_list[idx] = ""

        if len(processed_images) != len([_ for _ in images if _ is not None]):
            logger.warning("Cannot process all images for sample. Dropping")
            return None

        # Process all image tokens at once
        processed_text = process_text_with_images(
            texts_list, image_dimensions, tokenizer, special_tokens, add_eos=True
        )

        tokens = tokenizer.encode(processed_text)

        # Convert to tensors
        input_ids = torch.tensor(tokens)
        labels = torch.tensor(tokens)

        # Mask special tokens in labels
        special_token_ids = torch.tensor(
            [special_tokens.boi_id, special_tokens.eoi_id, special_tokens.img_id]
        )
        labels = torch.where(
            torch.isin(labels, special_token_ids), special_tokens.ignore_id, labels
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": processed_images,
        }

    except Exception as e:
        logger.warning(f"Error processing sample: {e}")
        return None


def _process_obelics_sample(
    sample: dict[str, Any],
    tokenizer: HuggingFaceTokenizer,
    patch_size: int,
    spatial_merge_size: int,
    max_patch_per_image: int,
    special_tokens: SpecialTokens,
) -> dict[str, Any] | None:
    """Process a sample from the OBELICS dataset."""
    return _process_mm_sample(
            texts=sample.get("texts", []),
            images=sample.get("images", []),
            tokenizer=tokenizer,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            max_patch_per_image=max_patch_per_image,
            special_tokens=special_tokens,
            )

def _process_finevision_sample(
        sample: dict[str, Any],
        ) -> dict[str, Any] | None:

    sample_texts = sample.get('texts')
    sample_images = sample.get('images')

def _process_cc12_wd_sample(
    sample: dict[str, Any],
    tokenizer: BaseTokenizer,
    patch_size: int,
    spatial_merge_size: int,
    max_patch_per_image: int,
    special_tokens: SpecialTokens,
) -> dict[str, Any] | None:
    """Process a sample from the CC12-WD dataset.
    Transforms CC12-WD format to match Interleaved format:
    - texts: [None, text] to indicate image position
    - images: [image, None] to match text position
    """
    text = sample.get("txt", "")
    image = sample.get("jpg", None)

    return {
        "images": images,
        "texts": sample_texts,
    }


MM_DATASETS = {
    "obelics": DatasetConfig(
        path="HuggingFaceM4/OBELICS",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_process_obelics_sample,
    ),
    "cc12m": DatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_process_cc12_wd_sample,
    ),
    "finevision": MMDatasetConfig(
        path="HuggingFaceM4/FineVision",
        loader=lambda path: load_dataset(path, split="train", name='docvqa', streaming=True),
        sample_processor=_process_finevision_sample,
    ),
}


def _validate_mm_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in MM_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(MM_DATASETS.keys())}"
        )

    config = MM_DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


def find_tensor_match_all(
    subsequence_tensor: torch.Tensor, main_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Finds all occurrences of a subsequence tensor within a main tensor. Used for the creation of a mask.
    """
    sub_len = len(subsequence_tensor)
    main_len = len(main_tensor)

    assert sub_len < main_len

    if sub_len > main_len:
        return torch.zeros(main_len, dtype=torch.bool)

    all_possible_slices = main_tensor.unfold(0, sub_len, 1)

    comparison_result = torch.eq(all_possible_slices, subsequence_tensor)
    is_match_start = torch.all(comparison_result, dim=1)

    output_tensor = torch.zeros(main_len, dtype=torch.bool)
    start_indices = torch.nonzero(is_match_start, as_tuple=True)[0]

    for start_index in start_indices:
        output_tensor[start_index : start_index + sub_len] = True

    return output_tensor


class MultiModalDataset(IterableDataset, Stateful):
    """MultiModal Dataset with support for sample packing."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        batch_size: int,
        seq_len: int,
        patch_size: int,
        spatial_merge_size: int,
        #max_patches_per_image: int,
        #max_images_per_batch: int,
        packing_buffer_size: int,
        special_tokens: SpecialTokens,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, self.sample_processor = _validate_mm_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        #self.max_patches_per_image = max_patches_per_image
        #self.max_images_per_batch = max_images_per_batch
        self.special_tokens = special_tokens
        self.enable_packing = packing_buffer_size > 0
        if self.enable_packing:
            self.packer = SamplePacker(
                max_seq_length=seq_len,
                buffer_size=packing_buffer_size,
                batch_size=batch_size,
            )
        self.infinite = infinite
        self._sample_idx = 0

        self.chat_template = open("torchtitan/experiments/vlm/datasets/template.jinja").read()

        self._tokenizer.image_id = 49190

        # MAGIC NUMBERS
        tile_size = 224
        max_num_tiles = 1
        self.transform_image = CLIPTransform(
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
        )


    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1
                # all happens inside here
                processed = self.process_sample(sample)

                if processed["input_ids"].shape[0] > self.seq_len:
                    logger.warning(
                        f"Sample length {processed['input_ids'].shape[0]} > training {self.seq_len=}. Skip"
                    )
                    continue

                if self.enable_packing:
                    self.packer.add_sample(processed)

                    if self.packer.has_batch_ready():
                        batch = self.packer.get_next_batch()
                        if batch:
                            yield from batch
                else:
                    yield processed  # individual sample

                """
                except Exception as e:
                    logger.warning(f"Error in iteration: {e}")
                    continue
                """

            if self.enable_packing:
                # Handle remaining samples in packer
                while True:
                    batch = self.packer.get_next_batch()
                    if batch:
                        yield from batch
                    else:
                        break

            if not self.infinite:
                break
            else:
                self._sample_idx = 0

    def process_sample(
        self, sample
    ) -> dict[str, Any]:
        sample = self.sample_processor(sample)
        processed_images = [self.transform_image(img) for img in sample["images"]]

        images = [img['image'] for img in processed_images]
        aspect_ratios = [img['aspect_ratio'] for img in processed_images]

        messages = sample['texts']

        conv_ids, mask, attention_mask = self.apply_template_and_create_mask(
            messages
        )

        labels = self._get_labels(conv_ids, mask)

        return {
            "images": images,
            "aspect_ratios": aspect_ratios,
            "input_ids": conv_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_data_iter(self):
        try:
            if not hasattr(self._data, "iterable_dataset"):
                if isinstance(self._data, Dataset) and (
                    self._sample_idx == len(self._data)
                ):
                    return iter([])

            it = iter(self._data)

            if self._sample_idx > 0:
                for _ in range(self._sample_idx):
                    next(it)

            return it
        except Exception as e:
            logger.error(f"Error in _get_data_iter: {e}")
            return iter([])


    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, IGNORE_INDEX)
        labels = labels.roll(-1, dims=0)
        labels[-1] = IGNORE_INDEX  # last token has no target
        return labels


    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

        # Restore packer state if available
        if (
            self.enable_packing
            and hasattr(self, "packer")
            and "packer_state" in state_dict
        ):
            packer_state = state_dict["packer_state"]
            self.packer.sample_buffer.clear()
            self.packer.packed_samples.clear()
            self.packer.sample_buffer.extend(packer_state["sample_buffer"])
            self.packer.packed_samples.extend(packer_state["packed_samples"])

    def state_dict(self):
        state = {"sample_idx": self._sample_idx}

        # Save packer state if packing is enabled
        if self.enable_packing and hasattr(self, "packer"):
            state["packer_state"] = {
                "sample_buffer": list(self.packer.sample_buffer),
                "packed_samples": list(self.packer.packed_samples),
            }

        return state

    def apply_template_and_create_mask(self, messages):
        """
        Adding BOS and EOS tokens and applies the conversation turns.
        """
        conv_ids = self._tokenizer.apply_chat_template(
            messages,
            tools=None,
            tokenize=True,
            chat_template=self.chat_template,
            add_special_tokens=True,
            return_dict=True,
        )

        masks = []
        for msg in messages:
            all_ids = torch.tensor(
                self._tokenizer.apply_chat_template(
                    [msg],
                    tools=None,
                    tokenize=True,
                    add_generation_prompt=False,
                    chat_template=self.chat_template,
                ),
                dtype=torch.uint64,
            )

            ass_ids = torch.tensor(
                self._tokenizer.encode(msg["assistant"]),
                dtype=torch.uint64,
            )

            tmp_mask = find_tensor_match_all(ass_ids, all_ids)

            assert all_ids.shape == tmp_mask.shape

            masks.append(tmp_mask)

        labels = torch.cat(masks)
        # the mask that is passed is for the whole image, not just one Q/A pair
        assert len(conv_ids["input_ids"]) == labels.shape[0]

        return (
            torch.tensor(conv_ids["input_ids"]),
            labels,
            torch.tensor(conv_ids["attention_mask"]),
        )



def build_mm_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: HuggingFaceTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for multimodal datasets.

    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer for text processing
        job_config: Job configuration
        infinite: Whether to loop infinitely

    Returns:
        DataLoader with appropriate parallelism handling
    """
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    max_images_per_batch = job_config.data.max_images_per_batch
    max_patches_per_image = job_config.data.max_patches_per_image
    # NOTE: technically patch_size belongs to model variants, but we don't
    # have access to model_args here. To discuss later.
    patch_size = job_config.data.patch_size
    spatial_merge_size = job_config.data.spatial_merge_size
    packing_buffer_size = job_config.data.packing_buffer_size
    special_tokens = SpecialTokens.from_tokenizer(tokenizer)

    dataset = MultiModalDataset(
        dataset_name=job_config.training.dataset,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        packing_buffer_size=packing_buffer_size,
        special_tokens=special_tokens,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    collate_fn = MultiModalCollatorNLD(
        batch_size=batch_size,
        seq_len=job_config.training.seq_len,
        patch_size=patch_size,
        max_images_per_batch=max_images_per_batch,
        max_patches_per_image=max_patches_per_image,
        special_tokens=special_tokens,
    )

    base_dataloader = ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    return base_dataloader
