import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, Dataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.datasets.hf_datasets import DATASETS, _validate_dataset
from torchtitan.tools.logging import logger
from torchtitan.config import JobConfig

from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer

from typing import Any

from .transform import CLIPTransform
from .collator import MultiModalCollator

CHAT_TEMPLATE = "{%- for message in messages %}{{'<image><|im_start|>user' + '\\n' + message['user'] + '<|im_end|>' }}\n{{'<|im_start|>assistant' + '\\n' + message['assistant'] + '<|im_end|>' }}{%- endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

IGNORE_INDEX = -100
IMAGE_TOKEN_ID = 49190


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


class Multimodal_dataset(IterableDataset, Stateful):
    """
    Works for a single images. Concats all of the text, including the separators for the user
    messages and the assistant messages (see CHAT_TEMPLATE).

    Labels are provided to make the Next Token Prediction task, only showing the IDs of the answers,
    which is the part of the text that the model would have to learn to predict.

    The loss should be defined only over this tokens [citation needed].
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,  # Default to 1 for single-process testing
        infinite: bool = False,
        tile_size: int = 224,  # Added default value
        max_num_tiles: int = 1,  # Added default value
    ) -> None:
        dataset_name = dataset_name.lower()

        # they come from the dataset config
        path, dataset_loader, sample_processor = _validate_dataset(
            dataset_name, dataset_path
        )

        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        # we only need the data parallel degree
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._sample_processor = sample_processor

        self._sample_idx = 0
        self._token_buffer: list[int] = []

        # MAGIC NUMBERS
        self.transform_image = CLIPTransform(
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
        )

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def __iter__(self):
        while True:
            data_iter = self._get_data_iter()
            if not (sample := next(data_iter, None)):  # Check for end of iterator
                if self.infinite:
                    self._sample_idx = 0
                    continue
                else:
                    break

            yield self.process_sample(sample)

    def process_sample(self, sample) -> dict[str, Any]:
        # gives dict[str, list[str] | list[torch.Tensor]]
        sample = self._sample_processor(sample)

        processed_images = [self.transform_image(img) for img in sample["images"]]

        images, aspect_ratios = processed_images

        assert len(images), len(aspect_ratios)

        messages = self._get_messages(sample)

        conversation_ids, mask, attention_mask = self._apply_template_and_create_mask(
            messages
        )

        labels = self._get_labels(conversation_ids, mask)

        # the image shapes are calculated later on
        return {
            "encoder_inputs": {
                "images": images,
                "aspect_ratios": aspect_ratios,
            },
            "input_ids": conversation_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_template_and_create_mask(self, messages):
        """
        Adding BOS and EOS tokens and applies the conversation turns.
        """
        conv_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=True,
            return_dict=True,
        )

        masks = []
        for msg in messages:
            all_ids = torch.tensor(
                self._tokenizer.apply_chat_template(
                    [msg], tokenize=True, add_generation_prompt=False
                ),
                dtype=torch.uint64,
            )

            ass_ids = torch.tensor(
                self._tokenizer.encode(msg["assistant"]), dtype=torch.uint64
            )

            tmp_mask = find_tensor_match_all(ass_ids, all_ids)

            assert all_ids.shape == tmp_mask.shape

            masks.append(tmp_mask)

        labels = torch.cat(masks)
        # the mask that is passed is for the whole image, not just one Q/A pair
        assert len(conv_ids["input_ids"]) == labels.shape[0]

        # Mask BOS, EOS & image tokens from the loss
        labels = torch.where(
            torch.isin(
                labels,
                torch.LongTensor(
                    [
                        self._tokenizer.bos_id,
                        self._tokenizer.eos_id,
                        self._tokenizer.image_id,
                    ]
                ),
            ),
            IGNORE_INDEX,
            labels,
        )

        return (
            torch.tensor(conv_ids["input_ids"]),
            labels,
            torch.tensor(conv_ids["attention_mask"]),
        )

    def _get_messages(self, item):
        return item["texts"]

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, IGNORE_INDEX)
        labels = labels.roll(-1, dims=0)
        labels[-1] = IGNORE_INDEX  # last token has no target
        return labels

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # TODO: needs testing
        self._token_buffer = state_dict["token_buffer"]
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self) -> dict[str, Any]:
        return {
            "token_buffer": self._token_buffer,
            "sample_idx": self._sample_idx,
        }


def build_tokenizer(config):
    pass


def build_mm_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    pad_max_tiles = 4

    # special token 1
    padding_idx = 49191

    ds = Multimodal_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        infinite=infinite,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )

    collate_fn = MultiModalCollator(
            padding_idx=padding_idx,
            pad_max_tiles=pad_max_tiles,
    )

    # this is a titan component
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
