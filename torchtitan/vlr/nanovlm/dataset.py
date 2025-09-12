import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, Dataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.datasets.hf_datasets import DATASETS, _validate_dataset
from torchtitan.experiments.multimodal.transform import CLIPTransform
from torchtitan.tools.logging import logger
from torchtitan.config import JobConfig

from datasets.distributed import split_dataset_by_node


class Multimodal_dataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 0,
        infinite: bool = False,
    ) -> None:
        dataset_name = dataset_name.lower()

        # they come from the dataset config
        path, dataset_loader, process_sample = _validate_dataset(
            dataset_name, dataset_path
        )

        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        # we only need the data parallel degree
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._process_sample = process_sample

        self._sample_idx = 0
        self._token_buffer: list[int] = []

        # TODO: see what this is (numerically)
        self.prefix_len = self._get_prefix_len()

        # MAGIC NUMBERS
        self.transform_image = CLIPTransform(
            image_mean=(
                0.48145466,
                0.4578275,
                0.40821073,
            ),  # TODO(tj.solergibert) What should we do with `image_mean` & `image_std`?,
            image_std=(0.26862954, 0.26130258, 0.27577711),
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
        )

    def _get_prefix_len(self):
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self._tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string_5_letters}],
            tokenize=False,
            add_special_tokens=False,
        )
        random_string_location = random_string_chat_templated.find(
            random_string_5_letters
        )
        return len(
            self._tokenizer.encode(
                random_string_chat_templated[:random_string_location]
            )
        )

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            # we change the type of iter depending on the type of dataset
            for sample in self._get_data_iter():
                # raw text input (should include `<|image|>`)
                sample_input = self._process_sample(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_input, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1
                # the batch is prepared to fill the max amount of tokens

                processed_images = []
                for image in sample["images"]:
                    logger.warning(image.shape)
                    # TODO: load images and process

                    # this image is a tensor, apply torch transform
                    processed_images.append(image)

                tokens = self._tokenizer.encode(
                    sample["text"],
                    add_bos=True,
                    add_eos=True,
                    allowed_special=set(["<|image|>"]),
                )

                # is really neccesary a long tensor?
                input_ids = torch.LongTensor(tokens[:-1])

                # we need to mask out the assistant tokens
                messages = self._get_messages(sample)
                mask = self._apply_template_and_create_mask(messages)
                labels = self._get_labels(input_ids, mask)

                yield {
                    "images": processed_images,
                    "input_ids": input_ids,
                    "attention_mask": mask,
                    "labels": labels,
                }

    # from nanoVLM
    def _apply_template_and_create_mask(self, messages):
        conversation_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,  # no need since this is just to create a mask
            return_dict=True,
        )

        # start with blank mask
        mask = [0] * len(conversation_ids["input_ids"])

        cursor = 0
        # we need to mask out the assistant messages
        for msg in messages:
            segment_ids = self._tokenizer.apply_chat_template(
                [msg],
                tokenize=True,
                add_special_tokens=False,
            )

            seq_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seq_len
                mask[start:end] = [1] * (end - start)

            cursor += seq_len

        return (
            torch.tensor(conversation_ids["input_ids"]),
            torch.tensor(mask).to(torch.bool),
            torch.tensor(conversation_ids["attention_mask"]),
        )

    def _get_messages(self, item):
        """
        we need to divide the sample into the different messages
        """

        messages = []
        for text in item["texts"]:
            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        return messages

    # from nanoVLM
    # https://github.com/huggingface/nanoVLM/blob/9de5e17ac2f4c578c32085131d966464cdd252b5/data/datasets.py#L146C5-L151C22
    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        labels[-1] = -100  # Last token has no target

        return labels

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self) -> dict[str, Any]:
        # definition of the state_dict
        _state_dict = {"toke_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


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

    ds = Multimodal_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        infinite=infinite,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )

    collate_fn = MM_Collator()

    # this is a titan component
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
