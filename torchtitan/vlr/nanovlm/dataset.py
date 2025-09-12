import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, Dataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.datasets.hf_datasets import DATASETS, _validate_dataset
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
        dp_world_size: bool = False,
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

                encoder_input = {"images": [], "aspect_ratio": []}
                for image in sample["images"]:
                    logger.warning(image.shape)

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

    ds = Multimodal_dataset()

    collate_fn = MM_Collator()

    # this is a titan component
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
