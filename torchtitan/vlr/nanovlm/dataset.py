from torch.distributed.checkpoint.stateful import Stateful
from 

from torchtitan.components.dataloader import ParallelAwareDataloader

class MM_Dataset(IterableDataset, Stateful):
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

    ds = MM_Dataset(

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