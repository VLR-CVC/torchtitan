from dataclasses import dataclass

from torch import nn

from torchtitan.config.job_config import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs


@dataclass
class SmolVLMArgs(BaseModelArgs):
    hidden_size = 1152
    intermediate_size = 3072
    num_hidden_layers = 12
    num_attention_heads = 16
    num_channels = 3
    image_size = 224
    patch_size = 32
    hidden_act = "gelu_pytorch_tanh"
    layer_norm_eps = 1e-6
    attention_dropout = 0.0
    initializer_range = 0.02

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        return super().update_from_config(job_config, **kwargs)

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return super().get_nparams_and_flops(model, seq_len)
