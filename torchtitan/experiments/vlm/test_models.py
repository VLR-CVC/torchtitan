import torch

from .model import Llama3Siglip2Transformer
from .args import Llama3Siglip2ModelArgs

from torchtitan.experiments.vlm import llama3_siglip2_configs

device = torch.device("meta")
config = llama3_siglip2_configs["256M"]
model = Llama3Siglip2Transformer(config).to(device)

print(model)
