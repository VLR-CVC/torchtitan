# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import time
from typing import Optional, List, Dict, Any

import torch
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger

# --- Generation utilities from scripts/generate/_generation.py ---

def multinomial_sample_one(
    probs: torch.Tensor, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    # The model forward pass in torchtitan expects a `tokens` argument.
    logits = model(tokens=x)
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token


@torch.no_grad()
def _generate_sequence(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # ensure batch dimension (T,) --> (B, T)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    generated_tokens = input_ids.clone()

    for _ in range(max_new_tokens):
        next_token = generate_next_token(
            model,
            x=generated_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    return generated_tokens

# --- End of generation utilities ---


class Generator:
    """Generator class for Llama3 model inference."""

    def __init__(self, job_config: JobConfig):
        torch._C._log_api_usage_once("torchtitan.generate")

        self.job_config = job_config

        logger.info(f"Starting generation: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ.get('LOCAL_RANK', 0))}")
        device_module.set_device(self.device)

        # For generation, we usually use a single process or TP.
        # We will not initialize the full distributed setup unless necessary.
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            dist_utils.init_distributed(
                job_config.comm,
                enable_cpu_backend=False,
                base_folder=job_config.job.dump_folder,
            )

        parallelism_config = job_config.parallelism
        self.parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

        dist_utils.set_determinism(
            self.parallel_dims.world_mesh if world_size > 1 else None,
            self.device,
            job_config.training.seed,
            deterministic=False,
        )

        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        self.tokenizer = self.train_spec.build_tokenizer_fn(job_config)

        model_args = self.train_spec.model_args[job_config.model.flavor]
        model_args.update_from_config(job_config)
        self.model_args = model_args

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            model = self.train_spec.model_cls(model_args)

        model_converters = build_model_converters(job_config, self.parallel_dims)
        model_converters.convert(model)

        if self.parallel_dims.pp_enabled:
            raise NotImplementedError("Pipeline parallelism not supported for generation")
        else:
            model = self.train_spec.parallelize_fn(model, self.parallel_dims, job_config)

            init_device = self.device.type
            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights()
            model.eval()

            self.model_parts = [model]

        self.checkpointer = CheckpointManager(
            dataloader=None,
            model_parts=self.model_parts,
            optimizers=None,
            lr_schedulers=None,
            states={},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                self.train_spec.state_dict_adapter(
                    model_args, job_config.model.hf_assets_path
                )
                if self.train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.job.dump_folder,
            ft_manager=None,
        )

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Loaded checkpoint from step {job_config.checkpoint.load_step}")

        self.max_new_tokens = getattr(job_config, 'max_new_tokens', 256)
        self.temperature = getattr(job_config, 'temperature', 0.7)
        self.top_k = getattr(job_config, 'top_k', 50)

        logger.info("Generator initialized successfully")

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k

        model = self.model_parts[0]
        model.eval()

        # For simplicity, this example handles one prompt at a time.
        # Batching can be added for efficiency.
        generated_texts = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output_ids = _generate_sequence(
                    model=model,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                )

            generated_ids = output_ids[0, input_ids.shape[0]:]
            generated_text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts

    def close(self):
        """Cleanup resources."""
        if hasattr(self, 'checkpointer'):
            self.checkpointer.close()
        logger.info("Generator closed")


@record
def main():
    """Main entry point for generation."""
    init_logger()

    # Parse configuration
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    generator = None
    try:
        # Initialize generator
        generator = Generator(config)

        prompts = [
            "What is the meaning of life?",
            "Translate 'hello world' to French.",
        ]

        logger.info(f"Generating for prompts: {prompts}")
        start_time = time.perf_counter()

        responses = generator.generate(prompts)

        generation_time = time.perf_counter() - start_time
        logger.info(f"Generation completed in {generation_time:.2f}s")

        for prompt, response in zip(prompts, responses):
            print("-" * 20)
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 20)

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        if generator:
            generator.close()
        raise
    else:
        if generator:
            generator.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
