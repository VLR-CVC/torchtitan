# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import time
from typing import Optional, List, Dict, Any
import numpy as np

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoProcessor
from PIL import Image

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger


class Generator:
    """Generator class for SmolVLM model inference."""
    
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
        
        # Initialize distributed
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=False,
            base_folder=job_config.job.dump_folder,
        )
        
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )
        
        world_mesh = parallel_dims.world_mesh
        
        # Set random seed
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            deterministic=False,
        )
        
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)
        
        # Build tokenizer
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )
        
        # Build model
        model_args = self.train_spec.model_args[job_config.model.flavor]
        model_args.update_from_config(job_config)
        self.model_args = model_args
        
        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_args}"
        )
        
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            model = self.train_spec.model_cls(model_args)
        
        # Build model converters (e.g., for float8)
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)
        
        # Apply parallelism
        if parallel_dims.pp_enabled:
            raise NotImplementedError("Pipeline parallelism not supported for generation")
        else:
            model = self.train_spec.parallelize_fn(model, parallel_dims, job_config)
            
            # Move to device and initialize
            init_device = self.device.type
            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights()
            model.eval()
            
            self.model_parts = [model]
        
        # Setup checkpoint manager for loading
        self.checkpointer = CheckpointManager(
            dataloader=None,  # No dataloader needed for generation
            model_parts=self.model_parts,
            optimizers=None,  # No optimizer needed for generation
            lr_schedulers=None,  # No lr_scheduler needed for generation
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
            ft_manager=None,  # No fault tolerance for generation
        )
        
        # Load checkpoint
        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Loaded checkpoint from step {job_config.checkpoint.load_step}")
        
        # Setup HF processor for image processing
        #processor_path = getattr(model_args, 'tokenizer_name',)
        self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-256M-Video-Instruct')
        self.image_processor = self.processor.image_processor
        
        # Load chat template
        template_path = "torchtitan/vlr/smolvlm/datasets/template.jinja"
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                self.chat_template = f.read()
        else:
            logger.warning(f"Chat template not found at {template_path}, using default")
            self.chat_template = None
        
        # Setup generation parameters
        self.max_new_tokens = getattr(job_config, 'max_new_tokens', 256)
        self.temperature = getattr(job_config, 'temperature', 0.7)
        self.top_p = getattr(job_config, 'top_p', 0.9)
        self.top_k = getattr(job_config, 'top_k', 50)
        
        logger.info("Generator initialized successfully")
    
    @torch.no_grad()
    def generate(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: bool = True,
    ) -> str:
        """Generate text from messages and optional images.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            images: Optional list of PIL images
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling or greedy decoding
        
        Returns:
            Generated text string
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        
        model = self.model_parts[0]
        model.eval()
        
        # Process images if provided
        pixel_values = None
        patch_attention_mask = None
        
        if images:
            # Process images using HF processor
            vision_inputs = self.image_processor(images)
            pixel_values = torch.tensor(
                np.array(vision_inputs['pixel_values'])
            ).to(self.device, dtype=torch.bfloat16)
            
            if 'pixel_attention_mask' in vision_inputs:
                patch_attention_mask = torch.tensor(
                    vision_inputs['pixel_attention_mask']
                ).to(self.device)
            
            # Handle batch dimension
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(0)
            if patch_attention_mask is not None and patch_attention_mask.dim() == 3:
                patch_attention_mask = patch_attention_mask.unsqueeze(0)
        
        # Tokenize input
        if self.chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                chat_template=self.chat_template,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            # Fallback to default chat template
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        
        input_ids = input_ids.to(self.device)
        
        # Setup generation context (compile if enabled)
        generate_fn = self._generate_tokens
        if self.job_config.compile.enable and "model" in self.job_config.compile.components:
            generate_fn = torch.compile(generate_fn, mode="reduce-overhead")
        
        # Generate tokens
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output_ids = generate_fn(
                model=model,
                input_ids=input_ids,
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
        
        # Decode output
        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def _generate_tokens(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        patch_attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """Core generation loop."""
        
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        
        # Cache for key-value pairs (if using KV cache in the future)
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                # Prepare input dict
                input_dict = {
                    "input_ids": generated_ids,
                    "eos_id": self.tokenizer.eos_token,
                }
                
                if pixel_values is not None:
                    input_dict["pixel_values"] = pixel_values
                
                if patch_attention_mask is not None:
                    input_dict["patch_attention_mask"] = patch_attention_mask
                
                # Get model output
                logits = model(**input_dict)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample or greedy decode
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check for EOS token
                if (next_token == self.tokenizer.eos_token):
                    break
        
        return generated_ids
    
    def interactive_generate(self):
        """Interactive generation mode for testing."""
        logger.info("Starting interactive generation mode. Type 'quit' to exit.")
        
        while True:
            try:
                user_input = input("\nEnter your prompt (or 'quit' to exit): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                # Check if user wants to include an image
                image_path = input("Enter image path (or press Enter to skip): ").strip()
                
                images = None
                if image_path and os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    # Resize to expected size
                    image = image.resize((512, 512))
                    images = [image]
                    logger.info(f"Loaded image from {image_path}")
                elif image_path:
                    logger.warning(f"Image path {image_path} not found, proceeding without image")
                
                # Create message format
                messages = [
                    {
                        "user": user_input,
                        "assistant": ""  # Will be filled by generation
                    }
                ]
                
                logger.info("Generating response...")
                start_time = time.perf_counter()
                
                response = self.generate(messages, images=images)
                
                generation_time = time.perf_counter() - start_time
                logger.info(f"Generation completed in {generation_time:.2f}s")
                
                print(f"\nGenerated response:\n{response}")
                
            except KeyboardInterrupt:
                logger.info("\nInterrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
    
    def batch_generate(self, input_file: str, output_file: str):
        """Generate responses for a batch of inputs from a file.
        
        Args:
            input_file: Path to JSON file with inputs
            output_file: Path to save outputs
        """
        import json
        
        logger.info(f"Loading inputs from {input_file}")
        
        with open(input_file, 'r') as f:
            inputs = json.load(f)
        
        results = []
        for i, item in enumerate(inputs):
            logger.info(f"Processing item {i+1}/{len(inputs)}")
            
            messages = item.get('messages', [])
            image_paths = item.get('images', [])
            
            # Load images if provided
            images = []
            for path in image_paths:
                if os.path.exists(path):
                    image = Image.open(path).convert('RGB').resize((512, 512))
                    images.append(image)
            
            # Generate response
            response = self.generate(messages, images=images if images else None)
            
            results.append({
                'input': item,
                'output': response
            })
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
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
        
        # Check for generation mode from config or command line
        generation_mode = getattr(config, 'generation_mode', 'interactive')
        
        if generation_mode == 'interactive':
            generator.interactive_generate()
        elif generation_mode == 'batch':
            input_file = getattr(config, 'input_file', 'inputs.json')
            output_file = getattr(config, 'output_file', 'outputs.json')
            generator.batch_generate(input_file, output_file)
        else:
            # Single generation example
            messages = [
                {
                    "user": "What is the capital of France?",
                    "assistant": ""
                }
            ]
            response = generator.generate(messages)
            logger.info(f"Generated: {response}")
    
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        if generator:
            generator.close()
        raise
    else:
        if generator:
            generator.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
