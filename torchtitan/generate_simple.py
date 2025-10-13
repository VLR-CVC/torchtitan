# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor

from torchtitan.config import JobConfig, ConfigManager, TORCH_DTYPE_MAP
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.distributed import ParallelDims, utils as dist_utils

# Import SmolVLM specific components
from torchtitan.vlr.smolvlm.model.args import Llama3Siglip2ModelArgs, Siglip2ModelArgs
from torchtitan.vlr.smolvlm.model.model import Llama3Siglip2Transformer
from torchtitan.vlr.smolvlm.model.state_dict_adapter import SmolVLMStateDictAdapter


class SimpleGenerator:
    """Barebones generator for debugging using CheckpointManager."""
    
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config
        
        # Setup device
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ.get('LOCAL_RANK', 0))}")
        device_module.set_device(self.device)
        
        logger.info(f"Device: {self.device}")
        
        # Init distributed (needed for checkpoint loading)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1 or int(os.environ.get("RANK", 0)) >= 0:
            dist_utils.init_distributed(
                job_config.comm,
                enable_cpu_backend=False,
                base_folder=job_config.job.dump_folder,
            )
        
        # Setup parallel dims (minimal - no parallelism for inference)
        self.parallel_dims = ParallelDims(
            dp_shard=1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size,
        )
        
        # Load tokenizer using model's hf_assets_path
        tokenizer_path = job_config.model.hf_assets_path
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path)
        self.tokenizer.image_id = job_config.special_tokens.img_id
        
        logger.info(f"Tokenizer loaded from: {tokenizer_path}")
        logger.info(f"Vocab size: {len(self.tokenizer)}")
        logger.info(f"Special tokens - BOS: {self.tokenizer.bos_id}, EOS: {self.tokenizer.eos_id}, PAD: {self.tokenizer.pad_id}")
        logger.info(f"Image token ID: {self.tokenizer.image_id}")
        
        # Load image processor
        processor = AutoProcessor.from_pretrained(tokenizer_path)
        self.image_processor = processor.image_processor
        
        # Load chat template
        template_path = Path("torchtitan/vlr/smolvlm/datasets/template.jinja")
        if template_path.exists():
            with open(template_path, 'r') as f:
                self.chat_template = f.read()
            logger.info("Chat template loaded")
        else:
            logger.warning(f"Template not found at {template_path}")
            self.chat_template = None
        
        # Build model
        self.model_args = self._get_model_args()
        self.model = self._build_model()
        
        # Load checkpoint using CheckpointManager
        self._load_checkpoint()
        
        self.model.eval()
        logger.info("Model loaded and ready")
    
    def _get_model_args(self):
        """Get model args from job config."""
        from torchtitan.protocols import train_spec as train_spec_module
        
        train_spec = train_spec_module.get_train_spec(self.job_config.model.name)
        model_args = train_spec.model_args[self.job_config.model.flavor]
        model_args.update_from_config(self.job_config)
        
        # Override for inference
        model_args.use_flex_attn = False
        model_args.encoder.use_flex_attn = False
        
        logger.info(f"Model args: {model_args}")
        return model_args
    
    def _build_model(self):
        """Build model using torchtitan's approach."""
        logger.info(f"Building {self.job_config.model.name} {self.job_config.model.flavor}")
        
        dtype = TORCH_DTYPE_MAP[self.job_config.training.dtype]
        
        with torch.device("meta"), utils.set_default_dtype(dtype):
            model = Llama3Siglip2Transformer(self.model_args)
        
        # Initialize on device
        device_type = utils.device_type
        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_encoder_weights(buffer_device=device_type)
        
        logger.info("Model structure created")
        return model
    
    def _load_checkpoint(self):
        """Load checkpoint using CheckpointManager."""
        logger.info("Setting up CheckpointManager")
        
        # Create state dict adapter if available
        sd_adapter = SmolVLMStateDictAdapter(
            self.model_args,
            self.job_config.model.hf_assets_path
        )
        
        # Create checkpoint manager
        self.checkpointer = CheckpointManager(
            dataloader=None,  # Not needed for inference
            model_parts=[self.model],
            optimizers=None,  # Not needed for inference
            lr_schedulers=None,  # Not needed for inference
            states={},  # No training state needed
            checkpoint_config=self.job_config.checkpoint,
            sd_adapter=sd_adapter,
            base_folder=self.job_config.job.dump_folder,
            ft_manager=None,
        )
        
        # Load checkpoint
        load_step = self.job_config.checkpoint.load_step
        logger.info(f"Loading checkpoint at step: {load_step}")
        self.checkpointer.load(step=load_step)
        logger.info("Checkpoint loaded successfully")
    
    def prepare_inputs(self, prompt: str, image_path: Optional[str] = None):
        """Prepare inputs - debug version."""
        
        # Create messages
        messages = [{"user": prompt, "assistant": ""}]
        
        # Apply chat template (without tokenizing first)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            chat_template=self.chat_template,
            add_generation_prompt=True,
        )
        
        print("\n" + "="*80)
        print("FORMATTED TEXT:")
        print(repr(text))
        print("="*80)
        
        # Tokenize
        input_ids = self.tokenizer.encode(text)
        print(f"\nInput tokens ({len(input_ids)}): {input_ids[:50]}...")
        
        # Decode to verify
        decoded = self.tokenizer.decode(input_ids)
        print(f"\nDecoded input:\n{repr(decoded[:200])}...")
        
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Process image
        pixel_values = None
        patch_attention_mask = None
        
        if image_path:
            image = Image.open(image_path).resize((512, 512))
            vision_inputs = self.image_processor([image])
            pixel_values = torch.tensor(np.array(vision_inputs['pixel_values'])).squeeze()
            pixel_values = pixel_values.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
            
            patch_attention_mask = torch.tensor(vision_inputs['pixel_attention_mask'])
            patch_attention_mask = patch_attention_mask.unsqueeze(0).unsqueeze(0).to(self.device)
            
            print(f"\nImage processed. Pixel values shape: {pixel_values.shape}")
        
        return input_ids, pixel_values, patch_attention_mask
    
    @torch.no_grad()
    def generate_greedy(self, prompt: str, image_path: Optional[str] = None, max_tokens: int = 50):
        """Greedy generation with detailed logging."""
        
        print("\n" + "="*80)
        print("STARTING GENERATION")
        print("="*80)
        
        input_ids, pixel_values, patch_attention_mask = self.prepare_inputs(prompt, image_path)
        
        print(f"\nInitial input_ids shape: {input_ids.shape}")
        print(f"Starting generation loop...\n")
        
        generated = input_ids.clone()
        
        for step in range(max_tokens):
            # Forward pass
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits = self.model(
                    input_ids=generated,
                    pixel_values=pixel_values,
                    patch_attention_mask=patch_attention_mask,
                )
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Decode the token
            token_text = self.tokenizer.decode([next_token.item()])
            
            print(f"Step {step:3d} | Token: {next_token.item():5d} | Text: {repr(token_text)}")
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_id:
                print("\n*** EOS token generated ***")
                break
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for repetition
            if step > 5:
                last_tokens = generated[0, -6:].tolist()
                if len(set(last_tokens)) <= 2:
                    print(f"\n*** WARNING: Repetition detected in last 6 tokens: {last_tokens} ***")
        
        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        
        # Decode full response
        generated_ids = generated[0].tolist()
        full_text = self.tokenizer.decode(generated_ids)
        
        print(f"\nGenerated tokens: {generated_ids}")
        print(f"\nFull decoded text:\n{full_text}")
        
        # Try to extract assistant response
        if "<|im_start|>assistant" in full_text:
            response = full_text.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            response = response.strip()
            print(f"\nExtracted assistant response:\n{response}")
            return response
        
        return full_text
    
    @torch.no_grad()
    def test_forward_pass(self, prompt: str, image_path: Optional[str] = None):
        """Test a single forward pass with detailed output."""
        
        print("\n" + "="*80)
        print("TESTING FORWARD PASS")
        print("="*80)
        
        input_ids, pixel_values, patch_attention_mask = self.prepare_inputs(prompt, image_path)
        
        print(f"\nInput shapes:")
        print(f"  input_ids: {input_ids.shape}")
        if pixel_values is not None:
            print(f"  pixel_values: {pixel_values.shape}")
            print(f"  patch_attention_mask: {patch_attention_mask.shape}")
        
        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            )
        
        print(f"\nOutput logits shape: {logits.shape}")
        print(f"Logits dtype: {logits.dtype}")
        print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
        
        # Get next token predictions
        last_logits = logits[0, -1, :]
        print(f"\nLast position logits stats:")
        print(f"  Mean: {last_logits.mean().item():.4f}")
        print(f"  Std: {last_logits.std().item():.4f}")
        print(f"  Min: {last_logits.min().item():.4f}")
        print(f"  Max: {last_logits.max().item():.4f}")
        
        # Top 10 tokens
        top_logits, top_indices = torch.topk(last_logits, k=10)
        print(f"\nTop 10 predicted tokens:")
        for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
            token_text = self.tokenizer.decode([idx.item()])
            print(f"  {i+1}. Token {idx.item():5d} (logit: {logit.item():7.2f}): {repr(token_text)}")
        
        return logits


def main():
    config_manager = ConfigManager()
    job_config = config_manager.parse_args()
    
    # Initialize logger
    init_logger()
    
    logger.info("Job config loaded:")
    logger.info(f"  Model: {job_config.model.name} / {job_config.model.flavor}")
    logger.info(f"  HF assets path: {job_config.model.hf_assets_path}")
    logger.info(f"  Checkpoint folder: {job_config.checkpoint.folder}")
    logger.info(f"  Load step: {job_config.checkpoint.load_step}")
    
    # Create generator
    generator = SimpleGenerator(job_config)
    
    # Run test or generation
    if args.test_forward:
        generator.test_forward_pass(args.prompt, args.image)
    else:
        generator.generate_greedy(args.prompt, args.image, args.max_tokens)


if __name__ == "__main__":
    main()
