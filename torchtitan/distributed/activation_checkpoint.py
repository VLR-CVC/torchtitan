# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torchtitan.config.job_config import ActivationCheckpoint as ACConfig
from torchtitan.tools.logging import logger, warn_once


_layer_sac_count = 0


def _apply_layer_sac(
    module: nn.Module, ac_config: ACConfig, *, ac_freq: int | None = None
) -> nn.Module:
    """Apply layer selective activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply layer selective activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.

    Returns:
        nn.Module: The module with layer selective activation checkpointing applied.
    """
    global _layer_sac_count
    _layer_sac_count += 1
    ac_freq = int(ac_config.selective_ac_option) if ac_freq is None else ac_freq
    if not ac_freq or _layer_sac_count % ac_freq == 0:
        return ptd_checkpoint_wrapper(
            module, preserve_rng_state=False, early_stop=ac_config.early_stop
        )
    else:
        return module


def _apply_op_sac(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    save_list: set[torch._ops.OpOverload],
) -> nn.Module:
    """Apply selective activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply selective activation checkpointing to.
        ac_config (ActivationCheckpoint): The activation checkpointing config.
        base_fqn (str, optional): The base fqn of the module. Defaults to None.
        save_list (set[torch._ops.OpOverload]): The list of ops to save when selective
            activation checkpointing is used.

    Returns:
        nn.Module: The module with selective activation checkpointing applied.

    """
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    mm_recompute_shapes = set()
    if len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
        for module_fqn, submod in module.named_modules():
            fqn = module_fqn
            if base_fqn is not None:
                fqn = f"{base_fqn}.{module_fqn}"
            if not any(
                filter_fqn in fqn
                for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
            ):
                continue
            if not isinstance(submod, nn.Linear):
                raise ValueError(
                    "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                    f"a nn.Linear, but got: {submod}"
                )
            out_f, in_f = submod.weight.shape
            mm_recompute_shapes.add((in_f, out_f))
        logger.debug(
            f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
        )

    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                if args[1].shape in mm_recompute_shapes:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in save_list and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
        early_stop=ac_config.early_stop,
    )


def _apply_full_ac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    return ptd_checkpoint_wrapper(
        module, preserve_rng_state=False, early_stop=ac_config.early_stop
    )


def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    model_compile_enabled: bool = False,
    use_flex_attn: bool = False,
    save_list: set[torch._ops.OpOverload] | None = None,
) -> nn.Module:
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return _apply_full_ac(module, ac_config)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )

    if use_op_sac:
        save_list = save_list or set()
        if use_flex_attn:
            return _apply_op_sac_to_transformer_block_with_flex(
                module,
                ac_config,
                base_fqn=base_fqn,
                model_compile_enabled=model_compile_enabled,
                save_list=save_list,
            )
        else:
            return _apply_op_sac(
                module, ac_config, base_fqn=base_fqn, save_list=save_list
            )

    return _apply_layer_sac(module, ac_config)


def _apply_op_sac_to_transformer_block_with_flex(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    model_compile_enabled: bool = False,
    save_list: set[torch._ops.OpOverload],
) -> nn.Module:
    warn_once(
        logger,
        (
            "Flex Attention requires compilation for good performance.\n"
            "Thus, torch.compile is always used for Flex Attention, "
            "regardless of the compile.enable flag.\n"
            "However, when selective activation checkpointing (SAC) is enabled, "
            "torch.compile may be invalidated:\n"
            "1. If compile.enable is False, SAC will ignore any torch.compile "
            "inside the SAC region.\n"
            "2. If compile.enable is True but the transformer block contains a MoE module.\n\n"
            "For both cases, SAC will not be directly applied to the TransformerBlock.\n"
            "   - For case 1: SAC will be used for MoE and FeedForward modules, "
            "while full AC will be used for the Attention module.\n"
            "   - For case 2: SAC will be used for both MoE and Attention modules, "
            "but they will be wrapped independently.\n"
        ),
    )
    if True:
        if (moe := getattr(module, "moe", None)) is not None:
            moe = _apply_op_sac(
                moe,
                ac_config,
                base_fqn=f"{base_fqn}.moe" if base_fqn else "moe",
                save_list=save_list,
            )
            attention = _apply_full_ac(module.attention, ac_config)
            """
            attention = _apply_op_sac(
                module.attention,
                ac_config,
                base_fqn=f"{base_fqn}.attention" if base_fqn else "attention",
                save_list=save_list,
            )
            """
            module.register_module("moe", moe)
            module.register_module("attention", attention)
        else:
            module = _apply_full_ac(module, ac_config)
            """
            module = _apply_op_sac(
                module,
                ac_config,
                base_fqn=base_fqn,
                save_list=save_list,
            )
            """
    else:
        for name in ("feed_forward", "moe"):
            if (m := getattr(module, name, None)) is not None:
                module.register_module(
                    name,
                    _apply_op_sac(
                        m,
                        ac_config,
                        base_fqn=f"{base_fqn}.{name}" if base_fqn else name,
                        save_list=save_list,
                    ),
                )
        attention = _apply_full_ac(module.attention, ac_config)
        module.register_module("attention", attention)

    return module


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    use_flex_attn: bool = False,
    save_list: set[torch._ops.OpOverload] | None = None,
) -> None:
    """Apply activation checkpointing to the model.

    Note that SAC, Flex Attention and model compilation have some conflicts.
    We explicitly ask the user to pass these configs to warn if there are conflicts.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        ac_config (ActivationCheckpoint): The activation checkpointing config.
        model_compile_enabled (bool): Whether torch.compile is enabled for the model.
        use_flex_attn (bool): Whether flex attention is enabled for the model.
        save_list (set[torch._ops.OpOverload]): The list of ops to save when selective
            activation checkpointing is used.
    Returns:
        None
    """

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block,
            ac_config,
            base_fqn=f"layers.{layer_id}",
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
            save_list=save_list,
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
