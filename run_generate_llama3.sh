#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_1b.toml"}
INFERENCE_FILE=${INFERENCE_FILE:-"torchtitan.generate_llama3"}


CUDA_VISIBLE_DEVICES=2 \
NCCL_P2P_DISABLE=1 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${INFERENCE_FILE} --job.config_file ${CONFIG_FILE} "$@"
