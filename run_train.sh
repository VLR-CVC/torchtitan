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
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/multimodal/debug_train_config.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

torchrun --nproc_per_node=1 --local-ranks-filter ${LOG_RANK} --role rank \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"
