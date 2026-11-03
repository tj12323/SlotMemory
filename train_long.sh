#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
# Project path and config
CONFIG=configs/train_long.yaml
LOGDIR=logs/train_long
mkdir -p $LOGDIR
echo "CONFIG="$CONFIG

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE=$LOGDIR/log_${TIMESTAMP}.txt

torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --disable-wandb \
  --no-one-logger  2>&1 | tee $LOGFILE