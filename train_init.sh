#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Project path and config
CONFIG=configs/train_init.yaml
LOGDIR=logs/train_init

mkdir -p $LOGDIR
WANDB_SAVE_DIR=wandb

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE=$LOGDIR/log_${TIMESTAMP}.txt

echo "CONFIG="$CONFIG
echo "LOGFILE="$LOGFILE

torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --no-one-logger 2>&1 | tee $LOGFILE