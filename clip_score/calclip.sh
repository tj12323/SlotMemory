#!/bin/sh
set -x
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python clip_score/calculateqwen3vl.py \
    --config_path configs/interactive_inference.yaml \
    --video_dir videos/interactive \
    --max_frames_per_segment 10