#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

N=2

for (( i=1; i<=N; i++ )); do
    SEED=$((RANDOM % 10000 + 100))
    PORT=$((RANDOM % 10000 + 20000))

    TMP_CFG=$(mktemp /tmp/interactive_cfg_XXXX.yaml)

    cp configs/interactive_inference.yaml "${TMP_CFG}"

    sed -i "s/^seed:.*/seed: ${SEED}/" "${TMP_CFG}"

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        --master_port=${PORT} \
        interactive_inference.py \
        --config_path "${TMP_CFG}"

    rm -f "${TMP_CFG}"

    echo "Finished run $i, seed=${SEED}"
done