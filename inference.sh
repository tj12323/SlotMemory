#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

N=5

# ===== 在这里改你要覆盖的值 =====
OUTPUT_FOLDER="videos/long30"
NUM_OUTPUT_FRAMES=120
# ================================

for (( i=1; i<=N; i++ )); do
    SEED=$((RANDOM % 10000 + 100))
    PORT=$((RANDOM % 10000 + 20000))

    TMP_CFG=$(mktemp /tmp/interactive_cfg_XXXX.yaml)

    cp configs/inference.yaml "${TMP_CFG}"

    sed -i "s/^seed:.*/seed: ${SEED}/" "${TMP_CFG}"
    sed -i "s|^output_folder:.*|output_folder: ${OUTPUT_FOLDER}|" "${TMP_CFG}"
    sed -i "s/^num_output_frames:.*/num_output_frames: ${NUM_OUTPUT_FRAMES}/" "${TMP_CFG}"

    torchrun \
        --nproc_per_node=2 \
        --master_port=${PORT} \
        inference.py \
        --config_path "${TMP_CFG}"

    rm -f "${TMP_CFG}"

    echo "Finished run $i, seed=${SEED}, frames=${NUM_OUTPUT_FRAMES}, out=${OUTPUT_FOLDER}"
done