#!/bin/bash

EXP_NAME="training_setting_1"

python inference.py \
    --experiment_name "${EXP_NAME}" \
    --input_dir "" \
    --eval_stage 2 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407 \
    --weight data_path/training_setting_1.pth
