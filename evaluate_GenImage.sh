#!/bin/bash

EXP_NAME="GenImage"

python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root data_path/test \
    --eval_stage 2 \
    --WSGM_count 4 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407 \
    --weight data_path/GenImage.pth