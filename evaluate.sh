#!/bin/bash

EXP_NAME="training_setting_1"
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root data_path/Evaluation \
    --eval_stage 2 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407 \
    --weight data_path/training_setting_1.pth


# EXP_NAME="training_setting_2"
# python evaluate.py \
#     --experiment_name ${EXP_NAME} \
#     --eval_data_root data_path/Evaluation \
#     --eval_stage 2 \
#     --WSGM_count 8 \
#     --WSGM_reduction_factor 4 \
#     --FAFormer_layers 4 \
#     --FAFormer_reduction_factor 1 \
#     --FAFormer_head 2 \
#     --num_workers 4 \
#     --seed 3407 \
#     --weight data_path/training_setting_2.pth

# EXP_NAME="training_setting_3"
# python evaluate.py \
#     --experiment_name ${EXP_NAME} \
#     --eval_data_root data_path/Evaluation \
#     --eval_stage 2 \
#     --WSGM_count 8 \
#     --WSGM_reduction_factor 4 \
#     --FAFormer_layers 16 \
#     --FAFormer_reduction_factor 1 \
#     --FAFormer_head 2 \
#     --num_workers 4 \
#     --seed 3407 \
#     --weight data_path/training_setting_3.pth


