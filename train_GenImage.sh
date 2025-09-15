#!/bin/bash

EXP_NAME="GenImage"

# train stage 1
python train.py \
    --experiment_name ${EXP_NAME} \
    --train_data_root data_path/train \
    --val_data_root dataset_path/val \
    --train_classes stable_diffusion_v_1_4 \
    --val_classes stable_diffusion_v_1_4 \
    --training_stage 1 \
    --stage1_batch_size 16 \
    --stage1_epochs 20 \
    --stage1_learning_rate 0.00005 \
    --stage1_lr_decay_step 2 \
    --stage1_lr_decay_factor 0.7 \
    --WSGM_count 4 \
    --WSGM_reduction_factor 4 \
    --stage2_batch_size 16 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.0000025 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

# train stage 2
python train.py \
    --experiment_name ${EXP_NAME} \
    --train_data_root dataset_path/train \
    --val_data_root dataset_path/val \
    --train_classes stable_diffusion_v_1_4 \
    --val_classes stable_diffusion_v_1_4 \
    --training_stage 2 \
    --stage1_batch_size 16 \
    --stage1_epochs 20 \
    --stage1_learning_rate 0.00005 \
    --stage1_lr_decay_step 2 \
    --stage1_lr_decay_factor 0.7 \
    --WSGM_count 4 \
    --WSGM_reduction_factor 4 \
    --stage2_batch_size 16 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.0000025 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

# evaluate stage 1
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root dataset_path/test \
    --eval_stage 1 \
    --WSGM_count 4 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407


# evaluate stage 2
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root dataset_path/test \
    --eval_stage 2 \
    --WSGM_count 4 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407