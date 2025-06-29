#!/bin/bash

EXP_NAME="training_setting_2"

# train stage 1
python train.py \
    --experiment_name ${EXP_NAME} \
    --train_data_root dataset_path/Training/train_setting_2 \
    --val_data_root dataset_path/Training/val \
    --train_classes car cat chair horse \
    --val_classes car cat chair horse \
    --training_stage 1 \
    --stage1_batch_size 32 \
    --stage1_epochs 5 \
    --stage1_learning_rate 0.00001 \
    --stage1_lr_decay_step 1 \
    --stage1_lr_decay_factor 0.7 \
    --WSGM_count 8 \
    --WSGM_reduction_factor 4 \
    --stage2_batch_size 32 \
    --stage2_epochs 5 \
    --stage2_learning_rate 0.0000001 \
    --stage2_lr_decay_step 1 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 4 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

# train stage 2
python train.py \
    --experiment_name ${EXP_NAME} \
    --train_data_root dataset_path/Training/train_setting_2 \
    --val_data_root dataset_path/Training/val \
    --train_classes car cat chair horse \
    --val_classes car cat chair horse \
    --training_stage 2 \
    --stage1_batch_size 32 \
    --stage1_epochs 5 \
    --stage1_learning_rate 0.00001 \
    --stage1_lr_decay_step 1 \
    --stage1_lr_decay_factor 0.7 \
    --WSGM_count 8 \
    --WSGM_reduction_factor 4 \
    --stage2_batch_size 32 \
    --stage2_epochs 5 \
    --stage2_learning_rate 0.0000001 \
    --stage2_lr_decay_step 1 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 4 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

# evaluate stage 1
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root data_path/Evaluation \
    --eval_stage 1 \
    --WSGM_count 8 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 4 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407


# evaluate stage 2
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root data_path/Evaluation \
    --eval_stage 2 \
    --WSGM_count 8 \
    --WSGM_reduction_factor 4 \
    --FAFormer_layers 4 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407