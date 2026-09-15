#!/bin/bash

module load miniconda

conda activate oidicDS

python /home/yb262/project_pi_jb948/yb262/oidicDS_proj/v7_1DIC_2d_code/train.py \
--experiment_name 'train_live_cell_COS7_mito_SSIMLoss_lr1e-4_batch_job_1DIC' \
--output_path '/home/yb262/project_pi_jb948/yb262/oidicDS_proj/v7_1DIC_2d_code/' \
--model_type 'model_cnn' \
--data_root '/home/yb262/project_pi_jb948/yb262/oidicDS_proj/data/Processed_10_2_4DIC_2d_arrange_mito_live_training_1DIC/' \
--net_G 'DuRDN4' \
--norm 'BN' \
--net_filter 64 \
--n_denselayer 6 \
--growth_rate 32 \
--lr 1e-4 \
--step_size 1 \
--gamma 0.995 \
--n_epochs 400 \
--batch_size 4 \
--n_patch_train 32 \
--patch_size_train 128 128 \
--n_patch_test 1 \
--patch_size_test 416 416 \
--n_patch_valid 1 \
--patch_size_valid 416 416 \
--eval_epochs 5 \
--snapshot_epochs 5 \
--num_workers 8 \
--gpu_ids 0