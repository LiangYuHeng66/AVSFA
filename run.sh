#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python finetune.py \
--name finetune_CUHK-PEDES \
--img_aug \
--batch_size 128 \
--dataset_name $DATASET_NAME \
#--loss_names 'TAL+id+VTC+FGSM' \
--loss_names 'TAL+id' \
--num_epoch 30 \
--root_dir ../../data \
--ot_gamma 1.0 \
--img_k_ratio 0.7 \
--i_feats_fg_gamma 0.1 \
--finetune The_pretrained_checkpoint
