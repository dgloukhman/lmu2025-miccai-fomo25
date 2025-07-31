#!/bin/bash

#SBATCH -p mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -t 2-00:00:00
#SBATCH -e logs/pretraining_%j.out
#SBATCH -o logs/pretraining_%j.out


echo "Running test script on host $(hostname)"
echo ${date}

source ~/.zshrc
conda activate codebase

srun python src/pretrain.py \
    --save_dir=~/unet \
    --pretrain_data_dir=/dss/mcmlscratch/04/ra58seq2/finetuning/data/preprocessed/Task002_FOMO2 \
    --model_name=unet_xl_lw_dec \
    --patch_size=96 \
    --batch_size=4 \
    --epochs=100 \
    --warmup_epochs=5 \
    --num_workers=96 \
    --num_devices=4 \
    --augmentation_preset=all \
    --model_type=joint_mae_seg \
    --seg_loss_weight=0.4 \
    --pretrained_encoder_path=/dss/dsshome1/04/ra58seq2/unet/models/FOMO60k/unet_xl_lw_dec/versions/version_0/epoch=99.ckpt