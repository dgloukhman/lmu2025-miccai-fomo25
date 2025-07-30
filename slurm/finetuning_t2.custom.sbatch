#!/bin/bash

#SBATCH -p mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH -t 24:00:00
#SBATCH -e logs/finetuning_t2_%j.out
#SBATCH -o logs/finetuning_t2_%j.out


echo "Running test script on host $(hostname) custom finetuning"
echo ${date}

source ~/.zshrc
conda activate codebase

srun python src/finetune.py \
    --data_dir=/dss/mcmlscratch/04/ra58seq2/finetuning/data/preprocessed \
    --save_dir=/dss/dsshome1/04/ra58seq2/unet/finetuned/t2 \
    --pretrained_weights_path=/dss/dsshome1/04/ra58seq2/unet/models/FOMO60k/unet_xl_lw_dec/versions/version_0/epoch=99.ckpt \
    --model_name=unet_xl \
    --patch_size=96 \
    --taskid=2 \
    --batch_size=4 \
    --epochs=500 \
    --train_batches_per_epoch=100 \
    --new_version \
    --augmentation_preset=clone_all \
    --experiment='finetuning with larger artificial dataset'

