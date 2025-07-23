#!/bin/bash

#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --mincpu=50
#SBATCH -t 9:00:00
#SBATCH -e logs/preprocessing_%j.out
#SBATCH -o logs/preprocessing_%j.out

echo "Running test script on host $(hostname)"
echo ${date}

source ~/.zshrc
conda activate codebase

srun python src/data/fomo-60k/preprocess.py --in_path=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra58seq2/fomo60k/fomo-60k --out_path=/dss/mcmlscratch/04/ra58seq2/preprocessed2
