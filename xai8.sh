#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --gres=gpu

source activate sj_torch1.0
python train_UNION_BUTD.py --num_epochs 100 --checkpoint_dir model-39.pth
source deactivate