#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --gres=gpu

source activate sj_torch1.0
python train_CIDER_Opt.py --batch_size 350
source deactivate