#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --gres=gpu:2

source activate sj_torch1.0
python train_UNION_BUTD.py --t_method mean --model_num 2 --batch_size 1024
source deactivate

