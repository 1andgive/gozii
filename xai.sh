#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --gres=gpu:2

source activate sj_torch1.0
python explain_ban_coco.py --isBUTD True --t_method mean --hsc_epoch 20
source deactivate

