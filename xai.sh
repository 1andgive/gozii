#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --gres=gpu

source activate sj_torch1.0
python train_UNION_BUTD.py --t_method mean --batch_size 100 --isAdaptive True --model_path models_BUTD/standard_vocab/
source deactivate

