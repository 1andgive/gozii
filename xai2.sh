#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --gres=gpu

source activate sj_torch1.0
python train_UNION_BUTD.py --t_method mean --model_path models_BUTD/standard_vocab/ --vocab_path data/vocab_standard_train_val.pkl --batch_size 360 --hidden_size 1000 --embed_size 1000 --paramH 512
source deactivate

