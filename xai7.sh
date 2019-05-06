#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --gres=gpu

source activate sj_torch1.0
CUDA_VISIBLE_DEVICES=3 python train_CIDER_Opt-gozii.py --batch_size 10 --model_path /mnt/server5_hard1/seungjun/XAI/BAN_XAI/models_BUTD/standard_vocab/ --isAdaptive True --checkpoint_dir model-75.pth 
source deactivate




# 여러명이서 서버를 사용하는데 누가 쓰고 있을 때 거기에 다른 사람이 돌리면 기존에 돌리던 사람도 같이꺼지니까 그걸 방지하기위한 파일임. 요청한 순서대로 배분해줌 