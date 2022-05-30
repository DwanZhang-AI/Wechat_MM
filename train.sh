#!/bin/bash
#SBATCH -o node4_smallbs_0.5.out
#SBATCH --partition=gpu
#SBATCH -J wechat
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --qos=v100_qos

source activate torch
python main.py
# python data_helper.py
