#!/bin/bash
#SBATCH -o infer.out
#SBATCH --partition=gpu
#SBATCH -J wechat
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --qos=v100_qos

source activate OOD_v1-wenjian
python inference.py > inference.out
