#! /bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=/data/houbb/.logs/biowulf/rpd/faf_only_model.err
#SBATCH --output=/data/houbb/.logs/biowulf/rpd/faf_only_model.out
#SBATCH --time=5-00:00:00

source /data/houbb/_venv/python39/bin/activate
module load cuDNN/8.2.1/CUDA-11.3 python/3.9 CUDA/11.3  # gcc/9.2.0

python main_single.py
#python main_dual.py
