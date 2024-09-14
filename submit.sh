#! /bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=/data/houbb/.logs/biowulf/rpd/faf_cfp_model_nopp_1024.err
#SBATCH --output=/data/houbb/.logs/biowulf/rpd/faf_cfp_model_nopp_1024.out
#SBATCH --time=5-00:00:00

module load cuDNN/8.2.1/CUDA-11.3 python/3.9 CUDA/11.3  # gcc/9.2.0
source /data/houbb/_venv/tfenv/bin/activate

export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/CUDA/11.3.0/'
export WANDB_API_KEY='REDACTED'

python train.py --config configs/default.json
