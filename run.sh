#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=logs/wigglythings_%j.out
#SBATCH --error=logs/wigglythings_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

source ~/miniconda/etc/profile.d/conda.sh

# export LIBMATHDX_LOG_LEVEL=5
export LC_ALL=C
export LANG=C
export PYTHONUNBUFFERED=1

conda activate gpuenv
python src/main.py