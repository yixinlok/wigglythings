#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=logs/wigglythings_%j.out
#SBATCH --error=logs/wigglythings_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

export LC_ALL=C
export LANG=C
export PYTHONUNBUFFERED=1

source ~/miniconda/etc/profile.d/conda.sh
conda activate gpuenv
python src/main.py