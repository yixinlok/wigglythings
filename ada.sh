#!/bin/bash
#SBATCH --job-name=ada
#SBATCH --output=logs/wigglythings_%j.out
#SBATCH --error=logs/wigglythings_%j.err
#SBATCH --nodelist=calypso
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

source ~/miniconda/etc/profile.d/conda.sh

# export LIBMATHDX_LOG_LEVEL=5
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONUNBUFFERED=1

conda activate gpuenv
python src/main.py

cd /scratch/thirty/yixinlok
ls