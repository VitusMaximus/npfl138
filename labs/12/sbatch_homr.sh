#!/bin/bash
#SBATCH --job-name=homr-competition
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu-ffa
#SBATCH --output=/home/babkavi/npfl138/slurm/homr_%j.out
#SBATCH --error=/home/babkavi/npfl138/slurm/homr_%j.err

# Activate virtual environment
source VENV_DIR/bin/activate


python -u labs/12/homr_competition.py --epochs=150 --threads=0
