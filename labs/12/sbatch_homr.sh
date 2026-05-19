#!/bin/bash
#SBATCH --job-name=homr-competition
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=/home/babkavi/npfl138/slurm/homr_%j.out
#SBATCH --error=/home/babkavi/npfl138/slurm/homr_%j.err

# Activate virtual environment
source VENV_DIR/bin/activate


python -u labs/12/homr_competition.py --epochs=10 --threads=0