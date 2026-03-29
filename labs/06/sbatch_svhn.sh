#!/bin/bash
#SBATCH --job-name=svhn-detector
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=/home/babkavi/npfl138/slurm/svhn_%j.out
#SBATCH --error=/home/babkavi/npfl138/slurm/svhn_%j.err

# Activate virtual environment
source VENV_DIR/bin/activate

# Run the training script
python -u labs/06/svhn_competition.py --epochs=100 --threads=0
