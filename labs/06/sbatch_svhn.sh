#!/bin/bash
#SBATCH --job-name=svhn-detector
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --output=logs/svhn_%j.out
#SBATCH --error=logs/svhn_%j.err

# Activate virtual environment
source /home/vitek/School/npfl138/VENV_DIR/bin/activate

# Navigate to the project directory
cd /home/vitek/School/npfl138

# Run the training script
python labs/06/svhn_competition.py
