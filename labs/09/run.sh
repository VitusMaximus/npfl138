#!/bin/bash
#SBATCH -J speechRec
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH -G1
#SBATCH -c1
#SBATCH -o ./predictions2.out
#SBATCH -e ./speech2.err

~/VENV_DIR/bin/python ./speech_recognition.py