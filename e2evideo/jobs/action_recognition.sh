#!/bin/bash
#SBATCH --job-name="action_recognition"
#SBATCH --partition=hgx2q
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=./logs/%j-%x-stdout.txt
#SBATCH --error=./logs/%j-%x-stderr.txt

echo "This is Faiga's process running on $(hostname)"
cd ..
srun python action_recognition.py
echo 'Done!'
