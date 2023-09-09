#!/bin/bash
#SBATCH --job-name="dinov2"
#SBATCH --partition=habanaq
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output="./logs/%j-%x-stdout.txt"
#SBATCH --error="./logs/%j-%x-stderr.txt"
set -e 
echo "This is Faiga's process running on $(hostname)"
eval "$(conda shell.bash hook)"
conda activate video
cd ..
srun python dinov2.py
echo 'Done!'
