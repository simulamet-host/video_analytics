#!/bin/bash
#SBATCH --job-name="feature_extractor"
#SBATCH --partition=hgx2q
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output="./logs/%j-%x-stdout.txt"
#SBATCH --error="./logs/%j-%x-stderr.txt"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="faiga@simula.no"

echo "This is Faiga's process running on $(hostname)"
cd ..
srun python feature_extractor.py --dataset_name 'action_recognition'
echo 'Done!'