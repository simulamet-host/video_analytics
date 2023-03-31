#!/bin/bash
#SBATCH --job-name="feature_extractor"
#SBATCH --partition=dgx2q
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output="./logs/%j-%x-stdout.txt"
#SBATCH --error="./logs/%j-%x-stderr.txt"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="faiga@simula.no"

echo "This is Faiga's process running on $(hostname)"
eval "$(conda shell.bash hook)"
conda activate video
cd ..
srun python feature_extractor.py  --dataset_name 'action_recognition'  --mode 'train' 
echo 'Done!'