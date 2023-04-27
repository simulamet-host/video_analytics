#!/bin/bash
#SBATCH --job-name="e2evideo"
#SBATCH --partition=dgx2q
#SBATCH --time=0-00:00:10
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
srun e2epipline.sh complete_pipeline
echo 'Done!'
