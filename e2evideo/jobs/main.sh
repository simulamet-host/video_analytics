#!/bin/bash
#SBATCH --job-name="e2evideo"
#SBATCH --partition=dgx2q
#SBATCH --time=0-00:04:00
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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/
cd ..
srun e2epipline.sh complete_pipeline
echo 'Done!'
