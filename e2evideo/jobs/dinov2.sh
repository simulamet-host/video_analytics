#!/bin/bash
#SBATCH --job-name="feature_extraction"
#SBATCH --partition=fpgaq
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output="./logs/%j-%x-stdout.txt"
#SBATCH --error="./logs/%j-%x-stderr.txt"
set -e 
echo "This is Faiga's process running on $(hostname)"
eval "$(conda shell.bash hook)"
conda activate e2evideo
source /home/faiga/.cache/pypoetry/virtualenvs/e2evideo-GfxAQqu6-py3.10/bin/activate
cd ..
srun python feature_extractor_new.py
echo 'Done!'
