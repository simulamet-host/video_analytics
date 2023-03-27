#!/bin/bash
#SBATCH --job-name="action_recognition"
#SBATCH --partition=hgx2q
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --output="./logs/%j-%x-stdout.txt"
#SBATCH --error="./logs/%j-%x-stderr.txt"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="faiga@simula.no"

echo "This is Faiga's process running on $(hostname)"
cd ..
srun python action_recognition.py \
--images_array './results/all_images.npz' \
--data_folder './data/images_ucf101/' \
--no_classes 101 \

echo 'Done!'
