#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -q wildfire
#SBATCH -t 0-03:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwinzurk@asu.edu

# Purge modules to ensure consistent environments
module purge
# Load required modules for job's environment
module load anaconda3/5.3.0
source activate serengeti

# train model
python run.py --dataset "CCT" \
              --runs 1 \
              --batch_size 64 \
              --arch "resnet50"  \
              --task "classification" \
              --level "species" \
