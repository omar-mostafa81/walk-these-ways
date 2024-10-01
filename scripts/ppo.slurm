#!/bin/bash
#SBATCH --job-name=embeddedvisi      # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=20           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/logs_%j.out # output file name
#SBATCH --error=logs/logs_%j.err  # error file name

cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
conda activate rlgpu
set -x

python3 newtrain.py task=Go2Parkour train=SoloTerrainPPO  headless=True train.params.config.max_epochs=8000 train.params.config.name="urdf_limits_ar120_fc120_zinit44"
