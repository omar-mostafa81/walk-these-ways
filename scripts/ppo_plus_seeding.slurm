#!/bin/bash
#SBATCH --job-name=action_improve_seeds      # job name
#SBATCH --ntasks=1                           # number of tasks
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --gres=gpu:1                         # number of GPUs per node
#SBATCH --cpus-per-task=20                   # number of CPU cores per task
#SBATCH --hint=nomultithread                 # use physical cores, not logical
#SBATCH --time=02:00:00                      # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/logs_%j.out            # output file name
#SBATCH --error=logs/logs_%j.err             # error file name
#SBATCH --array=1-5                          # array of tasks (5 seeds)

cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
conda activate rlgpu
set -x

# Fixed hyperparameters
N=5
sigma=0.3
alpha=0.2
num_improvement_steps=3

# Seed values (from 1 to 5)
seed=$SLURM_ARRAY_TASK_ID

# Construct hyperparameter arguments for the training script
hyperparam_args="train.params.config.N=$N train.params.config.sigma=$sigma train.params.config.alpha=$alpha train.params.config.num_improvement_steps=$num_improvement_steps seed=$seed"

# Generate a clean name for logging and identification
clean_name="N${N}_sigma${sigma}_alpha${alpha}_steps${num_improvement_steps}_seed${seed}"

# Run the training script with the specified hyperparameters
python3 newtrain.py task=Go2Terrain train=SoloTerrainPPO+ headless=True train.params.config.max_epochs=1000 $hyperparam_args train.params.config.name="$clean_name"
