#!/bin/bash
#SBATCH --job-name=action_improve_grid_search      # job name
#SBATCH --ntasks=1                                 # number of tasks
#SBATCH --ntasks-per-node=1                        # number of tasks per node
#SBATCH --gres=gpu:1                               # number of GPUs per node
#SBATCH --cpus-per-task=20                         # number of CPU cores per task
#SBATCH --hint=nomultithread                       # use physical cores, not logical
#SBATCH --time=02:00:00                            # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/logs_%j.out                  # output file name
#SBATCH --error=logs/logs_%j.err                   # error file name
#SBATCH --array=1-36                               # array of tasks (total combinations)

cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
conda activate rlgpu
set -x

# Define hyperparameter values
N_values=(5 10 20)
sigma_values=(0.2 0.3)
alpha_values=(0.05 0.2)
num_improvement_steps_values=(2 3 4)

# Total combinations
total_N=${#N_values[@]}                  # 3
total_sigma=${#sigma_values[@]}          # 2
total_alpha=${#alpha_values[@]}          # 2
total_steps=${#num_improvement_steps_values[@]}  # 3

total_combinations=$((total_N * total_sigma * total_alpha * total_steps))  # 36

# Ensure SLURM_ARRAY_TASK_ID is within bounds
if [ $SLURM_ARRAY_TASK_ID -gt $total_combinations ]; then
    echo "Error: SLURM_ARRAY_TASK_ID exceeds total number of combinations"
    exit 1
fi

# Calculate indices for each hyperparameter
task_id=$((SLURM_ARRAY_TASK_ID - 1))

num_alpha_steps=$((total_alpha * total_steps))         # 2 * 3 = 6
num_sigma_alpha_steps=$((total_sigma * num_alpha_steps))  # 2 * 6 = 12

N_index=$((task_id / num_sigma_alpha_steps))
remaining=$((task_id % num_sigma_alpha_steps))

sigma_index=$((remaining / num_alpha_steps))
remaining=$((remaining % num_alpha_steps))

alpha_index=$((remaining / total_steps))
num_improvement_steps_index=$((remaining % total_steps))

# Retrieve hyperparameter values based on indices
N=${N_values[N_index]}
sigma=${sigma_values[sigma_index]}
alpha=${alpha_values[alpha_index]}
num_improvement_steps=${num_improvement_steps_values[num_improvement_steps_index]}

# Construct hyperparameter arguments for the training script
hyperparam_args="train.params.config.N=$N train.params.config.sigma=$sigma train.params.config.alpha=$alpha train.params.config.num_improvement_steps=$num_improvement_steps"

# Generate a clean name for logging and identification
clean_name="N${N}_sigma${sigma}_alpha${alpha}_steps${num_improvement_steps}"

# Run the training script with the specified hyperparameters
python3 newtrain.py task=Go2Terrain train=SoloTerrainPPO+ headless=True train.params.config.max_epochs=1000 $hyperparam_args train.params.config.name="$clean_name"
