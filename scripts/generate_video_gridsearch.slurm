#!/bin/bash
#SBATCH --job-name=embeddedvisi      # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=20           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=02:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/logs_%j.out # output file name
#SBATCH --error=logs/logs_%j.err  # error file name
#SBATCH --array=1-1

declare -A hyperparameters=(
    ["task.env.learn.constraints_CaT.feetAirTimeConstraint"]="15 15 15 15 15 20 20 20 20 25 25 25 30 30"
    ["task.env.learn.constraints_CaT.feetMaxAirTimeConstraint"]="15 20 25 30 35 20 25 30 35 25 30 35 30 35"
)

hyperparam_args=""
for key in "${!hyperparameters[@]}"; do
    values=(${hyperparameters[$key]})
    num_values=${#values[@]}
    # Select the value based on the ID (assuming ID starts from 1)
    selected_value=${values[$(( (SLURM_ARRAY_TASK_ID - 1) % num_values ))]}
    # selected_value=${values[$((SLURM_ARRAY_TASK_ID - 1 >= num_values ? num_values - 1 : SLURM_ARRAY_TASK_ID - 1))]}
    hyperparam_args+=" $key=$selected_value"
done

cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
conda activate rlgpu
set -x

# After constructing hyperparam_args
# Remove leading space if any
clean_name="${hyperparam_args# }"

# Convert hyperparam_args to a clean file name
clean_name=$(echo "$clean_name" | tr ' ' '_' | tr -cd '[:alnum:]_-')

python3 newtrain.py task=Go2Terrain train=SoloTerrainPPO  headless=True train.params.config.max_epochs=1000 $hyperparam_args train.params.config.name="$clean_name"
#python3 newtrain.py task=Go2Terrain train=SoloTerrainPPO  headless=True $hyperparam_args train.params.config.name="$clean_name"
