#!/bin/bash

# Loop over all matching directories in ./runs
for dir in ./runs/taskenvlearnconstraints_CaTfeetAirTimeConstraint*_taskenvlearnconstraints_CaTfeetMaxAirTimeConstraint*
do
    # Extract the base directory name without the path
    base_dir=$(basename "$dir")

    # Set the full experiment folder
    full_experiment_folder="videos/$base_dir"
    mkdir -p $full_experiment_folder

    # Loop over levels from 0 to 9

    #######################################################################################################
    python3 newtrain.py task=Go2TerrainRender train=SoloTerrainPPO headless=True test=True \
        checkpoint=$dir/cleanrl_model.pt task.env.onlyForwards=True num_envs=1 \
        task.env.enableCameraSensors=True \
        task.video_save_path=${full_experiment_folder}/flat.mp4
    rm *core*
done
