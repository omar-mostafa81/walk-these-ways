#!/bin/bash
full_experiment_folder="videos/urdf_limits_ar120_fc120_zinit44_27-23-42-48"
mkdir $full_experiment_folder

# Loop over levels from 0 to 9
for lvl in {0..9}
do
    terrain_args="task.env.terrain.terrainProportions.gap_parkour=0.0"
    terrain_args+=" task.env.terrain.terrainProportions.jump_parkour=1.0"
    terrain_args+=" task.env.terrain.terrainProportions.stairs_parkour=0.0"
    terrain_args+=" task.env.terrain.terrainProportions.hurdle_parkour=0.0"
    terrain_args+=" task.env.terrain.terrainProportions.crawl_parkour=0.0"
    terrain_args+=" task.env.terrain.minInitMapLevel=$lvl"
    terrain_args+=" task.env.terrain.maxInitMapLevel=$lvl"

    #######################################################################################################
    python3 newtrain.py task=Go2ParkourRender train=SoloTerrainPPO headless=True test=True checkpoint=runs/urdf_limits_ar120_fc120_zinit44_27-23-42-48/cleanrl_model.pt task.env.onlyForwards=True num_envs=1 task.env.enableCameraSensors=True $terrain_args task.video_save_path=${full_experiment_folder}/jump_lvl${lvl}.mp4
done
rm *core*
