#!/bin/bash
full_experiment_folder="videos/parkour_baseline_31-20-43-10"
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
    #python3 newtrain.py task=Go2ParkourRender train=SoloTerrainPPO headless=True test=True checkpoint=runs/parkour_resample_ofter_longer_traj_25s_27-01-16-00/cleanrl_model.pt task.env.onlyForwards=True num_envs=1 task.env.enableCameraSensors=True $terrain_args task.video_save_path=${full_experiment_folder}/jump_lvl${lvl}.mp4
    python3 newtrain.py task=Go2ParkourRender train=SoloTerrainPPO headless=True test=True checkpoint=runs/parkour_baseline_31-20-43-10/cleanrl_model.pt num_envs=1 task.env.enableCameraSensors=True $terrain_args task.video_save_path=${full_experiment_folder}/jump_lvl${lvl}.mp4 seed=1
done
rm *core*
