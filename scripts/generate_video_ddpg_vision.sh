#!/bin/bash
full_experiment_folder="videos/Go2Parkour_DDPG_demos_rnn_seqlen50_bs128_seed42_1730470417_cam"
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

    ######################
    python3 newtrain.py task=Go2ParkourRender train=SoloParkourDDPG_demos_rnn_vision headless=True test=True \
        checkpoint=./runs/Go2Parkour_DDPG_demos_rnn_seqlen50_bs128_seed42_1730470417/cleanrl_model.pt task.env.onlyForwards=True num_envs=1 \
        task.env.enableCameraSensors=True task.env.depth.use_depth=True $terrain_args task.video_save_path=${full_experiment_folder}/jump_lvl${lvl}.mp4
done
rm *core*
