#!/bin/bash
policy_path=./runs/parkour_baseline_01-03-59-04/cleanrl_model.pt

python newtrain.py task=Go2Parkour train=SoloParkourDDPG_demos_generate headless=True task.env.numEnvs=256 task.env.enableCameraSensors=True task.env.depth.use_depth=True task.env.terrain.maxInitMapLevel=9 train.params.config.target_policy_path=${policy_path}
