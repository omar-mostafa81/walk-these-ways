#!/bin/bash
# replace with the correct path to the privileged experience folder
demo_path="./runs/Go2Parkour_DDPG_demos_generate_1730462312/" # folder which contains rb_demos.pkl

python newtrain.py task=Go2Parkour train=SoloParkourDDPG_demos_rnn_vision headless=True task.env.numEnvs=256 task.env.enableCameraSensors=True task.env.depth.use_depth=True train.params.config.demos_rb_path=${demo_path}
