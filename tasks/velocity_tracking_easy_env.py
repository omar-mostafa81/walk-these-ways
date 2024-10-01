from isaacgym import gymutil, gymapi
import torch
from params_proto import Meta, PrefixProto
from typing import Union

from go2_gym.envs.base.legged_robot import LeggedRobot
from go2_gym.envs.base.legged_robot_config import Cfg
import gym
from gym import spaces
import numpy as np

def update_params_proto(cfg_class, cfg_dict):
    for key, value in cfg_dict.items():
        if hasattr(cfg_class, key):
            attr = getattr(cfg_class, key)
            if isinstance(value, dict):
                # If the attribute is a nested configuration (PrefixProto), recurse
                if isinstance(attr, Meta):
                    update_params_proto(attr, value)
                else:
                    setattr(cfg_class, key, value)
            else:
                setattr(cfg_class, key, value)
        else:
            # If the attribute doesn't exist, you can choose to set it or raise an error
            setattr(cfg_class, key, value)

class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(cfg["sim"], sim_params)

        update_params_proto(Cfg, cfg)
        cfg = Cfg

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        self.action_space = spaces.Box(np.ones(cfg.env.num_actions) * -1., np.ones(cfg.env.num_actions) * 1.)
        self.observation_space = spaces.Box(np.ones(cfg.env.num_observations) * -np.Inf, np.ones(cfg.env.num_observations) * np.Inf)

        #sim_params = gymapi.SimParams()
        #gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)


    def step(self, actions):
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        self.extras["true_dones"] = self.reset_buf
        return self.obs_buf, self.rew_buf, torch.zeros_like(self.reset_buf), self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

