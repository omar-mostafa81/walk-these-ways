import gym
import numpy as np
import time
import torch
from collections import deque
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

class ActuatorModelWrapper(gym.Wrapper):
    def __init__(self, env, delay = 1, history_duration = 10, min_data_points = 2, alpha = 0.9, mu_v = 0.1, Fs = 0.3, Vs = 0.5, temperature=0.1):
        """ Initialize the actuator model wrapper
        Args:
            env: main environment
            delay (float): Controls the amount of delay applied to the actions.
            history_duration (float): Controls the number of actions used for interpolation.
            min_data_points (int): Minimum number of data points for interpolation.
            alpha (float): Controls the low pass filter.
            mu_v (float): Viscous friction coefficient.
            Fs (float): Static friction strength.
            temperature (float): parameter for the soft sign function.
        """
        super().__init__(env)
        self.env = env 
        self.delay = delay 
        self.alpha = alpha 
        self.mu_v = mu_v
        self.Fs = Fs
        self.Vs = Vs
        self.temperature = temperature
        self.min_data_points = min_data_points 
        # Save actions/time for interpolation to simulate delay
        self.current_time = time.perf_counter() # More accurate for short durations
        self.history_duration = history_duration
        self.time_buffers = []
        # self.action_buffers = [deque() for _ in range(self.env.num_envs)]
        self.action_buffers = []
        self.prev_actions = torch.zeros((self.env.num_envs, self.env.action_space.shape[0]), dtype=torch.float32, device=self.env.device)

    def reset(self):
        obs = super().reset() # LeggedRobot only has reset_idx(), which is called by VelocityTrackingEasyEnv with idx to all env
        
        # Reset buffers/variables
        self.time_buffers = []
        # self.action_buffers = [deque() for _ in range(self.env.num_envs)]
        self.action_buffers = []
        self.prev_actions = torch.zeros((self.env.num_envs, self.env.action_space.shape[0]), dtype=torch.float32, device=self.env.device)

        return obs
    
    def step(self, actions):
        """ Simulate delay, bandwidth, and friction before sending actions to the simulator
        Args: actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        print("initial actions: ", actions)
        # 1) Delay the actions
        self.current_time = time.perf_counter()
        self.time_buffers.append(self.current_time) 
        self.action_buffers.append(actions)
        ## 1.1) Remove old actions --> remove actions older than "history_duration" seconds
        while self.time_buffers and self.time_buffers[0] < (self.current_time - self.history_duration):
            self.time_buffers.pop(0)
            self.action_buffers.pop(0)
        ## 1.2) Get the delayed actions
        delayed_actions = self.apply_delay() if len(self.time_buffers) > self.min_data_points else actions 
        print("delayed actions: ", delayed_actions)
        # 2) Apply friction
        dq = self.env.dof_vel.cpu().numpy() # Get the current velocity 
        delayed_actions = delayed_actions - self.compute_friction(dq) 
        print("friction actions: ", delayed_actions)
        # 3) Apply LowPassFilter based on bandwidth
        new_actions = self.apply_LPF(delayed_actions)
        print("final actions: ", delayed_actions)

        # Return modified actions to the env
        self.prev_actions = new_actions
        return self.env.step(new_actions)
    
    def apply_delay(self):
        # Build tensors
        # length = len(self.time_buffers)
        # channels = self.env.num_envs
        x = torch.tensor(self.time_buffers, dtype=torch.float32, device=self.env.device)  # dim: (length)
        t = torch.tensor(self.current_time - self.delay, dtype=torch.float32, device=self.env.device)  # target eval time for each env

        interpolated_actions = []
        for i in range(self.env.action_space.shape[0]):
            # for each time stamp, get the action[i] for each env
            y = torch.stack([torch.stack([ab[env][i] for env in range(self.env.num_envs)]) for ab in self.action_buffers])  # dim: (length, channels)
            # Apply interpolation
            coeffs = natural_cubic_spline_coeffs(x, y)  
            spline = NaturalCubicSpline(coeffs)
            # Get the delayed action for each env by evaluating the spline at t 
            interpolated_actions.append(spline.evaluate(t)) # dim: Tensor(action_dim, num_envs)
        interpolated_actions = torch.stack(interpolated_actions).T  # Transpose to make it (num_envs, action_dim)

        return interpolated_actions

    def apply_LPF(self, actions):
        filtered_actions = (self.alpha * actions) + (1-self.alpha) * self.prev_actions 
        return filtered_actions

    def compute_friction(self, dq): # dq: shape (num_envs, action_dim), numpy array 
        tau_sticktion = self.Fs*self.softSign(dq, temperature=self.temperature)
        tau_viscose = self.mu_v*dq
        friction = torch.tensor(tau_sticktion+tau_viscose, dtype=torch.float32, device=self.env.device) # convert back to tensor
        return friction

    def softSign(self, u, temperature=0.1):
        return np.tanh(u/temperature)
    
    def set_params(self, delay=1, history_duration=10, min_data_points = 2, alpha=0.9, mu_v=0.1, Fs=0.3, temperature=0.1):
        """ Sets parameters of the actuator model 
        Args:
            delay (float): Controls the amount of delay applied to the actions.
            history_duration (float): Controls the number of actions used for interpolation.
            alpha (float): Controls the low pass filter.
            mu_v (float): Viscous friction coefficient.
            Fs (float): Static friction strength.
            temperature (float): parameter for the soft sign function.
        """
        self.delay = delay
        self.history_duration = history_duration
        self.min_data_points = min_data_points
        self.alpha = alpha
        self.mu_v = mu_v
        self.Fs = Fs
        self.temperature = temperature
