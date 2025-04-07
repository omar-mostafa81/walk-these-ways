import gym
import numpy as np
import time
import torch
from collections import deque
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

class ActuatorModelWrapper(gym.Wrapper):
    def __init__(self, env, delay = 1, history_duration = 10, alpha = 0.9, mu_v = 0.1, Fs = 0.3, Vs = 0.5, temperature=0.1):
        """ Initialize the actuator model wrapper
        Args:
            env: main environment
            delay (float): Controls the amount of delay applied to the actions.
            history_duration (float): Controls the number of actions used for interpolation.
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

        # Save actions/time for interpolation to simulate delay
        self.current_time = time.perf_counter() # More accurate for short durations
        self.history_duration = history_duration
        self.time_buffers = [deque() for _ in range(self.env.num_envs)]
        self.action_buffers = [deque() for _ in range(self.env.num_envs)]
        self.prev_actions = torch.zeros((self.env.num_envs, self.env.action_space.shape[0]), dtype=torch.float32, device=self.env.device)

        print("friction model is called")

    def reset(self):
        obs = super().reset() # LeggedRobot only has reset_idx(), which is called by VelocityTrackingEasyEnv with idx to all env
        
        # Reset buffers/variables
        self.time_buffers = [deque() for _ in range(self.env.num_envs)]
        self.action_buffers = [deque() for _ in range(self.env.num_envs)]
        self.prev_actions = torch.zeros((self.env.num_envs, self.env.action_space.shape[0]), dtype=torch.float32, device=self.env.device)

        return obs
    
    def step(self, actions):
        """ Simulate delay, bandwidth, and friction before sending actions to the simulator
        Args: actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Delay the actions
        print("Initial action is: ", actions[0]) # TO REMOVE
        self.current_time = time.perf_counter()
        for i in range(self.env.num_envs):
            self.time_buffers[i].append(self.current_time)
            self.action_buffers[i].append(actions[i])
        ## Remove old actions --> remove actions older than "history_duration" seconds
        for i in range(self.env.num_envs):
            while self.time_buffers[i] and self.time_buffers[i][0] < (self.current_time - self.history_duration):
                self.time_buffers[i].popleft()
                self.action_buffers[i].popleft()
        ## Get the delayed actions
        delayed_actions = self.apply_delay()
        print("Delayed action is: ", delayed_actions[0]) # TO REMOVE
        # Apply friction
        dq = self.env.dof_vel
        delayed_actions = delayed_actions - self.compute_friction(dq) 
        # Apply LowPassFilter based on bandwidth
        filtered_actions = self.apply_LPF(delayed_actions)
        print("Filtered action is: ", filtered_actions[0]) # TO REMOVE

        # Return modified actions to the env
        self.prev_actions = filtered_actions
        final_actions = torch.tensor(filtered_actions, dtype=torch.float32, device=self.env.device)
        return self.env.step(final_actions)
    
    def apply_delay(self):
        # Build tensors
        x = torch.tensor([list(tb) for tb in self.time_buffers], dtype=torch.float32)  # (num_envs, K)
        y = torch.tensor([list(ab) for ab in self.action_buffers], dtype=torch.float32)  # (num_envs, K, action_dim)
        t = torch.full((self.env.num_envs,), self.current_time - self.delay, dtype=torch.float32)  # target eval time for each env

        coeffs = natural_cubic_spline_coeffs(x, y)  # shape: (num_envs, K, action_dim)
        # Apply interpolation 
        spline = NaturalCubicSpline(coeffs)
        interpolated_actions = spline.evaluate(t)   # shape: (num_envs, action_dim)
        return interpolated_actions

    def apply_LPF(self, actions):
        filtered_actions = (self.alpha * actions) + (1-self.alpha) * self.prev_actions 
        return filtered_actions

    def compute_friction(self, dq):
        tau_sticktion = self.Fs*self.softSign(dq, temperature=self.temperature)
        tau_viscose = self.mu_v*dq
        return tau_sticktion+tau_viscose

    def softSign(self, u, temperature=0.1):
        return np.tanh(u/temperature)
    
    def set_params(self, delay=1, history_duration=10, alpha=0.9, mu_v=0.1, Fs=0.3, temperature=0.1):
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
        self.alpha = alpha
        self.mu_v = mu_v
        self.Fs = Fs
        self.temperature = temperature