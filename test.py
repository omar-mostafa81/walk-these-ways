from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.actuator_model_wrapper import ActuatorModelWrapper
import matplotlib.pyplot as plt
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
import torch

config_go2(Cfg)
test_env = VelocityTrackingEasyEnv(cfg=Cfg, sim_device='cuda', headless=False)
env = ActuatorModelWrapper(test_env)

obs = env.reset()
print("Initial observation shape:", obs.shape)

for step in range(5):
    random_action = torch.randn(env.env.num_envs, env.env.action_space.shape[0])
    obs, reward, done, info = env.step(random_action)
    print(f"Step {step}: reward: {reward}, done: {done}")

    img = env.render('rgb_array')
    plt.imshow(img)
    plt.show()
