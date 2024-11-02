import gym

import os
import random
import time
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from texttable import Texttable

class RunningMeanStd(nn.Module):
    def __init__(self, shape = (), epsilon=1e-08):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.ones(()))

        self.epsilon = epsilon

    def forward(self, obs, update = True):
        if update:
            self.update(obs)

        return (obs - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, correction=0, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.running_mean, self.running_var, self.count = update_mean_var_count_from_moments(
            self.running_mean, self.running_var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Critic network (Value function V(s))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        # Actor network (Policy network π(a|s))
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        # Q-value network (Q(s,a))
        self.q_net = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + np.array(envs.single_action_space.shape).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        self.obs_rms = RunningMeanStd(shape=envs.single_observation_space.shape)
        self.value_rms = RunningMeanStd(shape=())

    def get_value(self, x):
        return self.critic(x)

    def get_q_value(self, x, action):
        x_action = torch.cat([x, action], dim=-1)
        return self.q_net(x_action)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        value = self.critic(x)
        q_value = self.get_q_value(x, action)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value, q_value

class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]

def monitor(episode_sums, iteration):
    """
    Display key-value pairs from episode_sums in a table format.

    Args:
        episode_sums (dict): A dictionary with elements as keys and tensors as values.
        iteration (int): The current iteration number.
    """
    print(f"============ iter {iteration} ============")

    # Initialize the table
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l", "r"])  # Left and right alignment for columns
    table.set_cols_dtype(['t', 'f'])  # Text and float data types

    # Prepare the header and data rows
    headers = ["Element", "Mean Value"]
    rows = []

    # Collect the keys and their mean values
    keys = list(episode_sums.keys())
    values = [v.mean().item() for v in episode_sums.values()]

    # Combine headers and data
    table_data = [headers] + list(zip(keys, values))

    # Add rows to the table
    table.add_rows(table_data)

    # Print the table
    print(table.draw())

def PPO_plus(cfg: DictConfig, envs):
    run_path = f"runs/{cfg['train']['params']['config']['name']}_{datetime.now().strftime('%d-%H-%M-%S')}"

    writer = SummaryWriter(run_path)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    OmegaConf.save(config = cfg, f = f"{run_path}/config.yaml")

    LEARNING_RATE = cfg["train"]["params"]["config"]["learning_rate"]
    NUM_STEPS = cfg["train"]["params"]["config"]["horizon_length"]
    NUM_ITERATIONS = cfg["train"]["params"]["config"]["max_epochs"]
    GAMMA = cfg["train"]["params"]["config"]["gamma"]
    GAE_LAMBDA = cfg["train"]["params"]["config"]["tau"]
    UPDATES_EPOCHS = cfg["train"]["params"]["config"]["mini_epochs"]
    MINIBATCH_SIZE = cfg["train"]["params"]["config"]["minibatch_size"]
    CLIP_COEF = cfg["train"]["params"]["config"]["e_clip"]
    ENT_COEF = cfg["train"]["params"]["config"]["entropy_coef"]
    VF_COEF = cfg["train"]["params"]["config"]["critic_coef"]
    MAX_GRAD_NORM = cfg["train"]["params"]["config"]["grad_norm"]
    NORM_ADV = cfg["train"]["params"]["config"]["normalize_advantage"]
    CLIP_VLOSS = cfg["train"]["params"]["config"]["clip_value"]
    ANNEAL_LR = True

    # Action improvement hyperparameters
    N = cfg["train"]["params"]["config"].get("N", 10)  # Number of perturbations
    sigma = cfg["train"]["params"]["config"].get("sigma", 0.1)  # Standard deviation of perturbations
    alpha = cfg["train"]["params"]["config"].get("alpha", 0.1)  # Learning rate for action update
    num_improvement_steps = cfg["train"]["params"]["config"].get("num_improvement_steps", 1)  # Number of improvement iterations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs = ExtractObsWrapper(envs)

    BATCH_SIZE = int(envs.num_envs * NUM_STEPS)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    obs = torch.zeros((NUM_STEPS, envs.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((NUM_STEPS, envs.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((NUM_STEPS, envs.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((NUM_STEPS, envs.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((NUM_STEPS, envs.num_envs), dtype=torch.float).to(device)
    true_dones = torch.zeros((NUM_STEPS, envs.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((NUM_STEPS, envs.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = agent.obs_rms(next_obs)
    next_done = torch.zeros(envs.num_envs, dtype=torch.float).to(device)
    next_true_done = torch.zeros(envs.num_envs, dtype=torch.float).to(device)

    for iteration in range(1, NUM_ITERATIONS + 1):
        if ANNEAL_LR:
            frac = 1.0 - (iteration - 1.0) / NUM_ITERATIONS
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, NUM_STEPS):
            global_step += envs.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            true_dones[step] = next_true_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

                # Action improvement loop
                for _ in range(num_improvement_steps):
                    # 2. Duplicate action N times
                    action_expanded = action.unsqueeze(1).expand(-1, N, -1)  # Shape: (num_envs, N, action_dim)

                    # 3. Sample perturbations noise_i ~ N(0, sigma^2 I)
                    noise = torch.randn_like(action_expanded) * sigma  # Shape: (num_envs, N, action_dim)

                    # 4. Compute perturbed actions a_i = a_t + noise_i
                    perturbed_actions = action_expanded + noise  # Shape: (num_envs, N, action_dim)

                    # 5. Compute Q(s_t, a_i)
                    s_t_expanded = next_obs.unsqueeze(1).expand(-1, N, -1)  # Shape: (num_envs, N, obs_dim)
                    s_t_flat = s_t_expanded.reshape(-1, s_t_expanded.shape[-1])  # Shape: (num_envs * N, obs_dim)
                    a_i_flat = perturbed_actions.reshape(-1, perturbed_actions.shape[-1])  # Shape: (num_envs * N, action_dim)
                    q_values = agent.get_q_value(s_t_flat, a_i_flat)  # Shape: (num_envs * N, 1)
                    q_values = q_values.view(-1, N)  # Shape: (num_envs, N)

                    # 6. Compute delta_a = alpha * (1/(N * sigma)) * sum_i Q(s_t, a_i) * noise_i
                    weighted_noise = q_values.unsqueeze(-1) * noise  # Shape: (num_envs, N, action_dim)
                    delta_a = alpha * (1 / (N * sigma)) * weighted_noise.sum(dim=1)  # Shape: (num_envs, action_dim)

                    # 7. Update action a_t = a_t + delta_a
                    action = action + delta_a

                # Recompute logprob of the improved action under the policy
                _, logprob, _, _, _ = agent.get_action_and_value(next_obs, action)

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, info = envs.step(action)
            next_obs = agent.obs_rms(next_obs)
            next_done = next_done.float()
            next_true_done = info["true_dones"].float()
            """if "time_outs" in info:
                if info["time_outs"].any():
                    print("time outs", info["time_outs"].sum())
                    exit(0)"""

        for el, v in zip(list(envs.episode_sums.keys())[:envs.numRewards], (torch.mean(envs.rew_mean_reset, dim=0)).tolist()):
            writer.add_scalar(f"reward/{el}", v, iteration)
        writer.add_scalar(f"reward/cum_rew", (torch.sum(torch.mean(envs.rew_cum_reset, dim=0))).item(), iteration)
        writer.add_scalar(f"reward/avg_rew", envs.terrain_levels.float().mean().item(), iteration)
        writer.add_scalar(f"reward/disc_cum_rew", (torch.mean(envs.cat_discounted_cum_reward_reset, dim=0)).item(), iteration)
        for el, v in zip(envs.cstr_manager.get_names(), (100.0 * torch.mean(envs.cstr_mean_reset, dim=0)).tolist()):
            writer.add_scalar(f"cstr/{el}", v, iteration)
        writer.add_scalar("cstr/curriculum", envs.constraints["curriculum"], iteration)
        if (iteration + 1) % 10 == 0:
            if hasattr(envs, 'terrain_levels'):
                writer.add_histogram("infos/terrain_levels", envs.terrain_levels, iteration)
        if hasattr(envs, 'episode_length_reset'):
            writer.add_scalar("infos/episode_length", torch.mean(envs.episode_length_reset, dim=0).item(), iteration)
        #monitor(envs.episode_sums, iteration)
        #for el, v in envs.episode_sums.items():
        #    writer.add_scalar(f"reward/{el}", v.mean().item(), iteration)

        # CaT: must compute the CaT quantity
        not_dones = 1.0 - dones
        rewards *= not_dones

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    true_nextnonterminal = 1 - next_true_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    true_nextnonterminal = 1 - true_dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal * true_nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * true_nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_values = agent.value_rms(b_values)
        b_returns = agent.value_rms(b_returns)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(UPDATES_EPOCHS):
            b_inds = torch.randperm(BATCH_SIZE, device=device)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, q_value = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                newvalue = agent.value_rms(newvalue, update = False)
                if CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -CLIP_COEF,
                        CLIP_COEF,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Q-value loss
                q_value = q_value.view(-1)
                q_value_loss = 0.5 * ((q_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF + q_value_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        if (iteration + 1) % 24 == 0:
            model_path = f"{run_path}/cleanrl_model.pt"
            torch.save(agent.state_dict(), model_path)
            print("Saved model")

def eval_PPO(cfg: DictConfig, envs):
    checkpoint = cfg["checkpoint"]
    actor_sd = torch.load(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs = ExtractObsWrapper(envs)

    actor = Agent(envs).to(device)
    actor.load_state_dict(actor_sd)

    obs = envs.reset()

    for _ in range(2000):
        with torch.no_grad():
            actions, _, _, _ = actor.get_action_and_value(actor.obs_rms(obs, update = False))

        next_obs, rewards, terminations, infos = envs.step(actions)
        obs = next_obs
