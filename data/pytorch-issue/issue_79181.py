import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)
device = torch.device('mps')

model = PPO("MlpPolicy", env, device=device, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")