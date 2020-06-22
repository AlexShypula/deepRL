import gym
gym.logger.set_level(40)
import numpy as np
from collections import deque
import torch
torch.manual_seed(42)


class Simulation():
    def __init__(self, environment = "CartPole-v0"):
        self.env = gym.make(environment)
        self.env.seed(0)
    def reset(self):
        observation = self.env.reset()
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        return observation
    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        observaiton = torch.from_numpy(observation).float().unsqueeze(0)
        return observaiton, reward, is_done, info

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class NetworkParamSampler:
    def __init__(self, network, params_to_sample=200, cache_size=100, monitor_var=False):
        self.monitor_var = monitor_var
        if self.monitor_var:
            self.running_var_stats = []

        self.variance_deque = deque(maxlen=cache_size)

        self.n_params = count_parameters(network)
        self.sample_prob = params_to_sample / self.n_params
        self.sample_indices = []
        for param in network.parameters():
            indices = (torch.rand_like(param) <= self.sample_prob).nonzero()
            self.sample_indices.append(indices)

    def sample(self, network):
        param_sample = []
        for param, indices in zip(network.parameters(), self.sample_indices):
            if param.requires_grad:
                for i in indices:
                    param_sample.append(param.grad[tuple(i)].item())
        self.variance_deque.append(np.array(param_sample))
        if self.monitor_var and len(self.variance_deque) > 1:
            self.running_var_stats.append(self.get_running_var())

    def get_running_std(self):
        if len(self.variance_deque) > 1:
            stacked = np.stack(self.variance_deque, axis=0)
            std_arr = np.std(stacked, axis=0)
            return std_arr.mean()
        else:
            return 0

    def get_running_var(self):
        if len(self.variance_deque) > 1:
            stacked = np.stack(self.variance_deque, axis=0)
            var_arr = np.var(stacked, axis=0)
            return var_arr.mean()
        else:
            return 0
