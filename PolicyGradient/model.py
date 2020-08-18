import gym
gym.logger.set_level(40)

import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, device, state_dim=4, hidden_dim=32, out_dim=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.device = device

    def forward(self, state):
        intermediate = F.relu(self.fc1(state))
        out = F.softmax(self.fc2(intermediate), dim=1)
        return out

    def act(self, state):
        # unsqueeze will add a dimension at position 0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), m.entropy()

class Critic(nn.Module):
    def __init__(self, device,  state_dim = 4, hidden_dim = 32, out_dim = 1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.device = device
    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, device, method: str, state_dim = 4, hidden_dim = 32, policy_out_dim = 2, critic_out_dim = 1):
        super(ActorCritic, self).__init__()
        self.policy = Policy(device=device, state_dim=state_dim, hidden_dim=hidden_dim, out_dim=policy_out_dim)
        if method.lower() in ("actor_critic" or "critic_baseline"):
            self.has_critic = True
            self.critic = Critic(device=device, state_dim=state_dim, hidden_dim=hidden_dim, out_dim=critic_out_dim)
        else:
            self.has_critic = False
    def act(self, state):
        return self.policy.act(state)
    def value(self, state):
        assert self.has_critic, "The Critic is called, but it is not initialized"
        return self.critic(state)