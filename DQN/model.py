import gym
gym.logger.set_level(40)

import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
import random


class DQN_MLP(nn.Module):
    def __init__(self, state_dim = 4, hidden_dim = 32, out_dim = 2):
        super(DQN_MLP, self).__init__()
        self.out_dim = out_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, state):
        intermediate = F.relu(self.fc1(state))
        out = F.softmax(self.fc2(intermediate), dim = 1)
        #action = torch.argmax(out, dim = 1)
        return out
    def act(self, state, epsilon = 0.1):
        assert state.size(0) == 1
        self.eval()
        if random.random() < epsilon:
            action = random.randint(0, self.out_dim-1)
        else:
            out = self.forward(state)
            action = torch.argmax(out, dim = 1).item()
        return action

class DQN_MLP_2(nn.Module):
    def __init__(self, state_dim = 4, hidden_dim = 32, out_dim = 2):
        super(DQN_MLP_2, self).__init__()
        self.out_dim = out_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
    def forward(self, state):
        intermediate = F.relu(self.fc1(state))
        intermediate = F.relu(self.fc2(intermediate))
        out = self.fc3(intermediate)
        return out
    def act(self, state, epsilon = 0.1):
        assert state.size(0) == 1
        self.eval()
        if random.random() < epsilon:
            action = random.randint(0, self.out_dim-1)
        else:
            out = self.forward(state)
            action = torch.argmax(out, dim = 1).item()
        return action