import gym
gym.logger.set_level(40)
import numpy as np
from collections import deque

import torch
torch.manual_seed(42)
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import random
from DQN.utils import Simulation, NetworkParamSampler
from torch.utils.tensorboard import SummaryWriter
from itertools import islice


class DQN_Trainer:
    def __init__(self, q_net,
                 optimizer,
                 n_episodes=4000,
                 n_episodes_annealing = 3000,
                 max_t=200,
                 replay_buffer_size = 500,
                 batch_size=16,
                 gamma = 0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 polyak_factor = 0.999,
                 device = "cpu",
                 environmnet = "CartPole-v0",
                 loss = "mse",
                 early_stop_score = float("inf"),
                 metrics_buffer_size=500,
                 scores_buffer_size=100,
                 network_params_to_sample = 200,
                 logdir = "tensorboard",
                 **kwargs
                 ):
        self.update_no = 0
        self.q_net = q_net
        self.target_q_net = deepcopy(q_net)
        self.target_q_net.eval()
        self.optimizer = optimizer
        assert n_episodes >= n_episodes_annealing, "need to train for more or equal episodes than are used for annealing"
        assert epsilon_start >= epsilon_end, "epsilon should not increase over time"
        self.n_episodes = n_episodes
        self.n_episodes_annealing = n_episodes_annealing
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step_size = (epsilon_start - epsilon_end) / n_episodes_annealing
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        assert metrics_buffer_size >= max_t, "metrics buffer must be larger or equal to max episode length"
        self.metrics_buffer_size = metrics_buffer_size
        self.scores_buffer_size = scores_buffer_size
        self.max_t = max_t
        self.polyak_factor = polyak_factor
        self.device = device
        self.early_stop_score = early_stop_score
        self.network_params_to_sample = network_params_to_sample

        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.rewards_buffer = deque(maxlen=self.metrics_buffer_size)
        self.q_value_buffer = deque(maxlen=self.metrics_buffer_size)
        self.scores_buffer = deque(maxlen=self.scores_buffer_size)

        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "huber":
            self.loss = nn.SmoothL1Loss()
        elif loss == "mae":
            self.loss = nn.L1Loss()
        else:
            Exception(f"loss should be mse, huber, or mae, it is: {loss}")

        self.q_net_sampler = NetworkParamSampler(network=self.q_net,
                                                 params_to_sample=self.network_params_to_sample,
                                                 cache_size=self.metrics_buffer_size,
                                                 monitor_var=False)
        self.tgt_net_sampler = NetworkParamSampler(network=self.target_q_net,
                                                   params_to_sample=self.network_params_to_sample,
                                                   cache_size=self.metrics_buffer_size,
                                                   monitor_var=False)
        self.env = Simulation(environmnet)
        self.writer = SummaryWriter(logdir)

        #TODO: n_episodes, max_t, gamma_
    def _update_q_net(self, batch_size = None, clip_grad = True):
        batch_size = batch_size if batch_size else self.batch_size
        if len(self.replay_buffer) < batch_size:
            batch_size = len(self.replay_buffer)
        self.q_net.train()
        sample = random.sample(self.replay_buffer, batch_size)

        states = torch.cat([observation.get("state") for observation in sample], dim = 0).to(self.device)
        next_states = torch.cat([observation.get("next_state") for observation in sample], dim = 0).to(self.device)
        actions = torch.tensor([observation.get("action") for observation in sample]).unsqueeze(1).to(self.device)
        rewards = torch.tensor([observation.get("reward") for observation in sample]).unsqueeze(1).to(self.device)

        q_fitted = self.q_net(states)
        # max returns a tuple of the max and the indices of the max values; need to also add back the column dim
        q_next_fitted = self.target_q_net(next_states).max(dim=1)[0].unsqueeze(1)
        action_mask = torch.zeros_like(q_fitted).scatter_(1, actions, 1.).byte().to(self.device)
        # max returns a tuple of the max and the indices of the max values
        avg_q_val = q_fitted.max(dim=1)[0].mean().item()
        avg_tgt_val = q_next_fitted.mean().item()

        q_fitted_for_actions = q_fitted*action_mask
        q_approx = rewards + self.gamma * q_next_fitted
        q_approx_for_actions = q_approx.repeat(1, q_fitted.size(1))*action_mask

        loss = self.loss(q_fitted_for_actions, q_approx_for_actions)
        self.optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            for param in self.q_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self._update_target_q_net()

        return loss.detach(), avg_q_val, avg_tgt_val

    def _step_epsilon(self, episode_no: int):

        if episode_no < self.n_episodes_annealing:
            self.epsilon-=self.epsilon_step_size

    def _step_log(self, loss, q_value, tgt_value):

        self.q_value_buffer.append(q_value)
        traling_q_val = np.mean(self.q_value_buffer)
        trailing_reward = np.mean(self.rewards_buffer)

        self.q_net_sampler.sample(self.q_net)
        self.tgt_net_sampler.sample(self.target_q_net)
        q_net_grad_var = self.q_net_sampler.get_running_var()
        tgt_net_grad_var = self.tgt_net_sampler.get_running_var()

        self.writer.add_scalar('Train/loss', loss, self.update_no)
        self.writer.add_scalar('Train/q_value', q_value, self.update_no)
        self.writer.add_scalar('Train/tgt_value', tgt_value, self.update_no)

        self.writer.add_scalar('Train/trailing_q_value', traling_q_val, self.update_no)
        self.writer.add_scalar('Train/trailing_reward', trailing_reward, self.update_no)

        self.writer.add_scalar('Train/q_net_grad_var', q_net_grad_var, self.update_no)
        self.writer.add_scalar('Train/tgt_net_grad_var', tgt_net_grad_var, self.update_no)

        self.update_no += 1

    def _process_episode(self, episode_no: int, episode_length: int, print_every = 100):

        buffer_size = len(self.rewards_buffer)
        R = sum(islice(self.rewards_buffer, buffer_size-episode_length, buffer_size))
        self.scores_buffer.append(R)

        self.writer.add_scalar('Episode/score', R, episode_no)
        self.writer.add_scalar('Episode/trailing_score', np.mean(self.scores_buffer), episode_no)

        if episode_no % print_every == 0:
            print(f"Episode {episode_no}\t Avg Score: {np.mean(self.scores_buffer):.2f}")

        if np.mean(self.scores_buffer) >= self.early_stop_score:
            print(f"Environment solved in {episode_no - 1} with the avg score of last"
                  f"{self.scores_buffer_size} episodes as {np.mean(self.scores_buffer)}")
            return True
        else:
            return False

    def _update_target_q_net(self):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.polyak_factor * param.data + target_param.data * (1.0 - self.polyak_factor))

    def train(self, print_every = 100):
        for episode_no in tqdm(range(1, self.n_episodes + 1), position = 0, smoothing = 0):
            state = self.env.reset()
            for t in range(self.max_t):
                action = self.q_net.act(state.to(self.device), epsilon=self.epsilon)
                next_state, reward, is_finished, _ = self.env.step(action)
                self.rewards_buffer.append(reward)
                self.replay_buffer.append({"state": state,
                                           "action": action,
                                           "reward": reward if not is_finished else -reward,
                                           "next_state": next_state})
                loss, avg_q_val, avg_tgt_val = self._update_q_net()
                self._step_log(loss, avg_q_val, avg_tgt_val)

                if is_finished:
                    break
                else:
                    state = next_state

            # end of episode
            self._step_epsilon(episode_no)
            early_stop = self._process_episode(episode_no = episode_no,
                                               episode_length = t,
                                               print_every = print_every)

            if early_stop:
                break



